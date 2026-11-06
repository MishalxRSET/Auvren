/**
 * Auvren — Main entry point.
 *
 * Wires together the orb visualization, WebSocket communication,
 * speech recognition, and audio playback into a single experience.
 */

import { createOrb, type OrbState } from "./orb";
import { createVoiceInput, createAudioPlayer, createMicMonitor } from "./voice";
import { createSocket } from "./ws";
import { openSettings, checkFirstTimeSetup } from "./settings";
import { playIntro } from "./intro";
import "./style.css";

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

type State = "idle" | "listening" | "thinking" | "speaking" | "compacting";
let currentState: State = "idle";
let isMuted = false;

// The wake word (server-side; see wakeword.py). While asleep the orb rests at
// "idle" and the status line says what to say. The server sends the state in
// the `config` frame and a `wake` frame whenever it changes.
let wakeEnabled = false;
let awake = true;
let wakePhrase = "Hey Auvren";

// Who turns speech into text. "local": the server listens on this machine
// (stt.py) and Chrome's recogniser stays off — otherwise every request would
// arrive twice. "browser": Chrome's Web Speech API, as before.
let sttMode: "local" | "browser" = "browser";
let voiceStarted = false;
let kickedOff = false;

function setSttMode(mode: unknown) {
  sttMode = mode === "local" ? "local" : "browser";
  if (sttMode === "local") {
    if (voiceStarted) voiceInput.stop();
    voiceStarted = false;
  } else if (kickedOff && !voiceStarted && !isMuted) {
    startBrowserListening();
  }
}

/** Chrome's recogniser, and the level meter that watches it. Only in
 *  "browser" mode: on-device the page never touches the microphone, so a
 *  window opened by the wake word does not stop to ask for it. */
function startBrowserListening() {
  if (!micMonitor) micMonitor = createMicMonitor(onMicLevel, onMicEvent);
  voiceInput.start();
  voiceStarted = true;
}

/** Where the orb settles when nothing is happening. */
function restingState(): State {
  return isMuted || (wakeEnabled && !awake) ? "idle" : "listening";
}

const statusEl = document.getElementById("status-text")!;
const errorEl = document.getElementById("error-text")!;

function showError(msg: string) {
  errorEl.textContent = msg;
  errorEl.style.opacity = "1";
  setTimeout(() => {
    errorEl.style.opacity = "0";
  }, 5000);
}

function updateStatus(state: State) {
  const asleep = wakeEnabled && !awake && !isMuted;
  const labels: Record<State, string> = {
    idle: asleep ? `say “${wakePhrase}”` : "",
    listening: "listening...",
    thinking: "thinking...",
    speaking: "",
    compacting: "",          // the notice banner carries the words; the orb carries the state
  };
  statusEl.textContent = labels[state];
}

// ---------------------------------------------------------------------------
// Init components
// ---------------------------------------------------------------------------

const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const orb = createOrb(canvas);

const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${wsProto}//${window.location.host}/ws/voice`;
const socket = createSocket(WS_URL);

const audioPlayer = createAudioPlayer();
orb.setAnalyser(audioPlayer.getAnalyser());

let muteMicDuringSpeech = false;

function transition(newState: State) {
  if (newState === currentState) return;
  currentState = newState;
  const btn = document.getElementById("hush");
  if (btn) (btn as HTMLButtonElement).hidden = newState !== "speaking";
  orb.setState(newState as OrbState);
  updateStatus(newState);

  if (isMuted) return;
  if (newState === "speaking" && muteMicDuringSpeech) {
    voiceInput.pause();
  } else {
    voiceInput.resume();
  }
}

// ---------------------------------------------------------------------------
// Voice input
// ---------------------------------------------------------------------------

const voiceInput = createVoiceInput(
  (text: string) => {
    // The server decides whether this is echo, a barge-in, or a new turn.
    micMonitor?.sawSpeech();
    socket.send({ type: "transcript", text, isFinal: true });
  },
  (text: string) => {
    micMonitor?.sawSpeech();
    socket.send({ type: "interim", text });
  },
  (msg: string) => {
    showError(msg);
  },
  (event: string) => {
    // Mirror the recogniser's lifecycle to the server log. Going deaf is a
    // browser-side failure the server cannot otherwise see at all, and the
    // console it used to be confined to is never open when it happens.
    socket.send({ type: "mic", text: event });
  }
);

// A live meter for the microphone itself. If this moves when you speak, the
// microphone is working — whatever else is or is not happening. It answers
// "is it even hearing me?" without a log, a console or anyone to ask.
const micDot = document.createElement("div");
micDot.id = "mic-level";
micDot.title = "microphone input";
document.body.appendChild(micDot);

function onMicLevel(level: number) {
  const pct = Math.min(100, Math.round(level * 900));
  micDot.style.setProperty("--level", `${pct}%`);
  micDot.classList.toggle("is-hot", level > 0.02);
}

function onMicEvent(event: string) {
  socket.send({ type: "mic", text: event });
  // Proven deaf: sound going in, nothing coming out. Do not wait for the
  // rotation timer to happen along — measured once at 21 seconds, all of
  // it lost. Rebuild the recogniser now.
  if (event.startsWith("DEAF")) voiceInput.restart("deaf: audio in, no results");
}

let micMonitor: ReturnType<typeof createMicMonitor> | null = null;

// ── stopping him ──────────────────────────────────────────────────────────
// Escape, or the button that appears while he is talking. Not a spoken word:
// his voice comes back through the microphone garbled, and a mis-hear that
// looked like "stop" would cut him off at random. A keystroke cannot be
// misheard.
const hushBtn = document.createElement("button");
hushBtn.id = "hush";
hushBtn.type = "button";
hushBtn.textContent = "Stop";
hushBtn.title = "Stop speaking (Esc)";
hushBtn.hidden = true;
document.body.appendChild(hushBtn);

function hush() {
  if (currentState !== "speaking") return;
  // Locally first: the round trip is real and silence should be instant.
  audioPlayer.stop();
  socket.send({ type: "hush" });
  transition(restingState());
}

hushBtn.addEventListener("click", hush);
window.addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Escape") { e.preventDefault(); hush(); }
});

audioPlayer.onPlayed((utt, idx) => {
  socket.send({ type: "played", utt, idx });
});

// End of speech is the server's call (`status: idle` after every chunk is
// acked); a transient empty queue mid-utterance must not flip the UI.
audioPlayer.onFinished(() => {});

audioPlayer.onNeedsGesture(() => {
  showError("Click anywhere to enable audio");
});

// ---------------------------------------------------------------------------
// The intro, and being summoned
// ---------------------------------------------------------------------------

// The wake word opened this window (the server launched it with ?summoned=1).
// Once the intro has played AND the server is connected, say so: the server
// then asks "Yes, sir?" — or answers what was said in the same breath as the
// wake word — instead of greeting.
const summoned = new URLSearchParams(window.location.search).has("summoned");
let introDone = false;
let serverReady = false;
let helloSent = false;

function maybeHello() {
  if (!summoned || helloSent || !introDone || !serverReady) return;
  helloSent = true;
  socket.send({ type: "hello", summoned: true });
}

playIntro(summoned).then(() => {
  introDone = true;
  maybeHello();
});

// ---------------------------------------------------------------------------
// The wake word
// ---------------------------------------------------------------------------

function applyWake(w: Record<string, unknown>, cue: boolean) {
  wakeEnabled = Boolean(w.enabled);
  awake = !wakeEnabled || Boolean(w.awake);
  if (typeof w.phrase === "string" && w.phrase) wakePhrase = w.phrase;
  if (cue && awake) wakeChime();
  if (currentState === "idle" || currentState === "listening") {
    transition(restingState());
    updateStatus(currentState);   // the hint changes even when the state does not
  }
}

/** A short rising chirp: "I heard you", without words to talk over. */
function wakeChime() {
  const ctx = audioPlayer.getAnalyser().context as AudioContext;
  if (ctx.state !== "running") return;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const t = ctx.currentTime;
  osc.type = "sine";
  osc.frequency.setValueAtTime(880, t);
  osc.frequency.exponentialRampToValueAtTime(1320, t + 0.12);
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.06, t + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 0.2);
}

// ---------------------------------------------------------------------------
// WebSocket messages
// ---------------------------------------------------------------------------

socket.onMessage((msg) => {
  const type = msg.type as string;

  if (type === "config") {
    muteMicDuringSpeech = Boolean(msg.muteMicDuringSpeech);
    if (msg.wake && typeof msg.wake === "object") applyWake(msg.wake as Record<string, unknown>, false);
    setSttMode(msg.stt);
    serverReady = true;
    maybeHello();
  } else if (type === "stt") {
    setSttMode(msg.mode);
  } else if (type === "wake") {
    applyWake(msg, Boolean(msg.cue));
  } else if (type === "audio") {
    const data = msg.data as string;
    if (data) {
      if (currentState !== "speaking") transition("speaking");
      audioPlayer.enqueue(data, Number(msg.utt), Number(msg.idx));
    }
    if (msg.text) console.log("[Auvren]", msg.text);
  } else if (type === "stop") {
    audioPlayer.stop();
    transition(restingState());
  } else if (type === "drop_queued") {
    audioPlayer.dropQueued();
  } else if (type === "status") {
    const state = msg.state as string;
    if (state === "thinking") transition("thinking");
    else if (state === "speaking") transition("speaking");
    else if (state === "compacting") transition("compacting");
    else if (state === "idle") transition(restingState());
  } else if (type === "text") {
    // A chunk TTS could not voice: show it instead of losing it
    console.log("[Auvren]", msg.text);
    statusEl.textContent = String(msg.text);
  } else if (type === "notice") {
    // Shown, never spoken. The server sends one when it is about to be busy
    // for a few seconds (a context rotation), and an empty string to clear it.
    // Without it the pause looks like a crash.
    const text = String(msg.text ?? "");
    statusEl.textContent = text;
    if (text) console.log("[notice]", text);
  }
});

// ---------------------------------------------------------------------------
// Kick off
// ---------------------------------------------------------------------------

// Start listening after a brief delay for the orb to render
setTimeout(() => {
  kickedOff = true;
  if (sttMode === "browser" && !voiceStarted) startBrowserListening();
  if (currentState !== "speaking") transition(restingState());
}, 1000);

// Resume AudioContext on ANY user interaction (browser autoplay policy)
function ensureAudioContext() {
  const ctx = audioPlayer.getAnalyser().context as AudioContext;
  if (ctx.state === "suspended") {
    ctx.resume().then(() => console.log("[audio] context resumed"));
  }
}
document.addEventListener("click", ensureAudioContext);
document.addEventListener("touchstart", ensureAudioContext);
document.addEventListener("keydown", ensureAudioContext, { once: true });

// Try to resume audio context on load
ensureAudioContext();

// ---------------------------------------------------------------------------
// UI Controls
// ---------------------------------------------------------------------------

const btnMute = document.getElementById("btn-mute")!;
const btnMenu = document.getElementById("btn-menu")!;
const menuDropdown = document.getElementById("menu-dropdown")!;
const btnRestart = document.getElementById("btn-restart")!;
const btnFixSelf = document.getElementById("btn-fix-self")!;

btnMute.addEventListener("click", (e) => {
  e.stopPropagation();
  isMuted = !isMuted;
  btnMute.classList.toggle("muted", isMuted);
  // The server's own microphone (on-device ears, wake engine) mutes too.
  socket.send({ type: "mute", muted: isMuted });
  if (isMuted) {
    voiceInput.pause();
    transition("idle");
  } else {
    voiceInput.resume();
    transition(restingState());
  }
});

btnMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = menuDropdown.style.display === "none" ? "block" : "none";
});

document.addEventListener("click", () => {
  menuDropdown.style.display = "none";
});

btnRestart.addEventListener("click", async (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  statusEl.textContent = "restarting...";
  try {
    await fetch("/api/restart", { method: "POST" });
    // Wait a few seconds then reload
    setTimeout(() => window.location.reload(), 4000);
  } catch {
    statusEl.textContent = "restart failed";
  }
});

btnFixSelf.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  // Activate work mode on the WebSocket session (Auvren becomes Claude Code's voice)
  // Milestone 1 has no tools yet; "Fix yourself" returns as a brain tool later.
  statusEl.textContent = "fix-yourself is not available in this build";
});

// Settings button
const btnSettings = document.getElementById("btn-settings")!;
btnSettings.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  openSettings();
});

// First-time setup detection — check after a short delay for server readiness
setTimeout(() => {
  checkFirstTimeSetup();
}, 2000);
