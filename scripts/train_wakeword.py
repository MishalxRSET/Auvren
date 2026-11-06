"""Train the "Hey Auvren" wake-word model for openWakeWord.

    python scripts/train_wakeword.py all --work <dir> --out models/wakeword/hey_auvren_v0.1.onnx

Stages (each resumable; `all` runs them in order):

  download   Piper voices (many speakers), openWakeWord's feature models, a
             sample of its 2,000-hour ACAV100M negative feature set (fetched
             as slices by HTTP range, not the 17 GB whole), and its
             validation set for measuring false wakes per hour.
  synth      Piper speaks the wake phrase -- spelt "Hey Ovren", which Piper
             says OV-ren, /ˈɒvɹən/ -- in thousands of voices, plus hard
             negatives that sound close ("hey oven", "hey Owen", "heaven")
             and ordinary sentences. Every 10th speaker of each multi-speaker
             voice, and two whole voices, are held out for testing.
  features   Each clip is placed in a 2 s window ending just after the phrase,
             augmented (room noise, babble, reverb, microphone colouring,
             level) and turned into openWakeWord's 16 x 96 embedding frames.
  train      A small network (1536 -> 64 -> 32 -> 1) in numpy, two rounds
             (the second weighting the negatives the first got most wrong),
             exported to ONNX with openWakeWord's own input/output shape:
             float[batch, 16, 96] -> float[batch, 1]. Input normalisation is
             folded into the first layer, so the file is self-contained.
  eval       Recall on held-out voices, false wakes per hour on the
             validation set, and the model run through the REAL openWakeWord
             runtime on streamed speech.

Your own voice makes the biggest difference: record 20-50 clips of you
saying "Hey Auvren" (16 kHz mono WAV, one per file) and pass
`--my-recordings <dir>`; a fifth are held out to measure on.

Requirements (a separate environment is wise): requirements-train.txt.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
import urllib.request
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

SR = 16000
CLIP_SAMPLES = 2 * SR             # 2.0 s -> 16 embedding frames
FRAMES = 16
EMBED = 96

HF = "https://huggingface.co"
VOICE_BASE = f"{HF}/rhasspy/piper-voices/resolve/main/en"
OWW_FEATURES = f"{HF}/datasets/davidscripka/openwakeword_features/resolve/main"
ACAV_FILE = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
ACAV_ROWS = 5_625_000
ROW_BYTES = FRAMES * EMBED * 2    # float16
NPY_OFFSET = 128

# (voice, held-out?) -- multi-speaker voices hold out every 10th speaker.
VOICES = [
    ("en_US-libritts_r-medium", False), ("en_GB-vctk-medium", False),
    ("en_US-l2arctic-medium", False), ("en_GB-aru-medium", False),
    ("en_GB-semaine-medium", False), ("en_US-arctic-medium", False),
    ("en_GB-jenny_dioco-medium", False), ("en_GB-cori-medium", False),
    ("en_US-ryan-medium", False), ("en_US-amy-medium", False),
    ("en_GB-alan-medium", True), ("en_GB-northern_english_male-medium", True),
]

POSITIVES = [("Hey Ovren", 45), ("Hey, Ovren.", 20), ("Hey Ovren!", 10),
             ("Hey Ovren?", 10), ("Hey Ovrin", 8), ("Hey Ovren,", 7)]

ADVERSARIAL = [
    "Hey oven", "Hey Owen", "Hey over", "Hey even", "Hey Evan", "Hey Aaron",
    "Hey, Robin", "Hey Kevin", "Hey Devon", "Hey Ivan", "Hey everyone",
    "Hey, Orange", "Hey, Oliver", "Hey Olivia", "Hey Jarvis", "Hey Siri",
    "Hey Google", "Alexa", "Heaven", "Seven", "Eleven", "Oven", "Ovaltine",
    "Hovering", "Covering", "Governor", "Over and out", "Oh when", "Of them",
    "A vernal", "Hey, of when", "Hey, all of them", "Hey, overnight", "Hey you",
    "Hey there", "Hey man", "Hey, honey", "Hey, buddy", "Hey, brother",
    "Hey, oven's on", "Hey, open it", "Open the oven", "Ovens", "Hey Raven",
    "Hey Steven", "Hey Gavin", "Hey Bren", "Hey Glen", "Hey Ren", "Hey, Lauren",
    "Hey Warren", "Hey Owen, over here", "Hey, never mind", "Hey, all right",
    "Okay then", "Hey, how's it going", "Ever been", "Hovercraft", "Overrun",
]

SUBJECTS = ["I", "We", "My brother", "The teacher", "Someone", "You", "They",
            "My friend", "The lecturer", "Our team", "The doctor", "She", "He"]
VERBS = ["need to finish", "forgot about", "are talking about", "finally fixed",
         "should check", "will submit", "can't find", "just started", "keep missing",
         "want to watch", "have to call", "were reading", "never liked"]
OBJECTS = ["the assignment", "the data structures exam", "the electronics lab",
           "the Java project", "the cricket match", "dinner tonight", "the bus",
           "the new phone", "the hostel wifi", "the train tickets", "the notes",
           "the circuit diagram", "the weather", "the movie", "the library books"]
TAILS = ["before Friday.", "this evening.", "right now.", "again.", "tomorrow morning.",
         "after class.", "at the canteen.", "for the second time.", "with everyone.", ""]
LINES = [
    "What time does the shop close today?", "Can you pass me the charger?",
    "The over was bowled in under four minutes.", "Put the pizza in the oven for ten minutes.",
    "Owen said he would be late again.", "Heaven knows how long this will take.",
    "Seven people turned up for the meeting.", "Turn the volume down a little.",
    "I think the answer is eleven.", "Have you seen my blue notebook anywhere?",
    "Let's go over the slides once more.", "Evan and Aaron are coming over later.",
    "The professor covered recursion and linked lists.", "Open the window, it's warm in here.",
    "Hey, did you finish the lab report?", "Hey guys, the bus is here.",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --- download ------------------------------------------------------------------------------

def fetch(url: str, dest: Path, byte_range: tuple[int, int] | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "auvren-wakeword-trainer"})
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def stage_download(work: Path, acav_rows: int, acav_chunks: int) -> None:
    (work / "voices").mkdir(parents=True, exist_ok=True)
    for name, _ in VOICES:
        lang, voice, quality = name.split("-")
        for ext in (".onnx", ".onnx.json"):
            fetch(f"{VOICE_BASE}/{lang}/{voice}/{quality}/{name}{ext}",
                  work / "voices" / f"{name}{ext}")
        log("voice", name)

    oww = work / "oww"
    oww.mkdir(exist_ok=True)
    for f in ("melspectrogram.onnx", "embedding_model.onnx"):
        fetch(f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{f}", oww / f)
    fetch(f"{OWW_FEATURES}/validation_set_features.npy", work / "validation_set_features.npy")
    log("feature models and validation set")

    # A spread of slices across the whole 2,000 hours, not its first hours.
    out = _acav_path(work, acav_rows)
    if not out.exists():
        per = acav_rows // acav_chunks
        parts = []
        for i in range(acav_chunks):
            start = int(i * (ACAV_ROWS - per) / max(1, acav_chunks - 1))
            lo = NPY_OFFSET + start * ROW_BYTES
            hi = lo + per * ROW_BYTES - 1
            part = work / f"acav_{acav_rows}_{i:03d}.bin"
            fetch(f"{OWW_FEATURES}/{ACAV_FILE}", part, (lo, hi))
            parts.append(part)
            log(f"acav slice {i + 1}/{acav_chunks}")
        arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                        shape=(per * acav_chunks, FRAMES, EMBED))
        for i, part in enumerate(parts):
            arr[i * per:(i + 1) * per] = np.frombuffer(part.read_bytes(), dtype=np.float16) \
                .reshape(per, FRAMES, EMBED)
            part.unlink()
        arr.flush()
    log("acav sample ready")


def _acav_path(work: Path, rows: int) -> Path:
    return work / f"acav_{rows}.npy"


def _acav(work: Path) -> np.ndarray:
    """The largest ACAV sample downloaded."""
    found = sorted(work.glob("acav_*.npy"), key=lambda p: p.stat().st_size)
    if not found:
        raise SystemExit("no ACAV sample: run the download stage")
    return np.load(found[-1], mmap_mode="r")


# --- synthesis ------------------------------------------------------------------------------

def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if sr == SR:
        return x
    f = Fraction(SR, sr)
    return resample_poly(x, f.numerator, f.denominator)


def _trim(x: np.ndarray, margin: int = int(0.03 * SR)) -> np.ndarray:
    """Cut leading and trailing silence, so a clip ENDS where the speech does."""
    a = np.abs(x)
    thr = max(a.max() * 0.04, 1e-4)
    idx = np.nonzero(a > thr)[0]
    if idx.size == 0:
        return x
    return x[max(0, idx[0] - margin): idx[-1] + margin]


_VOICE_CACHE: dict = {}


def _one_thread_per_worker() -> None:
    """Each Piper session otherwise takes every core, and N workers on N
    cores then spend their time fighting (measured: load 12 on 4 cores).
    Piper builds its session with onnxruntime.SessionOptions(), so wrap it."""
    import onnxruntime
    if getattr(onnxruntime, "_one_thread", False):
        return
    original = onnxruntime.SessionOptions

    def options(*args, **kwargs):
        o = original(*args, **kwargs)
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        return o
    onnxruntime.SessionOptions = options
    onnxruntime._one_thread = True


def _speak(work: Path, voice: str, speaker: int | None, text: str, rng: random.Random):
    from piper import PiperVoice, SynthesisConfig
    v = _VOICE_CACHE.get(voice)
    if v is None:
        _one_thread_per_worker()
        v = _VOICE_CACHE[voice] = PiperVoice.load(str(work / "voices" / f"{voice}.onnx"))
    cfg = SynthesisConfig(speaker_id=speaker, length_scale=rng.uniform(0.75, 1.35),
                          noise_scale=rng.uniform(0.4, 0.9), noise_w_scale=rng.uniform(0.5, 1.0))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        v.synthesize_wav(text, w, syn_config=cfg)
    buf.seek(0)
    with wave.open(buf) as r:
        sr = r.getframerate()
        x = np.frombuffer(r.readframes(r.getnframes()), np.int16).astype(np.float32) / 32768
    return _trim(_resample(x, sr)).astype(np.float32)


def _speakers(work: Path, voice: str) -> int:
    cfg = json.loads((work / "voices" / f"{voice}.onnx.json").read_text("utf-8"))
    return int(cfg.get("num_speakers", 1) or 1)


def _jobs(work: Path, kind: str, n: int, seed: int):
    """(voice, speaker, text, split) for n clips, voices weighted by speakers."""
    rng = random.Random(seed)
    pool = []
    for voice, held_out in VOICES:
        for s in range(_speakers(work, voice)):
            split = "test" if held_out or (s % 10 == 0 and _speakers(work, voice) > 1) else "train"
            pool.append((voice, s if _speakers(work, voice) > 1 else None, split))
    texts = {"pos": POSITIVES, "adv": [(t, 1) for t in ADVERSARIAL]}.get(kind)
    jobs = []
    for i in range(n):
        voice, spk, split = rng.choice(pool)
        if kind == "gen":
            text = rng.choice(LINES) if rng.random() < 0.3 else " ".join(
                [rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS), rng.choice(TAILS)]).strip()
        else:
            text = rng.choices([t for t, _ in texts], weights=[w for _, w in texts])[0]
        jobs.append((voice, spk, text, split, seed * 1000003 + i))
    return jobs


def _synth_worker(args):
    work, jobs = args
    out = []
    for voice, spk, text, split, seed in jobs:
        try:
            out.append((split, _speak(Path(work), voice, spk, text, random.Random(seed))))
        except Exception as e:                              # a voice that will not say it
            print("skip:", voice, spk, text, e, file=sys.stderr)
    return out


def stage_synth(work: Path, n_pos: int, n_adv: int, n_gen: int, procs: int) -> None:
    from multiprocessing import Pool
    for kind, n, seed in (("pos", n_pos, 1), ("adv", n_adv, 2), ("gen", n_gen, 3)):
        dest = work / f"synth_{kind}.npz"
        if dest.exists():
            continue
        jobs = _jobs(work, kind, n, seed)
        # Grouped by voice, so each worker loads few voice models.
        jobs.sort(key=lambda j: j[0])
        chunks = [jobs[i::procs * 8] for i in range(procs * 8)]
        t0 = time.time()
        results = []
        with Pool(procs) as pool:
            for k, part in enumerate(pool.imap_unordered(_synth_worker,
                                                         [(str(work), c) for c in chunks])):
                results.extend(part)
                if (k + 1) % 8 == 0:
                    log(f"synth {kind}: {len(results)}/{n}")
        clips = {"train": [], "test": []}
        for split, x in results:
            clips[split].append(x)
        np.savez(dest, train=np.array(clips["train"], dtype=object),
                 test=np.array(clips["test"], dtype=object))
        log(f"synth {kind}: {len(clips['train'])} train / {len(clips['test'])} test "
            f"in {time.time() - t0:.0f}s")


# --- augmentation and features ----------------------------------------------------------------

def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.integers(0, 3)
    w = rng.standard_normal(n).astype(np.float32)
    if kind == 0:
        return w
    spec = np.fft.rfft(w)
    f = np.arange(spec.size) + 1.0
    spec /= f ** (0.5 if kind == 1 else 1.0)          # pink, brown
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (np.std(x) + 1e-8)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def _reverb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import fftconvolve
    rt60 = rng.uniform(0.15, 0.7)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    y = fftconvolve(x, ir)[: x.size]
    return y * (_rms(x) / _rms(y))


def _colour(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    lo, hi = rng.uniform(80, 300), rng.uniform(3500, 7500)
    return sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def _place(phrase: np.ndarray, others, rng: np.random.Generator, tail_max: float) -> np.ndarray:
    """A 2 s clip that ends `tail` after the phrase, over a background."""
    clip = np.zeros(CLIP_SAMPLES, np.float32)
    phrase = phrase[-int(1.6 * SR):]
    tail = int(rng.uniform(0.0, tail_max) * SR)
    end = CLIP_SAMPLES - tail
    start = max(0, end - phrase.size)
    clip[start:end] = phrase[phrase.size - (end - start):]
    level = _rms(phrase)
    r = rng.random()
    if r < 0.35:                               # someone else talking, quieter
        o = others[rng.integers(len(others))]
        o = np.resize(o, CLIP_SAMPLES) if o.size < CLIP_SAMPLES else o[:CLIP_SAMPLES]
        snr = rng.uniform(6, 20)
        clip += o * (level / _rms(o)) * 10 ** (-snr / 20)
    if r > 0.15:                               # the room
        snr = rng.uniform(5, 35)
        clip += _noise(CLIP_SAMPLES, rng) * level * 10 ** (-snr / 20)
    if rng.random() < 0.4:
        clip = _reverb(clip, rng)
    if rng.random() < 0.4:
        clip = _colour(clip, rng)
    peak = np.max(np.abs(clip)) + 1e-9
    clip *= rng.uniform(0.05, 0.9) / peak
    return np.clip(clip * 32767, -32768, 32767).astype(np.int16)


def _crop(sentence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2 s of an ordinary sentence, anywhere in it."""
    if sentence.size > CLIP_SAMPLES:
        s = rng.integers(0, sentence.size - CLIP_SAMPLES)
        return sentence[s:s + CLIP_SAMPLES]
    return sentence


def _embed(work: Path, clips: np.ndarray, procs: int) -> np.ndarray:
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures(melspec_model_path=str(work / "oww" / "melspectrogram.onnx"),
                       embedding_model_path=str(work / "oww" / "embedding_model.onnx"),
                       inference_framework="onnx")
    out = []
    for i in range(0, len(clips), 512):
        e = af.embed_clips(clips[i:i + 512], batch_size=64, ncpu=procs)
        out.append(e[:, -FRAMES:, :].astype(np.float16))
    return np.concatenate(out)


def _load_wavs(folder: Path) -> list[np.ndarray]:
    clips = []
    for p in sorted(folder.glob("*.wav")):
        with wave.open(str(p)) as r:
            x = np.frombuffer(r.readframes(r.getnframes()), np.int16).astype(np.float32) / 32768
            if r.getnchannels() > 1:
                x = x.reshape(-1, r.getnchannels()).mean(1)
            clips.append(_trim(_resample(x, r.getframerate())).astype(np.float32))
    return clips


def stage_features(work: Path, aug: int, procs: int, my_recordings: Path | None) -> None:
    dest = work / "features.npz"
    if dest.exists():
        return
    rng = np.random.default_rng(7)
    syn = {k: np.load(work / f"synth_{k}.npz", allow_pickle=True) for k in ("pos", "adv", "gen")}
    others = list(syn["gen"]["train"])
    feats = {}
    mine = _load_wavs(my_recordings) if my_recordings else []
    rng.shuffle(mine)
    n_mine_test = len(mine) // 5
    for split in ("train", "test"):
        reps = aug if split == "train" else 1
        pos = list(syn["pos"][split])
        if mine:
            pos += (mine[n_mine_test:] * 4) if split == "train" else mine[:n_mine_test]
        clips = {
            "pos": [_place(p, others, rng, 0.2) for _ in range(reps) for p in pos],
            "adv": [_place(a, others, rng, 0.6) for _ in range(reps) for a in syn["adv"][split]],
            "gen": [_place(_crop(g, rng), others, rng, 0.3) for _ in range(reps)
                    for g in syn["gen"][split]],
        }
        # Quiet rooms and plain noise, with nothing said at all.
        clips["gen"] += [np.clip(_noise(CLIP_SAMPLES, rng) * rng.uniform(30, 3000), -32768, 32767)
                         .astype(np.int16) for _ in range(300 * reps)]
        for kind, cl in clips.items():
            t0 = time.time()
            feats[f"{split}_{kind}"] = _embed(work, np.stack(cl), procs)
            log(f"features {split}/{kind}: {len(cl)} clips in {time.time() - t0:.0f}s")
        if split == "test" and mine:
            feats["test_mine"] = _embed(work, np.stack([_place(p, others, rng, 0.2)
                                                        for p in mine[:n_mine_test]]), procs)
    np.savez(dest, **feats)


# --- the network ------------------------------------------------------------------------------

class MLP:
    """1536 -> 64 -> 32 -> 1, ReLU, sigmoid output. Adam, weighted BCE."""

    def __init__(self, sizes=(FRAMES * EMBED, 64, 32, 1), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [(rng.standard_normal((a, b)) * math.sqrt(2 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        hs = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            hs.append(h)
        return hs

    def predict(self, X, batch=8192):
        out = []
        for i in range(0, len(X), batch):
            z = self.forward(X[i:i + batch])[-1][:, 0]
            out.append(1 / (1 + np.exp(-np.clip(z, -30, 30))))
        return np.concatenate(out)

    def step(self, X, y, w, lr, wd=1e-4):
        hs = self.forward(X)
        z = hs[-1][:, 0]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = ((p - y) * w / w.sum())[:, None].astype(np.float32)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = hs[i].T @ g + wd * self.W[i]
            gb[i] = g.sum(0)
            if i:
                g = (g @ self.W[i].T) * (hs[i] > 0)
        self.t += 1
        b1, b2 = 0.9, 0.999
        for j, (param, grad) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * grad
            self.v[j] = b2 * self.v[j] + (1 - b2) * grad * grad
            mh = self.m[j] / (1 - b1 ** self.t)
            vh = self.v[j] / (1 - b2 ** self.t)
            param -= lr * mh / (np.sqrt(vh) + 1e-8)


def _flat(x):
    return np.asarray(x, dtype=np.float32).reshape(len(x), -1)


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def stage_train(work: Path, out: Path, steps: int, max_neg_weight: float,
                target_fph: float) -> None:
    """openWakeWord's recipe, in numpy.

    Every batch has the same make-up whatever the dataset sizes -- 256
    positives, 256 near-misses, 128 sentences, 1,280 ACAV windows and up to
    128 of the hardest negatives found so far -- and the negatives' weight
    ramps from 1 to `max_neg_weight` over training. A model trained the other
    way round (positives weighted ~17x, 128 hours of negatives) recognised
    97% of held-out voices and woke 83 times an hour on the validation
    stream. The threshold is then chosen on the FIRST half of that stream
    for `target_fph` false wakes an hour, and folded into the output bias,
    so 0.5 -- the runtime default -- is the calibrated point.
    """
    f = np.load(work / "features.npz")
    acav = _acav(work)
    n_test = len(acav) // 10
    acav_train = np.arange(n_test, len(acav))
    pos, adv, gen = f["train_pos"], f["train_adv"], f["train_gen"]
    sample = np.concatenate([_flat(pos[:4000]), _flat(adv[:3000]), _flat(gen[:2000]),
                             _flat(acav[n_test:n_test + 20000])])
    mu = sample.mean(0)
    sd = sample.std(0) + 1e-3
    norm = lambda x: (_flat(x) - mu) / sd                            # noqa: E731
    log(f"train: {len(pos)} positives, {len(adv)} near-misses, {len(gen)} sentences, "
        f"{len(acav_train)} ACAV windows; {steps} steps, negative weight to {max_neg_weight}")

    net = MLP()
    rng = np.random.default_rng(0)
    hard = np.empty((0, FRAMES, EMBED), np.float16)
    mine_every = max(1, steps // 5)
    t0 = time.time()
    for step in range(steps):
        frac = step / steps
        neg_w = 1 + (max_neg_weight - 1) * frac
        lr = 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        parts = [pos[rng.integers(len(pos), size=256)], adv[rng.integers(len(adv), size=256)],
                 gen[rng.integers(len(gen), size=128)],
                 acav[np.sort(rng.choice(acav_train, 1280, replace=False))]]
        if len(hard):
            parts.append(hard[rng.integers(len(hard), size=min(128, len(hard)))])
        X = norm(np.concatenate(parts))
        y = np.zeros(len(X), np.float32)
        y[:256] = 1
        w = np.full(len(X), neg_w, np.float32)
        w[:256] = 1.0
        w[256:512] *= 2.0                     # near-misses count double
        net.step(X, y, w, lr)
        if (step + 1) % mine_every == 0 and step + 1 < steps:
            scores = np.concatenate([net.predict(norm(acav[i:i + 50000]))
                                     for i in range(n_test, len(acav), 50000)])
            worst = np.argsort(scores)[::-1][:20000]
            worst = worst[scores[worst] > 0.02]
            hard = np.asarray(acav[np.sort(worst + n_test)]) if len(worst) else hard
            p_pos = net.predict(norm(f["test_pos"]))
            log(f"step {step + 1}: held-out recall@0.5 {float((p_pos >= 0.5).mean()):.3f}, "
                f"ACAV windows >0.5: {int((scores > 0.5).sum())}, hard pool {len(hard)}, "
                f"{time.time() - t0:.0f}s")

    # Calibrate on the first half of the validation stream.
    tmp = work / "uncalibrated.onnx"
    _export(net, mu, sd, tmp)
    val = np.load(work / "validation_set_features.npy", mmap_mode="r")
    half = len(val) // 2
    p_val = _stream_scores(tmp, val[:half])
    hours = half * 0.08 / 3600
    p_pos = _score(tmp, f["test_pos"])
    chosen = 0.99
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99):
        fph = _activations(p_val, t) / hours
        log(f"threshold {t}: {fph:.2f} false wakes/hour (calibration half), "
            f"held-out recall {float((p_pos >= t).mean()):.3f}")
        if fph <= target_fph:
            chosen = t
            break
    net.b[-1] -= _logit(chosen)               # score >= 0.5  <=>  old score >= chosen
    _export(net, mu, sd, out)
    tmp.unlink(missing_ok=True)
    (work / "calibration.json").write_text(json.dumps({"threshold": chosen}))
    log(f"calibrated at {chosen} (now 0.5); wrote {out}")


def _export(net: MLP, mu, sd, out: Path) -> None:
    """ONNX with openWakeWord's shape; normalisation folded into layer one."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W0 = (net.W[0] / sd[:, None]).astype(np.float32)
    b0 = (net.b[0] - (mu / sd) @ net.W[0]).astype(np.float32)
    Ws, bs = [W0] + net.W[1:], [b0] + net.b[1:]
    nodes, inits = [helper.make_node("Flatten", ["input"], ["h0"], axis=1)], []
    h = "h0"
    for i, (W, b) in enumerate(zip(Ws, bs)):
        inits += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(b, f"b{i}")]
        nodes.append(helper.make_node("Gemm", [h, f"W{i}", f"b{i}"], [f"z{i}"]))
        if i < len(Ws) - 1:
            nodes.append(helper.make_node("Relu", [f"z{i}"], [f"h{i + 1}"]))
            h = f"h{i + 1}"
    nodes.append(helper.make_node("Sigmoid", [f"z{len(Ws) - 1}"], ["score"]))
    graph = helper.make_graph(
        nodes, "hey_auvren",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", FRAMES, EMBED])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="auvren-train_wakeword")
    model.ir_version = 8
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))


# --- evaluation -------------------------------------------------------------------------------

def _score(model_path: Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(str(model_path))
    out = [s.run(None, {"input": X[i:i + 4096].astype(np.float32)})[0][:, 0]
           for i in range(0, len(X), 4096)]
    return np.concatenate(out)


def _stream_scores(model_path: Path, frames: np.ndarray) -> np.ndarray:
    """Score a continuous stream of embedding frames the way the runtime
    does: the last 16 frames, at every 80 ms frame."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(model_path))
    out = []
    for i in range(0, len(frames) - FRAMES + 1, 8192):
        block = np.asarray(frames[i:i + 8192 + FRAMES - 1], dtype=np.float32)
        win = np.lib.stride_tricks.sliding_window_view(block, (FRAMES, EMBED))[:, 0]
        out.append(s.run(None, {"input": np.ascontiguousarray(win)})[0][:, 0])
    return np.concatenate(out)


def _activations(scores: np.ndarray, t: float, refractory: int = 20) -> int:
    """Wakes, not frames: a run over the threshold is one wake, and none
    can follow within `refractory` frames (1.6 s) — the listener's cooldown."""
    n, last = 0, -10 ** 9
    above = np.nonzero(scores >= t)[0]
    for i in above:
        if i - last > refractory:
            n += 1
        last = i if i - last > refractory else last
    return n


def _runtime_check(work: Path, model_path: Path, t: float = 0.5, n: int = 60) -> dict:
    """The model in the REAL openWakeWord runtime, fed 80 ms at a time:
    held-out voices saying the wake phrase between ordinary sentences and
    near-misses, over room noise. A wake within 1.5 s of a phrase's end is a
    hit; any other wake is a false one."""
    from openwakeword.model import Model
    syn = {k: np.load(work / f"synth_{k}.npz", allow_pickle=True)["test"] for k in ("pos", "adv", "gen")}
    rng = np.random.default_rng(11)
    pieces, ends, cursor = [], [], 0
    for i in range(n):
        for kind in ("gen", "adv", "pos", "gen"):
            clip = syn[kind][rng.integers(len(syn[kind]))]
            gap = np.zeros(int(rng.uniform(0.3, 1.0) * SR), np.float32)
            pieces += [gap, clip]
            cursor += gap.size + clip.size
            if kind == "pos":
                ends.append(cursor)
    audio = np.concatenate(pieces)
    audio = audio + _noise(audio.size, rng) * 0.003
    pcm = np.clip(audio * 0.5 * 32767, -32768, 32767).astype(np.int16)
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx",
              melspec_model_path=str(work / "oww" / "melspectrogram.onnx"),
              embedding_model_path=str(work / "oww" / "embedding_model.onnx"))
    wakes, last = [], -10 ** 9
    for k in range(0, pcm.size - 1280, 1280):
        score = max(m.predict(pcm[k:k + 1280]).values())
        if score >= t and k - last > 2 * SR:
            wakes.append(k + 1280)
            last = k
            m.reset()
    # A detection can land a frame BEFORE the trimmed clip's last sample.
    near = lambda w, e: -int(0.5 * SR) <= w - e <= int(1.5 * SR)       # noqa: E731
    hits = sum(any(near(w, e) for w in wakes) for e in ends)
    false = sum(not any(near(w, e) for e in ends) for w in wakes)
    return {"threshold": t, "phrases": len(ends), "heard": hits,
            "false_wakes": false, "stream_minutes": round(pcm.size / SR / 60, 1)}


def stage_eval(work: Path, model_path: Path, procs: int) -> dict:
    f = np.load(work / "features.npz")
    acav = _acav(work)
    val = np.load(work / "validation_set_features.npy", mmap_mode="r")
    val = val[len(val) // 2:]          # the half the threshold was NOT chosen on
    report = {}
    p_pos = _score(model_path, f["test_pos"])
    p_adv = _score(model_path, f["test_adv"])
    p_gen = _score(model_path, f["test_gen"])
    p_acav = _score(model_path, np.asarray(acav[: len(acav) // 10]))
    p_val = _stream_scores(model_path, val)
    hours_val = len(val) * 0.08 / 3600
    for t in (0.3, 0.5, 0.7, 0.9):
        report[str(t)] = {
            "recall_heldout_voices": round(float((p_pos >= t).mean()), 3),
            "near_miss_false_wakes": round(float((p_adv >= t).mean()), 4),
            "sentence_false_wakes": round(float((p_gen >= t).mean()), 4),
            "acav_false_wakes_per_hour": round(float((p_acav >= t).sum() / (len(p_acav) * 1.28 / 3600)), 2),
            "validation_false_wakes_per_hour_heldout_half": round(_activations(p_val, t) / hours_val, 2),
        }
        if "test_mine" in f:
            report[str(t)]["recall_your_voice"] = round(float((_score(model_path, f["test_mine"]) >= t).mean()), 3)
    report["runtime_stream"] = _runtime_check(work, model_path)
    report["validation_hours"] = round(hours_val, 1)
    report["heldout_positive_clips"] = int(len(p_pos))
    log(json.dumps(report, indent=2))
    (work / "eval.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["download", "synth", "features", "train", "eval", "all"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/wakeword/hey_auvren_v0.1.onnx"))
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--n-pos", type=int, default=6000)
    ap.add_argument("--n-adv", type=int, default=5000)
    ap.add_argument("--n-gen", type=int, default=3000)
    ap.add_argument("--aug", type=int, default=2, help="augmented copies of each training clip")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-neg-weight", type=float, default=100.0)
    ap.add_argument("--target-fph", type=float, default=0.5,
                    help="false wakes per hour to calibrate the threshold for")
    ap.add_argument("--acav-rows", type=int, default=1_200_000)
    ap.add_argument("--acav-chunks", type=int, default=60)
    ap.add_argument("--my-recordings", type=Path, default=None)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    stages = ["download", "synth", "features", "train", "eval"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"== {s}")
        if s == "download":
            stage_download(a.work, a.acav_rows, a.acav_chunks)
        elif s == "synth":
            stage_synth(a.work, a.n_pos, a.n_adv, a.n_gen, a.procs)
        elif s == "features":
            stage_features(a.work, a.aug, a.procs, a.my_recordings)
        elif s == "train":
            stage_train(a.work, a.out, a.steps, a.max_neg_weight, a.target_fph)
        elif s == "eval":
            stage_eval(a.work, a.out, a.procs)


if __name__ == "__main__":
    main()
