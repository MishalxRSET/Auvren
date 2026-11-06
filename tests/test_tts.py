import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_sends_balanced_latency_and_assembles_stream():
    import tts
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"ID3" + b"\x00" * 100)

    async with _client(handler) as c:
        r = await tts.synthesize_chunk("Good evening, sir.", api_key="k", voice_id="v", client=c)
    assert seen["url"] == tts.FISH_TTS_URL
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {"text": "Good evening, sir.", "reference_id": "v",
                            "format": "mp3", "mp3_bitrate": 128, "latency": "balanced"}
    assert r is not None and r.audio.startswith(b"ID3") and len(r.audio) == 103
    assert r.first_byte_sec >= 0 and r.total_sec >= r.first_byte_sec


@pytest.mark.asyncio
async def test_non_200_returns_none():
    import tts
    async with _client(lambda req: httpx.Response(401, content=b"nope")) as c:
        assert await tts.synthesize_chunk("x", api_key="k", voice_id="v", client=c) is None


@pytest.mark.asyncio
async def test_transport_error_returns_none():
    import tts

    def boom(request):
        raise httpx.ConnectError("down")

    async with _client(boom) as c:
        assert await tts.synthesize_chunk("x", api_key="k", voice_id="v", client=c) is None


@pytest.mark.asyncio
async def test_empty_text_or_missing_key_short_circuits():
    import tts
    calls = []

    async with _client(lambda req: calls.append(1) or httpx.Response(200, content=b"x")) as c:
        assert await tts.synthesize_chunk("   ", api_key="k", voice_id="v", client=c) is None
        assert await tts.synthesize_chunk("hi", api_key="", voice_id="v", client=c) is None
    assert calls == []


# --- Piper ---------------------------------------------------------------------

import asyncio
import io
import os
import stat
import wave

import tts as _tts


def _wav(seconds: float, rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _install(tmp_path, monkeypatch, *, json_config=True):
    exe = tmp_path / "bin" / "piper.exe"
    exe.parent.mkdir()
    exe.write_text("")
    model = tmp_path / "en_GB-alan-medium.onnx"
    model.write_text("")
    if json_config:
        (tmp_path / "en_GB-alan-medium.onnx.json").write_text(
            json.dumps({"audio": {"sample_rate": 22050}}))
    monkeypatch.setenv("PIPER_PATH", str(exe))
    monkeypatch.setenv("PIPER_MODEL", str(model))
    return str(exe), str(model)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for v in ("TTS_ENGINE", "FISH_API_KEY", "PIPER_PATH", "PIPER_MODEL",
              "PIPER_SPEAKER", "PIPER_LENGTH_SCALE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(_tts.shutil, "which", lambda name: None)


class _FakeProc:
    def __init__(self, out_path, *, rc=0, write=None, hang=False):
        self._out, self.returncode, self._write, self._hang = out_path, None, write, hang
        self._rc = rc
        self.stdin_got = None
        self.killed = False

    async def communicate(self, data=None):
        if data is not None:
            self.stdin_got = data
        if self._hang and not self.killed:
            await asyncio.sleep(999)
        if self._write is not None and self._out:
            Path(self._out).write_bytes(self._write)
        self.returncode = self._rc if not self.killed else -9
        return b"", b"boom" if self._rc else b""

    def kill(self):
        self.killed = True


def _patch_spawn(monkeypatch, **kw):
    seen = {}

    async def fake_exec(*argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        out = argv[argv.index("--output_file") + 1]
        seen["proc"] = _FakeProc(out, **kw)
        return seen["proc"]
    monkeypatch.setattr(_tts.asyncio, "create_subprocess_exec", fake_exec)
    return seen


def test_engine_defaults_to_piper_when_installed(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    monkeypatch.setenv("FISH_API_KEY", "k")
    assert _tts.engine() == "piper"


def test_engine_keeps_an_existing_fish_setup_speaking(clean_env, monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "k")
    assert _tts.engine() == "fish"


def test_engine_with_nothing_is_piper_and_says_what_is_missing(clean_env):
    assert _tts.engine() == "piper"
    assert "not installed" in _tts.voice_problem()


def test_explicit_engine_wins(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    monkeypatch.setenv("TTS_ENGINE", "FISH")
    assert _tts.engine() == "fish"
    assert "FISH_API_KEY" in _tts.voice_problem()


def test_a_voice_without_its_json_config_is_not_ready(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, json_config=False)
    assert ".onnx.json" in _tts.piper_problem()


def test_the_one_installed_voice_is_found_and_two_are_a_choice(clean_env, tmp_path, monkeypatch):
    voices = tmp_path / "data" / "piper" / "voices"
    voices.mkdir(parents=True)
    (voices / "a.onnx").write_text("")
    assert _tts.piper_model() == str(voices / "a.onnx")
    (voices / "b.onnx").write_text("")
    assert _tts.piper_model() is None, "two voices and no PIPER_MODEL is not a guess"


def test_argv_carries_flags_and_paths_never_text(clean_env, monkeypatch):
    monkeypatch.setenv("PIPER_SPEAKER", "3")
    monkeypatch.setenv("PIPER_LENGTH_SCALE", "1.1")
    argv = _tts.piper_argv("piper.exe", "v.onnx", "out.wav")
    assert argv == ["piper.exe", "--model", "v.onnx", "--output_file", "out.wav",
                    "--speaker", "3", "--length_scale", "1.1"]
    monkeypatch.setenv("PIPER_SPEAKER", "3; calc")
    monkeypatch.setenv("PIPER_LENGTH_SCALE", "--evil")
    assert _tts.piper_argv("p", "m", "o") == ["p", "--model", "m", "--output_file", "o"]


HOSTILE = 'Done, sir." & del /q C:\\* | %PATH% `$(calc)\nsecond line\r\nthird'


@pytest.mark.asyncio
async def test_text_goes_on_stdin_as_one_line_and_never_in_argv(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    seen = _patch_spawn(monkeypatch, write=_wav(1.0))
    r = await _tts.synthesize_piper(HOSTILE)
    assert r is not None and r.audio[:4] == b"RIFF"
    assert all(HOSTILE.split()[0] not in a for a in seen["argv"][1:] if a != seen["argv"][0])
    assert not any("calc" in a or "PATH" in a for a in seen["argv"])
    sent = seen["proc"].stdin_got.decode("utf-8")
    assert sent.count("\n") == 1 and sent.endswith("\n"), "exactly one utterance"
    assert "del /q" in sent and "%PATH%" in sent, "the text itself is untouched"
    assert seen["kwargs"]["cwd"] == str(Path(os.environ["PIPER_PATH"]).parent)


@pytest.mark.asyncio
async def test_the_temp_file_is_always_cleaned_up(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    seen = _patch_spawn(monkeypatch, write=_wav(0.5))
    await _tts.synthesize_piper("Good evening.")
    out = Path(seen["argv"][seen["argv"].index("--output_file") + 1])
    assert not out.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kw", [{"rc": 1, "write": _wav(1.0)},
                                {"write": b"not a wav at all"},
                                {"write": None}])
async def test_a_failed_synthesis_is_none_not_an_exception(clean_env, tmp_path, monkeypatch, kw):
    _install(tmp_path, monkeypatch)
    _patch_spawn(monkeypatch, **kw)
    assert await _tts.synthesize_piper("Good evening.") is None


@pytest.mark.asyncio
async def test_a_wedged_piper_is_killed(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    seen = _patch_spawn(monkeypatch, hang=True)
    assert await _tts.synthesize_piper("Good evening.", timeout=0.05) is None
    assert seen["proc"].killed


@pytest.mark.asyncio
async def test_a_missing_binary_is_none(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)

    async def boom(*a, **k):
        raise FileNotFoundError("piper.exe")
    monkeypatch.setattr(_tts.asyncio, "create_subprocess_exec", boom)
    assert await _tts.synthesize_piper("Good evening.") is None


@pytest.mark.asyncio
async def test_a_runaway_chunk_is_refused_before_anything_starts(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    seen = _patch_spawn(monkeypatch, write=_wav(1.0))
    assert await _tts.synthesize_piper("x" * (_tts.PIPER_MAX_CHARS + 1)) is None
    assert "argv" not in seen


@pytest.mark.asyncio
async def test_synthesize_dispatches_on_the_engine(clean_env, tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    _patch_spawn(monkeypatch, write=_wav(0.2))
    r = await _tts.synthesize("Hello.", fish_api_key="k", fish_voice_id="v")
    assert r.audio[:4] == b"RIFF", "Piper when installed, even with a Fish key"


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="a shebang stand-in for piper.exe")
async def test_a_real_subprocess_round_trip(clean_env, tmp_path, monkeypatch):
    """A stand-in `piper` run for real: argv, stdin, cwd and the WAV handoff."""
    _, model = _install(tmp_path, monkeypatch)
    fake = tmp_path / "bin" / "piper"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import sys, os, wave\n"
        "a = sys.argv\n"
        "assert a[a.index('--model') + 1].endswith('.onnx')\n"
        "text = sys.stdin.read()\n"
        "assert text.count('\\n') == 1\n"
        "assert os.path.isfile('piper.exe'), 'runs beside itself'\n"
        "with wave.open(a[a.index('--output_file') + 1], 'wb') as w:\n"
        "    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)\n"
        "    w.writeframes(b'\\x00\\x00' * 22050 * len(text.split()))\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    r = await _tts.synthesize_piper("one two three", exe=str(fake), model=model)
    assert r is not None
    import speech
    assert abs(speech.wav_seconds(r.audio) - 3.0) < 0.01
