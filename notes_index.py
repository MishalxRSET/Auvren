"""notes_index.py — Auvren answers from your own notes, PDFs and slides.

"What does Module 3 say about inheritance?" -> the passage, with the file and
the page it came from. Offline: everything is read and searched on this
machine.

Point it at folders with NOTES_DIRS (separated by `;` on Windows, `:`
elsewhere). Read:

  .pdf    page by page            (PyMuPDF: requirements-notes.txt)
  .pptx   slide by slide          (standard library: a .pptx is zipped XML)
  .docx   in parts of ~900 chars  (standard library, likewise)
  .txt .md                         (as text)

Search is SQLite FTS5 (BM25 keyword ranking) always, blended -- reciprocal
rank fusion -- with meaning-based search when Ollama has an embedding model
pulled (NOTES_EMBED_MODEL, default nomic-embed-text) and numpy is present, so
"inheritance" also finds the passage that only says "subclass extends". The
index lives in <data>/notes.db and is refreshed incrementally: only files
whose size or modification time changed are read again.

What comes back is text from files the user collected, not wrote -- a PDF
from anywhere -- so the server hands it to the brain inside an untrusted
block and marks the turn as having read it (server.TAINTING_TOOLS).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import zipfile
from array import array
from pathlib import Path
from typing import Callable, Optional
from xml.etree import ElementTree

log = logging.getLogger("jarvis.notes")

SUFFIXES = {".pdf", ".pptx", ".docx", ".txt", ".md"}
MAX_FILE_BYTES = 60 * 1024 * 1024
MAX_FILES = 5000
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
_SKIP_DIRS = {"node_modules", "__pycache__", "$recycle.bin", "system volume information"}

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def note_dirs() -> list[Path]:
    raw = os.getenv("NOTES_DIRS") or ""
    return [Path(p.strip().strip('"')).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def db_path() -> Path:
    import data_paths
    return data_paths.data_dir() / "notes.db"


def _pymupdf():
    """PyMuPDF; `pymupdf` is its name now, `fitz` the old one (deprecated)."""
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz
        return fitz


def pdf_problem() -> Optional[str]:
    try:
        _pymupdf()
    except ImportError:
        return "PDFs can't be read yet: pip install -r requirements-notes.txt"
    return None


# --- reading files -------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n{3,}", "\n\n", text or "")).strip()


def _split(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pieces of about `size` characters, broken at a paragraph or sentence
    where one is near, overlapping a little so no answer falls in a gap."""
    text = _clean(text)
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            if cut > start:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def _pdf(path: Path) -> list[tuple[str, str]]:
    out = []
    with _pymupdf().open(str(path)) as doc:
        for i, page in enumerate(doc, 1):
            for piece in _split(page.get_text("text")):
                out.append((f"page {i}", piece))
    return out


def _pptx(path: Path) -> list[tuple[str, str]]:
    out = []
    with zipfile.ZipFile(path) as z:
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1)))
        for n in slides:
            root = ElementTree.fromstring(z.read(n))
            paras = []
            for p in root.iter(f"{_A}p"):
                line = "".join(t.text or "" for t in p.iter(f"{_A}t")).strip()
                if line:
                    paras.append(line)
            num = int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1))
            for piece in _split("\n".join(paras)):
                out.append((f"slide {num}", piece))
    return out


def _docx(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{_W}p"):
        line = "".join(t.text or "" for t in p.iter(f"{_W}t")).strip()
        if line:
            paras.append(line)
    return [(f"part {i}", c) for i, c in enumerate(_split("\n".join(paras)), 1)]


def _text(path: Path) -> list[tuple[str, str]]:
    return [(f"part {i}", c) for i, c in
            enumerate(_split(path.read_text("utf-8", errors="replace")), 1)]


def extract(path: Path) -> list[tuple[str, str]]:
    """(location, text) pieces of one file. Raises on a file it cannot read."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf(path)
    if suffix == ".pptx":
        return _pptx(path)
    if suffix == ".docx":
        return _docx(path)
    return _text(path)


# --- meaning-based search (optional) ---------------------------------------------------------

class OllamaEmbedder:
    """Embeddings from Ollama's /api/embed. `available()` is cheap and cached."""

    def __init__(self, model: Optional[str] = None, url: Optional[str] = None, client=None):
        import local_brain
        self.model = (model or os.getenv("NOTES_EMBED_MODEL") or "nomic-embed-text").strip()
        self.url = url or local_brain.ollama_url()
        self._client = client
        self._ok: Optional[bool] = None
        self._checked = 0.0

    def _http(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def available(self) -> bool:
        if self._ok is not None and time.monotonic() - self._checked < 300:
            return self._ok
        self._checked = time.monotonic()
        try:
            import local_brain
            import numpy  # noqa: F401
            r = self._http().get(f"{self.url}/api/tags", timeout=3.0)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            self._ok = local_brain.model_present(self.model, names)
        except Exception:
            self._ok = False
        return self._ok

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self._http().post(f"{self.url}/api/embed",
                              json={"model": self.model, "input": texts, "keep_alive": "5m"})
        r.raise_for_status()
        return r.json()["embeddings"]


# --- the index ---------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER,
                                  pieces INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, path TEXT, loc TEXT, text TEXT,
                                   vec BLOB, vec_model TEXT);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""

_WORD = re.compile(r"[A-Za-z0-9]{2,}")


class NotesIndex:
    def __init__(self, path: Optional[Path] = None,
                 embedder: Optional[OllamaEmbedder] = None,
                 dirs: Optional[Callable[[], list[Path]]] = None):
        self.path = path or db_path()
        self.embedder = embedder
        self._dirs = dirs or note_dirs
        self._lock = threading.Lock()
        self.last_refresh: Optional[dict] = None

    def _db(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.path), timeout=30)
        con.executescript(_SCHEMA)
        return con

    # ── building ──────────────────────────────────────────────────────
    def _walk(self) -> dict[str, tuple[float, int]]:
        found: dict[str, tuple[float, int]] = {}
        for root in self._dirs():
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames
                               if not d.startswith(".") and d.lower() not in _SKIP_DIRS]
                for name in filenames:
                    p = Path(dirpath) / name
                    if p.suffix.lower() not in SUFFIXES or name.startswith(("~$", ".")):
                        continue
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_size <= MAX_FILE_BYTES:
                        found[str(p)] = (st.st_mtime, st.st_size)
                    if len(found) >= MAX_FILES:
                        return found
        return found

    def refresh(self) -> dict:
        """Bring the index up to date with the folders. Only changed files
        are read. Returns counts; never raises for one bad file."""
        with self._lock:
            t0 = time.monotonic()
            on_disk = self._walk()
            con = self._db()
            try:
                known = {p: (m, s) for p, m, s in con.execute("SELECT path, mtime, size FROM files")}
                gone = [p for p in known if p not in on_disk]
                changed = [p for p, ms in on_disk.items() if known.get(p) != ms]
                for p in gone + changed:
                    self._forget(con, p)
                added = failed = 0
                for p in changed:
                    try:
                        pieces = extract(Path(p))
                        error = None
                    except Exception as e:          # a corrupt or locked file
                        pieces, error = [], f"{type(e).__name__}: {e}"[:200]
                        failed += 1
                    for loc, text in pieces:
                        cur = con.execute("INSERT INTO chunks (path, loc, text) VALUES (?, ?, ?)",
                                          (p, loc, text))
                        con.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                                    (cur.lastrowid, text))
                    mtime, size = on_disk[p]
                    con.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                                (p, mtime, size, len(pieces), error))
                    added += 1
                    con.commit()
                embedded = self._embed_missing(con)
                con.commit()
                files, pieces = con.execute(
                    "SELECT COUNT(*), COALESCE(SUM(pieces), 0) FROM files").fetchone()
            finally:
                con.close()
            self.last_refresh = {"files": files, "pieces": pieces, "read": added,
                                 "removed": len(gone), "failed": failed, "embedded": embedded,
                                 "seconds": round(time.monotonic() - t0, 1)}
            if added or gone:
                log.info("notes: %s", self.last_refresh)
            return self.last_refresh

    @staticmethod
    def _forget(con: sqlite3.Connection, path: str) -> None:
        ids = [r[0] for r in con.execute("SELECT id FROM chunks WHERE path = ?", (path,))]
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            con.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({','.join('?' * len(part))})", part)
        con.execute("DELETE FROM chunks WHERE path = ?", (path,))
        con.execute("DELETE FROM files WHERE path = ?", (path,))

    def _embed_missing(self, con: sqlite3.Connection, batch: int = 32) -> int:
        if self.embedder is None or not self.embedder.available():
            return 0
        model = self.embedder.model
        rows = con.execute("SELECT id, text FROM chunks WHERE vec IS NULL OR vec_model != ?",
                           (model,)).fetchall()
        done = 0
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            try:
                vecs = self.embedder.embed([t for _, t in part])
            except Exception as e:
                log.warning(f"notes: embedding failed, keyword search only for now: {e}")
                break
            for (cid, _), v in zip(part, vecs):
                con.execute("UPDATE chunks SET vec = ?, vec_model = ? WHERE id = ?",
                            (array("f", v).tobytes(), model, cid))
            done += len(part)
            con.commit()
        return done

    # ── searching ─────────────────────────────────────────────────────
    def _keyword(self, con, query: str, limit: int) -> list[int]:
        words = _WORD.findall(query.lower())
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
        try:
            return [r[0] for r in con.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?", (match, limit))]
        except sqlite3.OperationalError:
            return []

    def _semantic(self, con, query: str, limit: int) -> list[int]:
        if self.embedder is None or not self.embedder.available():
            return []
        import numpy as np
        rows = con.execute("SELECT id, vec FROM chunks WHERE vec IS NOT NULL AND vec_model = ?",
                           (self.embedder.model,)).fetchall()
        if not rows:
            return []
        try:
            q = np.asarray(self.embedder.embed([query])[0], np.float32)
        except Exception:
            return []
        ids = np.array([r[0] for r in rows])
        mat = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        top = np.argsort(-sims)[:limit]
        return [int(ids[i]) for i in top if sims[i] > 0.3]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """The best passages for `query`: {file, path, loc, text, score}."""
        con = self._db()
        try:
            ranked: dict[int, float] = {}
            for ids in (self._keyword(con, query, 30), self._semantic(con, query, 30)):
                for rank, cid in enumerate(ids):
                    ranked[cid] = ranked.get(cid, 0.0) + 1.0 / (60 + rank)   # RRF
            best = sorted(ranked, key=ranked.get, reverse=True)[:limit]
            hits = []
            for cid in best:
                row = con.execute("SELECT path, loc, text FROM chunks WHERE id = ?", (cid,)).fetchone()
                if row:
                    hits.append({"file": Path(row[0]).name, "path": row[0], "loc": row[1],
                                 "text": row[2], "score": round(ranked[cid], 4)})
            return hits
        finally:
            con.close()

    def stats(self) -> dict:
        con = self._db()
        try:
            files, pieces = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(pieces), 0) FROM files").fetchone()
            failed = con.execute("SELECT COUNT(*) FROM files WHERE error IS NOT NULL").fetchone()[0]
            embedded = con.execute("SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL").fetchone()[0]
            return {"files": files, "pieces": pieces, "failed": failed, "embedded": embedded}
        finally:
            con.close()
