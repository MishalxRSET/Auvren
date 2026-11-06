"""Answering from the user's own notes: reading each file type, the index
(incremental refresh, keyword + optional meaning search), and the server's
search_notes tool. Files are built here; Ollama is faked."""
import asyncio
import importlib
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import notes_index as n

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def pptx(path, slides):
    with zipfile.ZipFile(path, "w") as z:
        for i, text in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml",
                       f'<p:sld xmlns:p="x" xmlns:a="{A}"><a:p><a:r><a:t>{text}</a:t>'
                       f'</a:r></a:p></p:sld>')
        z.writestr("ppt/slides/_rels/slide1.xml.rels", "<x/>")      # not a slide


def docx(path, paras):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paras)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "notes" / "sem3"
    d.mkdir(parents=True)
    pptx(d / "EDC.pptx", ["Diodes and rectifiers",
                         "A Zener diode regulates voltage in reverse breakdown."])
    docx(d / "DSA.docx", ["A binary search tree keeps smaller keys on the left.",
                         "Dijkstra finds shortest paths with non-negative weights."])
    (d / "todo.md").write_text("# Exams\nDSA exam on Friday.\n")
    (d / "ignored.exe").write_bytes(b"MZ")
    (d / "~$DSA.docx").write_bytes(b"lock file")
    hidden = tmp_path / "notes" / ".git"
    hidden.mkdir()
    (hidden / "secret.md").write_text("never indexed")
    return tmp_path / "notes"


@pytest.fixture
def index(folder, tmp_path):
    return n.NotesIndex(path=tmp_path / "notes.db", dirs=lambda: [folder])


# --- reading ---------------------------------------------------------------------------------

def test_slides_and_word_documents_without_extra_packages(folder):
    assert n.extract(folder / "sem3" / "EDC.pptx") == [
        ("slide 1", "Diodes and rectifiers"),
        ("slide 2", "A Zener diode regulates voltage in reverse breakdown.")]
    parts = n.extract(folder / "sem3" / "DSA.docx")
    assert parts[0][0] == "part 1" and "Dijkstra" in parts[0][1]


def test_long_text_is_split_with_overlap_at_sentence_ends():
    text = " ".join(f"Sentence number {i} is about trees." for i in range(200))
    pieces = n._split(text, size=300, overlap=60)
    assert len(pieces) > 5 and all(len(p) <= 320 for p in pieces)
    assert all(p.endswith(".") for p in pieces[:-1]), "broken at a sentence"


def test_a_real_pdf_page_by_page(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in ("Module 3: Java classes.", "Inheritance: a subclass extends its superclass."):
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(tmp_path / "m3.pdf")
    pages = n.extract(tmp_path / "m3.pdf")
    assert [p[0] for p in pages] == ["page 1", "page 2"]
    assert "subclass extends" in pages[1][1]


def test_without_pymupdf_pdfs_say_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    assert "requirements-notes.txt" in n.pdf_problem()


def test_notes_dirs_setting(monkeypatch, tmp_path):
    import os
    monkeypatch.setenv("NOTES_DIRS", f"{tmp_path / 'a'}{os.pathsep} {tmp_path / 'b'} ")
    assert n.note_dirs() == [tmp_path / "a", tmp_path / "b"]
    monkeypatch.delenv("NOTES_DIRS")
    assert n.note_dirs() == []


# --- the index -------------------------------------------------------------------------------

def test_refresh_reads_only_what_changed(index, folder):
    first = index.refresh()
    assert first["files"] == 3 and first["read"] == 3 and first["failed"] == 0
    assert index.refresh()["read"] == 0, "nothing changed, nothing read"
    (folder / "sem3" / "todo.md").write_text("# Exams\nDSA exam moved to Monday.\n")
    second = index.refresh()
    assert second["read"] == 1
    hits = index.search("when is the DSA exam")
    assert "Monday" in hits[0]["text"] and not any("Friday" in h["text"] for h in hits)
    (folder / "sem3" / "EDC.pptx").unlink()
    assert index.refresh()["removed"] == 1
    assert index.search("zener diode") == []


def test_search_names_the_file_and_the_place(index):
    index.refresh()
    hit = index.search("what is a zener diode")[0]
    assert (hit["file"], hit["loc"]) == ("EDC.pptx", "slide 2")
    assert index.search("shortest path")[0]["file"] == "DSA.docx"
    assert index.search("") == [] and index.search("?!") == []


def test_a_broken_file_is_noted_not_fatal(index, folder):
    (folder / "sem3" / "broken.docx").write_bytes(b"not a zip")
    r = index.refresh()
    assert r["failed"] == 1 and r["files"] == 4
    assert index.stats()["failed"] == 1
    assert index.search("zener")


class FakeEmbedder:
    """Two dimensions: [about electronics, about programming]."""
    model = "fake-embed"

    def available(self):
        return True

    def embed(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append([float(any(w in t for w in ("diode", "voltage", "current"))),
                        float(any(w in t for w in ("tree", "path", "graph", "algorithm")))])
        return out


def test_meaning_finds_what_no_keyword_matches(folder, tmp_path):
    pytest.importorskip("numpy")
    ix = n.NotesIndex(path=tmp_path / "e.db", dirs=lambda: [folder], embedder=FakeEmbedder())
    assert ix.refresh()["embedded"] == ix.stats()["pieces"]
    hits = ix.search("current regulation")          # no word in common with the slide
    assert hits and hits[0]["file"] == "EDC.pptx"
    assert ix.refresh()["embedded"] == 0, "embedded once"


def test_an_embedder_that_fails_leaves_keyword_search(folder, tmp_path):
    class Broken(FakeEmbedder):
        def embed(self, texts):
            raise ConnectionError("ollama gone")
    ix = n.NotesIndex(path=tmp_path / "b.db", dirs=lambda: [folder], embedder=Broken())
    assert ix.refresh()["embedded"] == 0
    assert ix.search("zener")[0]["file"] == "EDC.pptx"


# --- the tool --------------------------------------------------------------------------------

@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    import server as srv
    importlib.reload(srv)
    return srv


def test_no_folder_says_how_to_set_one(server, monkeypatch):
    monkeypatch.setattr(server, "notes", None)
    assert "NOTES_DIRS" in asyncio.run(server.tool_search_notes({"query": "zener"}))


def test_the_tool_answers_wrapped_and_taints_the_turn(server, index, monkeypatch):
    index.refresh()
    monkeypatch.setattr(server, "notes", index)
    out = asyncio.run(server.tool_search_notes({"query": "zener diode"}))
    assert "[EDC.pptx, slide 2]" in out
    assert "<session-output" in out and "untrusted" in out, "a PDF from anywhere is not his own words"
    assert server.TAINTING_TOOLS["search_notes"] == "a document in your notes"
    assert "search_notes" not in server.ACTING_TOOLS


def test_nothing_found_and_still_reading(server, index, monkeypatch):
    monkeypatch.setattr(server, "notes", index)
    assert "still reading" in asyncio.run(server.tool_search_notes({"query": "zener"}))
    index.refresh()
    assert asyncio.run(server.tool_search_notes({"query": "photosynthesis"})) == \
        "Nothing in your notes on that."


def test_both_brains_have_it():
    import jarvis_mcp
    import local_brain
    assert "search_notes" in {s["name"] for s in jarvis_mcp.TOOL_SPECS}
    assert "search_notes" in local_brain.LOCAL_TOOLS
    assert "search_notes" in local_brain.PERSONA
