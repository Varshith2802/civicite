"""Load documents (Markdown with front matter, HTML, text, optional PDF) and split them into chunks."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    agency: str = ""
    url: str = ""
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    agency: str
    url: str
    section: str
    text: str

    def header(self) -> str:
        return f"{self.title} > {self.section}" if self.section and self.section != self.title else self.title


_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip("'\"")
    return meta, text[m.end():]


class _HTMLText(HTMLParser):
    """Keeps headings as Markdown and paragraph breaks; drops scripts, styles and navigation."""

    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form", "button", "aside"}
    BLOCK = {"p", "div", "li", "br", "tr", "section", "article", "ul", "ol", "table", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False
        self._heading: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4") and not self.skip:
            self._heading = tag
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK and not self.skip:
            self.out.append("\n\n" if tag in ("p", "div", "section", "article") else "\n")
            if tag == "li":
                self.out.append("- ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == self._heading:
            self._heading = None
            self.out.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def html_to_markdown(html: str) -> tuple[str, str]:
    p = _HTMLText()
    p.feed(html)
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return p.title.strip(), text


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace") if path.suffix.lower() != ".pdf" else ""
    meta: dict[str, str] = {}
    if path.suffix.lower() in (".md", ".markdown", ".txt"):
        meta, body = parse_front_matter(raw)
    elif path.suffix.lower() in (".html", ".htm"):
        title, body = html_to_markdown(raw)
        meta["title"] = title
    elif path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PDF support needs: pip install pypdf") from e
        reader = PdfReader(str(path))
        body = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"unsupported file type: {path}")
    title = meta.get("title") or _first_heading(body) or path.stem.replace("-", " ").title()
    doc_id = path.stem
    return Document(doc_id=doc_id, title=title, text=body, agency=meta.get("agency", ""),
                    url=meta.get("url", ""), meta=meta)


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def load_directory(directory: Path) -> list[Document]:
    docs = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".md", ".markdown", ".txt", ".html", ".htm", ".pdf"):
            docs.append(load_document(p))
    return docs


def chunk_document(doc: Document, max_words: int = 160) -> list[Chunk]:
    """Heading-aware chunking: a chunk never spans two sections; long sections are packed by paragraph."""
    sections: list[tuple[str, list[str]]] = []
    current, paras, buf = doc.title, [], []
    for line in doc.text.splitlines():
        if line.startswith("#"):
            if buf:
                paras.append(" ".join(buf))
                buf = []
            if paras:
                sections.append((current, paras))
            current, paras = line.lstrip("#").strip(), []
        elif not line.strip():
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        paras.append(" ".join(buf))
    if paras:
        sections.append((current, paras))

    chunks: list[Chunk] = []
    for section, ps in sections:
        pack: list[str] = []
        words = 0
        for para in ps:
            n = len(para.split())
            if pack and words + n > max_words:
                chunks.append(_mk(doc, section, pack, len(chunks)))
                pack, words = [], 0
            pack.append(para)
            words += n
        if pack:
            chunks.append(_mk(doc, section, pack, len(chunks)))
    return chunks


def _mk(doc: Document, section: str, paras: list[str], i: int) -> Chunk:
    text = "\n".join(paras)
    return Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, title=doc.title, agency=doc.agency,
                 url=doc.url, section=section, text=text)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
