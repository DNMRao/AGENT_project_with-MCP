"""Load Markdown knowledge-base files (chunking comes next)."""

from dataclasses import dataclass
from pathlib import Path

import re

# Matches a whole line like "Source ID: refunds" and captures "refunds".
SOURCE_ID_RE = re.compile(r"^Source ID:\s*(?P<source_id>[a-z0-9-]+)\s*$", re.MULTILINE)


KB_DIR = Path("data/knowledge_base")



@dataclass(frozen=True)
class Document:
    document_id: str  # filename without ".md", used in citations
    title: str  # text of the "# ..." heading line
    body: str  # everything after the heading


def parse_markdown(document_id: str, raw: str) -> Document:
    """Split one Markdown file into its title line and body text."""
    lines = raw.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{document_id}: first line must be a '# Title' heading")
    title = lines[0].removeprefix("# ").strip()
    body = "\n".join(lines[1:]).strip()
    match = SOURCE_ID_RE.search(body)
    if match is None:
        raise ValueError(f"{document_id}: missing 'Source ID:' line")
    if match["source_id"] != document_id:
        raise ValueError(
            f"{document_id}: Source ID '{match['source_id']}' does not match the filename"
        )
    body = SOURCE_ID_RE.sub("", body).strip()   # remove the line before chunking

    if not body:
        raise ValueError(f"{document_id}: document has no body text")
    return Document(document_id=document_id, title=title, body=body)


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    """Read every .md file in kb_dir, in a fixed (alphabetical) order."""
    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown files found in {kb_dir.resolve()}")
    return [parse_markdown(p.stem, p.read_text(encoding="utf-8")) for p in paths]


@dataclass(frozen=True)
class Chunk:
    document_id: str  # which file it came from -> used in citations
    title: str  # document title, kept as metadata
    position: int  # 0, 1, 2 ... paragraph index within the document
    text: str  # what gets embedded: title + paragraph

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}#{self.position}"  # e.g. "billing-refunds#1"


def split_paragraphs(body: str) -> list[str]:
    """Split on blank lines and join wrapped lines into one line per paragraph."""
    paragraphs = (" ".join(p.split()) for p in body.split("\n\n"))
    return [p for p in paragraphs if p]


def build_chunks(docs: list[Document]) -> list[Chunk]:
    """One chunk per paragraph, with the title prepended to the text."""
    return [
        Chunk(
            document_id=doc.document_id,
            title=doc.title,
            position=i,
            text=f"{doc.title}\n{para}",
        )
        for doc in docs
        for i, para in enumerate(split_paragraphs(doc.body))
    ]


if __name__ == "__main__":
    docs = load_documents()
    chunks = build_chunks(docs)
    print(f"{len(chunks)} chunks from {len(docs)} documents\n")
    for doc in docs:
        n = sum(c.document_id == doc.document_id for c in chunks)
        print(f"{doc.document_id:28} {n} chunks  | {doc.title}")
    print("\nSample chunk:")
    print(chunks[0])
