"""Splits the markdown knowledge base into retrievable chunks.

We split on the ``##`` section headings rather than on a fixed number of
characters.  The knowledge base is written as short, self-contained policy
sections, so a section *is* the natural unit of retrieval: it never cuts a
rule in half, and the heading itself is useful context for the language model.
"""

from pathlib import Path

# Markdown files in the knowledge base folder that are documentation *about*
# the corpus rather than part of it. Indexing these would let developer notes
# be retrieved and quoted back to a customer as if they were policy.
EXCLUDED_FILE_NAMES = {"readme.md"}


class KnowledgeChunk:
    """One retrievable section of a knowledge base document."""

    def __init__(self, chunk_id: str, source: str, document_title: str, heading: str, text: str):
        self.chunk_id = chunk_id
        # File name the chunk came from, so answers can be traced back.
        self.source = source
        # The "# " title at the top of the document.
        self.document_title = document_title
        # The "## " heading of this section.
        self.heading = heading
        # The body text under the heading.
        self.text = text

    def searchable_text(self) -> str:
        """The text used to build the vector for this chunk.

        The headings are included because they carry strong keywords such as
        "refund window" or "escalation rules".
        """
        return self.document_title + ". " + self.heading + ". " + self.text

    def as_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "document_title": self.document_title,
            "heading": self.heading,
            "text": self.text,
        }


def chunk_markdown_document(path: Path) -> list[KnowledgeChunk]:
    """Split one markdown file into one chunk per ``##`` section."""
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.split("\n")

    document_title = path.stem
    current_heading = "Introduction"
    current_lines: list[str] = []
    chunks: list[KnowledgeChunk] = []

    def flush_current_section() -> None:
        """Turn whatever has been collected so far into a chunk."""
        body = "\n".join(current_lines).strip()
        if body == "":
            return
        chunk_id = path.stem + "#" + str(len(chunks) + 1)
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                source=path.name,
                document_title=document_title,
                heading=current_heading,
                text=body,
            )
        )

    for line in lines:
        if line.startswith("## "):
            # A new section starts: store the previous one first.
            flush_current_section()
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            # The document title line.
            document_title = line[2:].strip()
        else:
            current_lines.append(line)

    # The final section has no heading after it to trigger the flush.
    flush_current_section()
    return chunks


def chunk_knowledge_base(directory: Path) -> list[KnowledgeChunk]:
    """Chunk every markdown file in the knowledge base folder."""
    if not directory.exists():
        raise FileNotFoundError("Knowledge base folder not found at " + str(directory))

    all_chunks: list[KnowledgeChunk] = []
    # sorted() keeps chunk ids stable between runs.
    for path in sorted(directory.glob("*.md")):
        # The folder's own README documents the knowledge base for developers;
        # it is not policy the agent may quote to a customer, so it is skipped.
        if path.name.lower() in EXCLUDED_FILE_NAMES:
            continue
        all_chunks.extend(chunk_markdown_document(path))

    if len(all_chunks) == 0:
        raise ValueError("No markdown chunks found in " + str(directory))
    return all_chunks
