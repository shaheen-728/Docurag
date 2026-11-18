from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader

from docurag.ingestion.types import ParsedSection

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def parse_document(
    path: Path,
    *,
    source_file: str,
) -> list[ParsedSection]:
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {extension}")

    if extension == ".pdf":
        sections = _parse_pdf(path, source_file)
    else:
        # Reject invalid UTF-8 instead of silently corrupting the text.
        content = path.read_text(encoding="utf-8-sig")

        if extension in {".md", ".markdown"}:
            html = markdown.markdown(
                content,
                extensions=["fenced_code", "tables"],
            )
            sections = _parse_html(html, source_file)

        elif extension in {".html", ".htm"}:
            sections = _parse_html(content, source_file)

        else:
            sections = [
                ParsedSection(
                    content=content.strip(),
                    source_file=source_file,
                )
            ]

    sections = [section for section in sections if section.content.strip()]

    if not sections:
        raise ValueError("No extractable text found. Scanned PDFs require OCR.")

    return sections


def _parse_pdf(path: Path, source_file: str) -> list[ParsedSection]:
    reader = PdfReader(path)

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported yet.")

    sections = []

    for page_number, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()

        if content:
            sections.append(
                ParsedSection(
                    content=content,
                    source_file=source_file,
                    page_number=page_number,
                )
            )

    return sections


def _parse_html(
    html: str,
    source_file: str,
) -> list[ParsedSection]:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        ["script", "style", "noscript", "nav", "header", "footer", "head"]
    ):
        element.decompose()

    root = soup.body or soup

    sections: list[ParsedSection] = []
    heading: str | None = None
    parts: list[str] = []

    def flush() -> None:
        content = "\n".join(parts).strip()

        if content:
            sections.append(
                ParsedSection(
                    content=content,
                    source_file=source_file,
                    section_heading=heading,
                )
            )

        parts.clear()

    for node in root.descendants:
        if isinstance(node, Tag) and node.name in HEADING_TAGS:
            flush()
            heading = node.get_text(" ", strip=True)

        elif isinstance(node, NavigableString):
            # Heading text is stored separately, not duplicated in the body.
            if any(parent.name in HEADING_TAGS for parent in node.parents):
                continue

            # Exclude comments and other special string subclasses.
            if type(node) is not NavigableString:
                continue

            value = str(node).strip()

            if value:
                parts.append(value)

    flush()

    return sections
