import html
import json
from pathlib import Path
import re
from typing import Any

from app.models.document import Document


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {
        ".md",
        ".markdown",
        ".txt",
        ".json",
        ".html",
        ".htm",
        ".yaml",
        ".yml",
    }

    def load(self, file_path: str) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        raw_text = path.read_text(encoding="utf-8")
        parsed_content, extra_meta = self._parse_by_extension(raw_text, extension, path.stem)

        return Document(
            document_id=path.stem,
            source=str(path),
            source_type=extension.lstrip("."),
            title=path.stem,
            content=parsed_content,
            metadata=extra_meta,
        )

    def load_text(
        self,
        title: str,
        content: str,
        source_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Load a document directly from in-memory string.
        """
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", title.lower()).strip("_") or "doc"
        return Document(
            document_id=doc_id,
            source="inline",
            source_type=source_type,
            title=title,
            content=content,
            metadata=metadata or {},
        )

    def _parse_by_extension(
        self,
        raw_text: str,
        extension: str,
        fallback_title: str,
    ) -> tuple[str, dict[str, Any]]:
        meta: dict[str, Any] = {}

        if extension in {".md", ".markdown", ".txt"}:
            # Extract possible title from first H1
            h1_match = re.search(r"^#\s+(.+)$", raw_text, flags=re.MULTILINE)
            if h1_match:
                meta["title"] = h1_match.group(1).strip()
            return raw_text, meta

        elif extension == ".json":
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    title = data.get("title", data.get("name", fallback_title))
                    meta["title"] = title
                    # Flatten into structured markdown text
                    lines = [f"# {title}\n"]
                    for k, v in data.items():
                        if k not in {"title", "name"}:
                            lines.append(f"## {k.replace('_', ' ').title()}\n{v}\n")
                    return "\n".join(lines), meta
                elif isinstance(data, list):
                    lines = [f"# {fallback_title}\n"]
                    for idx, item in enumerate(data):
                        lines.append(f"## Item {idx + 1}\n{json.dumps(item, indent=2)}\n")
                    return "\n".join(lines), meta
            except json.JSONDecodeError:
                return raw_text, meta

        elif extension in {".html", ".htm"}:
            # Extract title if present
            title_match = re.search(r"<title>(.*?)</title>", raw_text, flags=re.IGNORECASE | re.DOTALL)
            if title_match:
                meta["title"] = html.unescape(title_match.group(1).strip())

            # Remove scripts and style tags
            clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_text, flags=re.IGNORECASE | re.DOTALL)
            # Convert headings to markdown headings
            clean = re.sub(r"<h1[^>]*>(.*?)</h1>", r"# \1\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<h2[^>]*>(.*?)</h2>", r"## \1\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<h3[^>]*>(.*?)</h3>", r"### \1\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", clean, flags=re.IGNORECASE)
            # Strip remaining HTML tags
            clean = re.sub(r"<[^>]+>", "", clean)
            clean = html.unescape(clean)
            clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
            return clean, meta

        elif extension in {".yaml", ".yml"}:
            return raw_text, meta

        return raw_text, meta