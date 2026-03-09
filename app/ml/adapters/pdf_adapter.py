from pathlib import Path

from pypdf import PdfReader

from app.ml.adapters.base import SourceAdapter


class PdfAdapter(SourceAdapter):
    source_name = "pdf"

    def read(self, payload: dict) -> str:
        file_path = payload.get("file_path")
        if not file_path:
            raise ValueError("Missing file_path for pdf source.")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
