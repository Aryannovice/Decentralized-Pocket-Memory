from typing import Dict

from app.ml.adapters.base import SourceAdapter
from app.ml.adapters.github_adapter import GitHubAdapter
from app.ml.adapters.pdf_adapter import PdfAdapter
from app.ml.adapters.reddit_adapter import RedditAdapter
from app.ml.adapters.stub_adapters import UnavailableAdapter
from app.ml.adapters.url_adapter import UrlAdapter


class InlineTextAdapter(SourceAdapter):
    source_name = "text"

    def read(self, payload: dict) -> str:
        # Inline text payload is passed directly via API.
        return str(payload.get("text", ""))

    def status(self) -> tuple[bool, str]:
        return True, "ready (inline via API payload)"


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: Dict[str, SourceAdapter] = {
            "pdf": PdfAdapter(),
            "url": UrlAdapter(),
            # Inline text ingestion is handled directly by the API; mark enabled for UI clarity.
            "text": InlineTextAdapter(),
            "reddit": RedditAdapter(),
            "slack": UnavailableAdapter("slack", "Provide token + workspace export setup."),
            "discord": UnavailableAdapter("discord", "Provide bot token + channel scope."),
            "github": GitHubAdapter(),
        }

    def get(self, source_name: str) -> SourceAdapter:
        if source_name not in self._sources:
            return UnavailableAdapter(source_name, "Unknown source, fallback to pdf/url/text.")
        return self._sources[source_name]

    def status(self) -> dict:
        result = {}
        for name, adapter in self._sources.items():
            enabled, message = adapter.status()
            result[name] = {"enabled": enabled, "message": message}
        return result
