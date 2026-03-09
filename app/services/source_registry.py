from typing import Dict

from app.ml.adapters.base import SourceAdapter
from app.ml.adapters.pdf_adapter import PdfAdapter
from app.ml.adapters.stub_adapters import UnavailableAdapter
from app.ml.adapters.url_adapter import UrlAdapter


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: Dict[str, SourceAdapter] = {
            "pdf": PdfAdapter(),
            "url": UrlAdapter(),
            "text": UnavailableAdapter("text", "Inline text uses API direct payload."),
            "slack": UnavailableAdapter("slack", "Provide token + workspace export setup."),
            "discord": UnavailableAdapter("discord", "Provide bot token + channel scope."),
            "github": UnavailableAdapter("github", "Provide PAT + repo connector settings."),
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
