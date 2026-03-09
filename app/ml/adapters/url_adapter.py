import httpx

from app.ml.adapters.base import SourceAdapter


class UrlAdapter(SourceAdapter):
    source_name = "url"

    def read(self, payload: dict) -> str:
        url = payload.get("url")
        if not url:
            raise ValueError("Missing url for url source.")
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        return response.text
