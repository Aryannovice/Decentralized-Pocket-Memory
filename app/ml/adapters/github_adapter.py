from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx

from app.ml.adapters.base import SourceAdapter


class GitHubAdapter(SourceAdapter):
    """Adapter for public GitHub resources.

    Supports:
    * raw.githubusercontent.com URLs (passes through)
    * blob URLs pointing at a single file (converted to raw)
    * repository root URLs (fetch README + top-level markdown/text files via API)
    * issue/discussion/community/other pages (fetch HTML text)

    A GitHub personal access token may be supplied via the
    ``GITHUB_TOKEN`` environment variable to increase API rate limits when
    enumerating repository contents. The token is optional; the code will
    still work unauthenticated for public repositories.
    """

    source_name = "github"

    def read(self, payload: dict) -> str:
        github_url = payload.get("url")
        if not github_url:
            raise ValueError("Missing url for github source.")

        parsed = urlparse(github_url)
        host = parsed.netloc.lower()
        path = parsed.path.strip("/")
        parts = path.split("/") if path else []

        # raw host: just fetch and return text
        if host == "raw.githubusercontent.com":
            return self._fetch_raw(github_url)

        if host not in {"github.com", "www.github.com"}:
            raise ValueError("GitHub source expects a github.com or raw.githubusercontent.com URL.")

        # file blob
        if len(parts) >= 5 and parts[2] == "blob":
            raw = self._to_raw_url(github_url)
            return self._fetch_raw(raw)

        # repository root
        if len(parts) == 2:
            owner, repo = parts[0], parts[1]
            return self._read_repo(owner, repo)

        # anything else: fetch the HTML page (issue, discussion, community, etc.)
        return self._fetch_html(github_url)

    # helper methods
    def _fetch_raw(self, raw_url: str) -> str:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(raw_url, headers=self._auth_header())
            response.raise_for_status()
        return response.text

    def _fetch_html(self, url: str) -> str:
        # some GitHub pages (issues/discussions) render server-side; grabbing the
        # HTML is generally sufficient for chunking/distilling.
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers=self._auth_header())
            response.raise_for_status()
        return response.text

    def _read_repo(self, owner: str, repo: str) -> str:
        """Fetch README plus a few top-level text files from a repository."""
        text_parts: list[str] = []
        # always attempt README via raw
        text_parts.append(f"Repository: {owner}/{repo}\n")
        readme_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md"
        try:
            text_parts.append(self._fetch_raw(readme_raw))
        except Exception:
            # ignore if README not present
            pass

        # try listing top-level contents via GitHub API; this may be rate limited
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(api_url, headers=self._auth_header())
            if resp.status_code == 200:
                for item in resp.json():
                    if item.get("type") == "file":
                        name = item.get("name", "").lower()
                        if name.endswith((".md", ".txt", ".rst")) and item.get("download_url"):
                            try:
                                text_parts.append("\n\n" + self._fetch_raw(item["download_url"]))
                            except Exception:
                                continue
        return "\n".join(text_parts).strip()

    def _to_raw_url(self, github_url: str) -> str:
        # This helper is only called for blob URLs so validation has occurred
        parsed = urlparse(github_url)
        path = parsed.path.strip("/")
        parts = path.split("/") if path else []
        owner, repo = parts[0], parts[1]
        branch = parts[3]
        file_path = "/".join(parts[4:])
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    def _auth_header(self) -> dict[str, str]:
        token = os.environ.get("GITHUB_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}