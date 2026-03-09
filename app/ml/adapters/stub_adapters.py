from app.ml.adapters.base import SourceAdapter


class UnavailableAdapter(SourceAdapter):
    def __init__(self, source_name: str, setup_hint: str) -> None:
        self.source_name = source_name
        self.setup_hint = setup_hint

    def read(self, payload: dict) -> str:
        raise NotImplementedError(
            f"{self.source_name} source is not configured. Fallback to pdf/url/text."
        )

    def status(self) -> tuple[bool, str]:
        return False, self.setup_hint
