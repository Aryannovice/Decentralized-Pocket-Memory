from abc import ABC, abstractmethod


class SourceAdapter(ABC):
    source_name: str = "base"

    @abstractmethod
    def read(self, payload: dict) -> str:
        raise NotImplementedError

    def status(self) -> tuple[bool, str]:
        return True, "ready"
