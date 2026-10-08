from abc import ABC, abstractmethod
from datetime import datetime


class RetrievalProvider(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        pass


class MockRetrievalProvider(RetrievalProvider):
    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        return [
            {
                "event_id": "mock-event-001",
                "camera_id": "cam_01",
                "timestamp": datetime.now(),
                "score": 0.95,
                "object_type": "car",
                "evidence": {
                    "frame_path": None,
                    "clip_path": None,
                },
            }
        ][:top_k]