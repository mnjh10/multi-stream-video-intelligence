from abc import ABC, abstractmethod
from datetime import datetime
import os
from pathlib import Path

from retrieval.embeddings import CLIPEmbedder
from retrieval.index_storage import IndexStorage
from retrieval.observation_encoder import ObservationEncoder
from retrieval.observation_loader import load_observations
from retrieval.observation_pipeline import ObservationRetrievalPipeline
from retrieval.pipeline import RetrievalPipeline
from retrieval.search import RetrievalEngine


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
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return [
            {
                "event_id": "mock-event-001",
                "camera_id": "cam_01",
                "timestamp": datetime.now(),
                "best_timestamp": 0.0,
                "timestamp_start": 0.0,
                "timestamp_end": 0.0,
                "score": 0.95,
                "object_id": "obj_mock_001",
                "object_type": "car",
                "evidence": {
                    "frame_path": None,
                    "clip_path": None,
                    "source_video": "data/videos/cam_01/vdo.avi",
                },
            }
        ][:top_k]


class RealRetrievalProvider(RetrievalProvider):
    """
    Real retrieval provider powered by Person 2's OpenCLIP + FAISS pipeline.
    """

    DEFAULT_INDEX_PATH = "data/index/argus.index"
    DEFAULT_METADATA_PATH = "data/index/argus_metadata.json"
    DEFAULT_OBSERVATIONS_PATH = "data/observations/observations.jsonl"

    def __init__(
        self,
        embedder: CLIPEmbedder | None = None,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        observations_path: str | Path | None = None,
        initial_observations_limit: int = 50,
        max_gap: float = 2.0,
    ):
        self.embedder = embedder or CLIPEmbedder()
        self.encoder = ObservationEncoder(embedder=self.embedder)
        self.retrieval_engine = RetrievalEngine(
            embedder=self.embedder,
            vector_index=None,
        )
        self.pipeline = ObservationRetrievalPipeline(
            encoder=self.encoder,
            retrieval_pipeline=RetrievalPipeline(
                retrieval_engine=self.retrieval_engine
            ),
            max_gap=max_gap,
        )

        self.index_path = Path(index_path or self.DEFAULT_INDEX_PATH)
        self.metadata_path = Path(metadata_path or self.DEFAULT_METADATA_PATH)
        self.observations_path = Path(
            observations_path or self.DEFAULT_OBSERVATIONS_PATH
        )

        self._initialize_index(initial_observations_limit)

    def _initialize_index(self, limit: int) -> None:
        """
        Load index from disk if available, or initialize from a slice of observations.
        """
        storage = IndexStorage()
        if self.index_path.exists() and self.metadata_path.exists():
            storage.load(
                self.pipeline.index_builder.vector_index,
                str(self.index_path),
                str(self.metadata_path),
            )
            self.retrieval_engine.vector_index = (
                self.pipeline.index_builder.vector_index
            )
        elif self.observations_path.exists():
            # Index a representative slice of real observations
            observations = load_observations(
                filepath=str(self.observations_path),
                limit=limit,
            )
            if observations:
                self.pipeline.index_observations(observations)
                try:
                    storage.save(
                        self.pipeline.index_builder.vector_index,
                        str(self.index_path),
                        str(self.metadata_path),
                    )
                except Exception:
                    pass

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search observations and return grouped temporal event results.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results = self.pipeline.search(
            query=query.strip(),
            top_k=top_k,
            filters=filters,
        )

        formatted_results = []
        for res in results:
            formatted_results.append(
                {
                    "event_id": res.event_id,
                    "camera_id": res.camera_id,
                    "timestamp": res.best_timestamp,
                    "best_timestamp": res.best_timestamp,
                    "timestamp_start": res.timestamp_start,
                    "timestamp_end": res.timestamp_end,
                    "score": float(res.score),
                    "object_id": res.object_id,
                    "object_type": res.object_type,
                    "evidence": {
                        "frame_path": None,
                        "clip_path": None,
                        "source_video": res.source_video,
                    },
                }
            )

        return formatted_results


_provider_instance: RetrievalProvider | None = None


def get_retrieval_provider() -> RetrievalProvider:
    """
    Singleton getter for the backend retrieval provider.
    """
    global _provider_instance
    if _provider_instance is None:
        use_mock = os.getenv("USE_MOCK_RETRIEVAL", "false").lower() in (
            "true",
            "1",
        )
        if use_mock:
            _provider_instance = MockRetrievalProvider()
        else:
            _provider_instance = RealRetrievalProvider()
    return _provider_instance


def set_retrieval_provider(provider: RetrievalProvider) -> None:
    """
    Override the global retrieval provider (useful for testing).
    """
    global _provider_instance
    _provider_instance = provider