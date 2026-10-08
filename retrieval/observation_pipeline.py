from retrieval.event_builder import EventBuilder
from retrieval.index_builder import ObservationIndexBuilder
from retrieval.observation import Observation
from retrieval.observation_encoder import ObservationEncoder
from retrieval.pipeline import RetrievalPipeline
from retrieval.retrieval_result import RetrievalResult
from retrieval.result_builder import ResultBuilder


class ObservationRetrievalPipeline:
    """
    Complete observation-to-event retrieval pipeline.

    Flow:

        Person 1 observations
                ↓
        Object crop embeddings
                ↓
        FAISS
                ↓
        Natural-language query
                ↓
        Query parsing + filtering
                ↓
        Semantic ranking
                ↓
        Temporal event grouping
                ↓
        RetrievalResult
    """

    def __init__(
        self,
        encoder: ObservationEncoder,
        retrieval_pipeline: RetrievalPipeline,
        max_gap: float = 2.0,
    ):
        self.index_builder = ObservationIndexBuilder(
            encoder=encoder,
            embedding_dimension=512,
        )

        self.retrieval_pipeline = retrieval_pipeline

        self.event_builder = EventBuilder(
            max_gap=max_gap,
        )

        self.result_builder = ResultBuilder()

    def index_observations(
        self,
        observations: list[Observation],
    ) -> None:
        """
        Encode and index Person 1 observations.
        """

        self.index_builder.add_observations(
            observations
        )

        self.retrieval_pipeline.retrieval_engine.vector_index = (
            self.index_builder.vector_index
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Search indexed observations and return temporal
        retrieval results.

        To ensure high-quality temporal event grouping and return up to
        `top_k` distinct events, an expanded pool of observation candidates
        is retrieved before grouping into events and ranking.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        pool_size = (
            candidate_pool_size
            if candidate_pool_size is not None
            else max(top_k * 10, 50)
        )

        candidates = self.retrieval_pipeline.search(
            query=query,
            top_k=pool_size,
            filters=filters,
        )

        events = self.event_builder.build_events(
            candidates
        )

        events.sort(
            key=lambda event: event.get("best_score", 0.0),
            reverse=True,
        )

        results = self.result_builder.build(
            events
        )

        return results[:top_k]

    def __len__(self) -> int:
        return len(self.index_builder)