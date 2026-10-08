from typing import Any

from retrieval.query_parser import QueryParser
from retrieval.ranking import ResultRanker
from retrieval.search import RetrievalEngine


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline for observation candidates.

    Flow:

        Natural-language query
                ↓
        Query parsing
                ↓
        Metadata filtering
                ↓
        Semantic retrieval
                ↓
        Query-aware ranking
                ↓
        Observation candidates
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        query_parser: QueryParser | None = None,
        ranker: ResultRanker | None = None,
    ):
        self.retrieval_engine = retrieval_engine
        self.query_parser = query_parser or QueryParser()
        self.ranker = ranker or ResultRanker()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Search indexed observations and return ranked candidates.
        """

        parsed_query = self.query_parser.parse(query)

        active_filters = self._build_filters(
            parsed_query
        )
        if filters:
            active_filters.update(filters)

        candidates = self.retrieval_engine.search(
            query=query,
            filters=active_filters,
            top_k=top_k,
        )

        ranked_candidates = self.ranker.rank(
            candidates,
            parsed_query,
        )

        return ranked_candidates[:top_k]

    @staticmethod
    def _build_filters(
        parsed_query: dict[str, Any],
    ) -> dict[str, Any]:

        filters = {}

        object_type = parsed_query.get(
            "object_type"
        )

        if object_type:
            if object_type == "vehicle":
                filters["object_type"] = ("car", "truck", "bus", "motorcycle", "vehicle")
            else:
                filters["object_type"] = object_type

        camera_id = parsed_query.get(
            "camera_id"
        )

        if camera_id:
            filters["camera_id"] = camera_id

        return filters