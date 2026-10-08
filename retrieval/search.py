from typing import Any

import numpy as np

from retrieval.embeddings import CLIPEmbedder
from retrieval.vector_index import VectorIndex


class RetrievalEngine:
    """
    Coordinates query embedding and vector search.

    Person 3 should interact with this layer rather than
    directly interacting with FAISS.
    """

    def __init__(
        self,
        embedder: CLIPEmbedder,
        vector_index: VectorIndex,
    ):
        self.embedder = embedder
        self.vector_index = vector_index

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search indexed visual embeddings using a natural-language query.

        Args:
            query: Natural-language search query.
            filters: Optional structured metadata filters.
            top_k: Maximum number of results.

        Returns:
            Ranked retrieval results.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedder.encode_text([query])

        results = self.vector_index.search(
            query_embedding.numpy(),
            top_k=top_k,
        )

        if filters:
            results = self._apply_filters(results, filters)

        return results

    @staticmethod
    def _apply_filters(
        results: list[dict],
        filters: dict[str, Any],
    ) -> list[dict]:
        """Apply structured metadata filters to retrieved candidates."""

        filtered_results = []

        for result in results:
            metadata = result["metadata"]

            matches = True

            for key, expected_value in filters.items():
                actual_value = metadata.get(key)

                if actual_value != expected_value:
                    matches = False
                    break

            if matches:
                filtered_results.append(result)

        return filtered_results