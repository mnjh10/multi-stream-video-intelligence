import faiss
import numpy as np


class VectorIndex:
    """FAISS-based vector index for semantic retrieval."""

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Inner product on normalized vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # Maps FAISS vector positions to application metadata.
        self.metadata = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Add normalized embeddings and their metadata."""

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with dimension {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match number of metadata records."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Search the vector index with optional metadata filtering.

        Args:
            query_embedding: Normalized query embedding.
            top_k: Number of results to return.
            filters: Optional exact-match metadata filters.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"got {query_embedding.shape[1]}."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        candidate_indices = self._get_candidate_indices(filters)

        if not candidate_indices:
            return []

        candidate_embeddings = np.vstack(
            [
                self.index.reconstruct(index)
                for index in candidate_indices
            ]
        ).astype(np.float32)

        scores = query_embedding @ candidate_embeddings.T

        scores = scores[0]

        ranked_positions = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for position in ranked_positions:
            index = candidate_indices[position]

            results.append(
                {
                    "score": float(scores[position]),
                    "metadata": self.metadata[index],
                }
            )

        return results

    def _get_candidate_indices(
        self,
        filters: dict | None,
    ) -> list[int]:
        """Return FAISS vector positions matching metadata filters."""

        if not filters:
            return list(range(len(self.metadata)))

        candidate_indices = []

        for index, metadata in enumerate(self.metadata):

            matches = True

            for key, expected_value in filters.items():
                actual_value = metadata.get(key)
                if isinstance(expected_value, (set, list, tuple)):
                    if actual_value not in expected_value:
                        matches = False
                        break
                elif actual_value != expected_value:
                    matches = False
                    break

            if matches:
                candidate_indices.append(index)

        return candidate_indices

    def __len__(self) -> int:
        return self.index.ntotal