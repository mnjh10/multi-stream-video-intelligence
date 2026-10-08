import numpy as np

from retrieval.index_storage import IndexStorage
from retrieval.observation import Observation
from retrieval.observation_encoder import ObservationEncoder
from retrieval.vector_index import VectorIndex


class ObservationIndexBuilder:
    """
    Builds and persists a FAISS vector index from Person 1 observations.
    """

    def __init__(
        self,
        encoder: ObservationEncoder,
        embedding_dimension: int = 512,
    ):
        self.encoder = encoder

        self.vector_index = VectorIndex(
            dimension=embedding_dimension,
        )

        self.storage = IndexStorage()

    def add_observations(
        self,
        observations: list[Observation],
    ) -> None:
        """Encode observations and add them to the vector index."""

        if not observations:
            return

        embeddings = []
        metadata = []

        for observation in observations:
            embedding, observation_metadata = self.encoder.encode(
                observation
            )

            embeddings.append(
                embedding.numpy()[0]
            )

            metadata.append(
                observation_metadata
            )

        embeddings_array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.vector_index.add(
            embeddings=embeddings_array,
            metadata=metadata,
        )

    def save(
        self,
        index_path: str,
        metadata_path: str,
    ) -> None:
        """Persist the current vector index."""

        self.storage.save(
            vector_index=self.vector_index,
            index_path=index_path,
            metadata_path=metadata_path,
        )

    def load(
        self,
        index_path: str,
        metadata_path: str,
    ) -> None:
        """Load a previously persisted vector index."""

        self.storage.load(
            vector_index=self.vector_index,
            index_path=index_path,
            metadata_path=metadata_path,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search the observation index."""

        return self.vector_index.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

    def __len__(self) -> int:
        return len(self.vector_index)