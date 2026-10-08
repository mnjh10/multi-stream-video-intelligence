from pathlib import Path

import numpy as np
from PIL import Image

from retrieval.embeddings import CLIPEmbedder
from retrieval.vector_index import VectorIndex


class BaselineRetriever:
    """
    Simple full-image semantic retrieval baseline.

    Baseline:

        Image
          ↓
        OpenCLIP
          ↓
        FAISS
          ↓
        Top-K
    """

    def __init__(
        self,
        embedder: CLIPEmbedder,
        embedding_dimension: int = 512,
    ):
        self.embedder = embedder

        self.vector_index = VectorIndex(
            dimension=embedding_dimension,
        )

    def index_images(
        self,
        image_records: list[dict],
    ) -> None:
        """
        Index full images.

        Each record must contain:

            image_path
            metadata
        """

        embeddings = []
        metadata = []

        for record in image_records:

            image_path = Path(
                record["image_path"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = Image.open(
                image_path
            ).convert("RGB")

            embedding = self.embedder.encode_image(
                image
            )

            embeddings.append(
                embedding.numpy()[0]
            )

            metadata.append(
                record["metadata"]
            )

        if not embeddings:
            return

        embeddings_array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.vector_index.add(
            embeddings=embeddings_array,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search the baseline index using a natural-language query.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_embedding = self.embedder.encode_text(
            [query]
        )

        return self.vector_index.search(
            query_embedding=query_embedding.numpy(),
            top_k=top_k,
        )