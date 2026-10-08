from pathlib import Path

import torch
from PIL import Image

from retrieval.embeddings import CLIPEmbedder
from retrieval.observation import Observation


class ObservationEncoder:
    """
    Converts Person 1 observations into OpenCLIP image embeddings.
    """

    def __init__(self, embedder: CLIPEmbedder):
        self.embedder = embedder

    def encode(
        self,
        observation: Observation,
    ) -> tuple[torch.Tensor, dict]:
        """
        Encode the object crop associated with an observation.

        Returns:
            embedding: Normalized OpenCLIP image embedding.
            metadata: Metadata associated with the embedding.
        """

        crop_path = Path(observation.crop_path)

        if not crop_path.exists():
            raise FileNotFoundError(
                f"Crop image not found: {crop_path}"
            )

        image = Image.open(crop_path).convert("RGB")

        embedding = self.embedder.encode_image(image)

        metadata = observation.to_dict()

        return embedding, metadata