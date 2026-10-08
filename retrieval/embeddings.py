import torch
import open_clip


class CLIPEmbedder:
    """Creates image and text embeddings using OpenCLIP."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Convert text queries into normalized embeddings."""

        tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)

        embeddings /= embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu()

    def encode_image(self, image) -> torch.Tensor:
        """Convert an image into a normalized embedding."""

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)

        embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu()