import json
from pathlib import Path

import faiss


class IndexStorage:
    """
    Saves and loads a FAISS index together with its metadata.
    """

    def save(
        self,
        vector_index,
        index_path: str,
        metadata_path: str,
    ) -> None:
        """
        Save the FAISS index and metadata mapping to disk.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            vector_index.index,
            str(index_path),
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                vector_index.metadata,
                file,
                indent=2,
            )

    def load(
        self,
        vector_index,
        index_path: str,
        metadata_path: str,
    ) -> None:
        """
        Load a FAISS index and its metadata mapping from disk.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        vector_index.index = faiss.read_index(
            str(index_path)
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            vector_index.metadata = json.load(file)

        if vector_index.index.d != vector_index.dimension:
            raise ValueError(
                "Loaded FAISS index dimension does not "
                "match the configured vector dimension."
            )

        if vector_index.index.ntotal != len(
            vector_index.metadata
        ):
            raise ValueError(
                "FAISS vector count does not match "
                "metadata count."
            )