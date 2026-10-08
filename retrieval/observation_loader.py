"""
Observation loader for Person 1 frozen observation dataset.
"""

from pathlib import Path
from typing import Iterator
import json

from retrieval.observation import Observation


def load_observations(
    jsonl_path: str | Path = "data/observations/observations.jsonl",
    limit: int | None = None,
    camera_id: str | None = None,
    object_type: str | None = None,
) -> list[Observation]:
    """
    Load Person 1 observation records from JSONL file into Observation dataclass instances.

    Args:
        jsonl_path: Path to observations.jsonl file.
        limit: Optional maximum number of records to load.
        camera_id: Optional filter by camera ID (e.g. 'cam_01').
        object_type: Optional filter by object type (e.g. 'car').

    Returns:
        List of Observation instances.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Observations file not found: {path}")

    observations: list[Observation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if camera_id and data.get("camera_id") != camera_id:
                continue
            if object_type and data.get("object_type") != object_type:
                continue
            observations.append(Observation(**data))
            if limit is not None and len(observations) >= limit:
                break

    return observations


def iter_observations(
    jsonl_path: str | Path = "data/observations/observations.jsonl",
) -> Iterator[Observation]:
    """
    Stream Person 1 observation records one by one to avoid large memory footprints.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Observations file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Observation(**json.loads(line))
