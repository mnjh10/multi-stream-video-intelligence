from dataclasses import dataclass


@dataclass
class ParsedQuery:
    object_type: str | None = None
    attributes: dict | None = None
    location: str | None = None
    time_range: dict | None = None
    reference: str | None = None


def parse_query(query_text: str) -> ParsedQuery:
    """
    Lightweight query parser boundary.

    This is intentionally a mock/stub. A real NLP or retrieval
    implementation can replace this later without changing the API.
    """
    text = query_text.lower()

    object_type = None

    for candidate in [
        "person",
        "car",
        "bicycle",
        "backpack",
        "dog",
        "suitcase",
    ]:
        if candidate in text:
            object_type = candidate
            break

    reference = None

    if "main gate" in text:
        reference = "main gate"

    return ParsedQuery(
        object_type=object_type,
        attributes={},
        location=None,
        time_range=None,
        reference=reference,
    )