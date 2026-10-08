from typing import Any


class ResultRanker:
    """
    Ranks retrieval candidates using semantic similarity
    plus structured query constraints.
    """

    def __init__(
        self,
        semantic_weight: float = 0.8,
        metadata_weight: float = 0.2,
    ):
        if semantic_weight < 0 or metadata_weight < 0:
            raise ValueError("Weights must be non-negative.")

        total = semantic_weight + metadata_weight

        if total == 0:
            raise ValueError("At least one weight must be greater than zero.")

        self.semantic_weight = semantic_weight / total
        self.metadata_weight = metadata_weight / total

    def rank(
        self,
        results: list[dict],
        query: dict[str, Any],
    ) -> list[dict]:
        """
        Rank retrieval candidates using semantic similarity
        and structured query constraints.
        """

        ranked_results = []

        for result in results:
            metadata = result["metadata"]

            semantic_score = float(result["score"])

            metadata_score = self._metadata_score(
                metadata,
                query,
            )

            final_score = (
                self.semantic_weight * semantic_score
                + self.metadata_weight * metadata_score
            )

            ranked_result = {
                **result,
                "semantic_score": semantic_score,
                "metadata_score": metadata_score,
                "final_score": final_score,
            }

            ranked_results.append(ranked_result)

        ranked_results.sort(
            key=lambda result: result["final_score"],
            reverse=True,
        )

        return ranked_results

    def _metadata_score(
        self,
        metadata: dict[str, Any],
        query: dict[str, Any],
    ) -> float:
        """
        Calculate how well candidate metadata matches
        structured query constraints.
        """

        constraints = 0
        matches = 0

        object_type = query.get("object_type")

        if object_type:
            constraints += 1

            actual_type = metadata.get("object_type")
            if actual_type == object_type or (
                object_type == "vehicle"
                and actual_type in {"car", "truck", "bus", "motorcycle"}
            ):
                matches += 1

        camera_id = query.get("camera_id")

        if camera_id:
            constraints += 1

            if metadata.get("camera_id") == camera_id:
                matches += 1

        if constraints == 0:
            return 0.0

        return matches / constraints