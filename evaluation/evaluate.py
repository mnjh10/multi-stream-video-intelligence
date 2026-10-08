def recall_at_k(
    results: list[str],
    ground_truth: str,
    k: int,
) -> float:
    """
    Return 1.0 if the ground-truth item appears
    within the top-k results, otherwise 0.0.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return float(
        ground_truth in results[:k]
    )


def reciprocal_rank(
    results: list[str],
    ground_truth: str,
) -> float:
    """
    Return the reciprocal rank of the ground-truth item.

    Returns 0.0 if the ground-truth item is not retrieved.
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result == ground_truth:
            return 1.0 / rank

    return 0.0


def evaluate_query(
    results: list[str],
    ground_truth: str,
) -> dict:
    """
    Evaluate one query.
    """

    return {
        "recall_at_1": recall_at_k(
            results,
            ground_truth,
            k=1,
        ),
        "recall_at_5": recall_at_k(
            results,
            ground_truth,
            k=5,
        ),
        "reciprocal_rank": reciprocal_rank(
            results,
            ground_truth,
        ),
    }


def evaluate_dataset(
    query_results: list[dict],
) -> dict:
    """
    Evaluate multiple queries.

    Each record must contain:

        results
        ground_truth
    """

    if not query_results:
        return {
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }

    evaluations = []

    for query in query_results:

        evaluations.append(
            evaluate_query(
                results=query["results"],
                ground_truth=query["ground_truth"],
            )
        )

    count = len(evaluations)

    return {
        "recall_at_1": sum(
            item["recall_at_1"]
            for item in evaluations
        ) / count,

        "recall_at_5": sum(
            item["recall_at_5"]
            for item in evaluations
        ) / count,

        "mrr": sum(
            item["reciprocal_rank"]
            for item in evaluations
        ) / count,
    }