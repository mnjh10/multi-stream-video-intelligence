from retrieval.retrieval_result import RetrievalResult


class ResultBuilder:
    """
    Converts temporal events into the final retrieval result contract.
    """

    def build(
        self,
        events: list[dict],
    ) -> list[RetrievalResult]:
        """
        Convert temporal events into RetrievalResult objects.
        """

        results = []

        for event in events:

            if not event["observations"]:
                continue

            best_observation = max(
                event["observations"],
                key=lambda observation: observation.get(
                    "score",
                    0.0,
                ),
            )

            result = RetrievalResult(
                event_id=event["event_id"],
                camera_id=event["camera_id"],
                object_id=event["object_id"],
                timestamp_start=event["timestamp_start"],
                timestamp_end=event["timestamp_end"],
                best_timestamp=event["best_timestamp"],
                score=event["best_score"],
                object_type=best_observation["object_type"],
                source_video=best_observation["source_video"],
            )

            results.append(result)

        return results