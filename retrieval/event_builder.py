from retrieval.temporal import TemporalEventGrouper


class EventBuilder:
    """
    Converts ranked observation candidates into temporal events.
    """

    def __init__(
        self,
        max_gap: float = 2.0,
    ):
        self.grouper = TemporalEventGrouper(
            max_gap=max_gap,
        )

    def build_events(
        self,
        candidates: list[dict],
    ) -> list[dict]:
        """
        Group observation candidates into temporal events.
        """

        if not candidates:
            return []

        observations = []

        for candidate in candidates:
            metadata = candidate["metadata"]

            observation = {
                **metadata,
                "score": candidate["final_score"],
            }

            observations.append(observation)

        return self.grouper.group(
            observations
        )