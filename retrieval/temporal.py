from collections import defaultdict

class TemporalEventGrouper:
    """
    Groups observations belonging to the same object
    into temporal events.
    """

    def __init__(self, max_gap: float = 2.0):
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative.")

        self.max_gap = max_gap

    def group(self, observations: list[dict]) -> list[dict]:
        """
        Group observations by camera and object ID.

        Each observation must contain:
            object_id
            camera_id
            timestamp
        """

        grouped = defaultdict(list)

        for observation in observations:
            key = (
                observation["camera_id"],
                observation["object_id"],
            )

            grouped[key].append(observation)

        events = []

        event_counter = 1

        for (camera_id, object_id), group in grouped.items():

            group.sort(
                key=lambda observation: observation["timestamp"]
            )

            current_group = []

            for observation in group:

                if not current_group:
                    current_group.append(observation)
                    continue

                previous_timestamp = current_group[-1]["timestamp"]

                gap = observation["timestamp"] - previous_timestamp

                if gap <= self.max_gap:
                    current_group.append(observation)

                else:
                    events.append(
                        self._create_event(
                            event_counter,
                            camera_id,
                            object_id,
                            current_group,
                        )
                    )

                    event_counter += 1

                    current_group = [observation]

            if current_group:
                events.append(
                    self._create_event(
                        event_counter,
                        camera_id,
                        object_id,
                        current_group,
                    )
                )

                event_counter += 1

        return events

    @staticmethod
    def _create_event(
        event_id: int,
        camera_id: str,
        object_id: str,
        observations: list[dict],
    ) -> dict:

        timestamps = [
            observation["timestamp"]
            for observation in observations
        ]

        best_observation = max(
            observations,
            key=lambda observation: observation.get(
                "score",
                0.0,
            ),
        )

        return {
            "event_id": f"evt_{event_id:06d}",
            "camera_id": camera_id,
            "object_id": object_id,
            "timestamp_start": min(timestamps),
            "timestamp_end": max(timestamps),
            "best_timestamp": best_observation["timestamp"],
            "best_score": best_observation.get(
                "score",
                0.0,
            ),
            "observations": observations,
        }