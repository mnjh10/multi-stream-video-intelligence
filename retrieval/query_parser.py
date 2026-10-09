import re


OBJECT_TYPES = {
    "person",
    "car",
    "vehicle",
    "bicycle",
    "backpack",
    "suitcase",
    "dog",
    "truck",
    "bus",
    "motorcycle",
}


class QueryParser:
    """Converts natural-language retrieval queries into structured filters."""

    def parse(self, query: str) -> dict:
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        result = {
            "raw_query": query,
            "object_type": self._extract_object_type(query),
            "attributes": self._extract_attributes(query),
            "camera_id": self._extract_camera_id(query),
            "location": self._extract_location(query),
            "time_range": self._extract_time_range(query),
            "action": self._extract_action(query),
            "reference": self._extract_reference(query),
        }

        return result

    def _extract_object_type(self, query: str) -> str | None:
        query_lower = query.lower()

        for object_type in OBJECT_TYPES:
            if object_type == "bus":
                pattern = r"\bbus(?:es)?\b"
            elif object_type == "person":
                pattern = r"\b(?:person|people)\b"
            else:
                pattern = rf"\b{re.escape(object_type)}s?\b"

            if re.search(pattern, query_lower):
                return object_type

        return None

    def _extract_attributes(self, query: str) -> list[str]:
        query_lower = query.lower()

        attributes = []

        known_attributes = [
            "red",
            "blue",
            "green",
            "white",
            "black",
            "yellow",
            "purple",
            "orange",
            "gray",
            "grey",
            "silver",
            "large",
            "small",
        ]

        for attribute in known_attributes:
            if re.search(
                rf"\b{re.escape(attribute)}\b",
                query_lower,
            ):
                attributes.append(attribute)

        return attributes

    def _extract_camera_id(self, query: str) -> str | None:
        """
        Extract camera references such as:
        'camera 3' -> 'cam_03'
        'camera 05' -> 'cam_05'
        'camera 5' -> 'cam_05'
        'cam_05' -> 'cam_05'
        'cam 05' -> 'cam_05'
        'cam 5' -> 'cam_05'
        'cam05' -> 'cam_05'
        'camera five' -> 'cam_05'
        """
        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }

        match = re.search(
            r"\b(?:camera|cam)[_\s-]*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            query.lower(),
        )

        if match:
            token = match.group(1)
            num = int(token) if token.isdigit() else word_to_num.get(token)
            if num is not None and 1 <= num <= 99:
                return f"cam_{num:02d}"

        return None

    def _extract_location(self, query: str) -> dict | None:
        query_lower = query.lower()

        # Camera references are structured metadata,
        # not semantic locations.
        if re.search(
            r"\b(?:camera|cam)[_\s-]*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
            query_lower,
        ):
            return None

        patterns = [
            r"\bnear\s+(?:the\s+)?(.+?)(?=\s+(?:in|during|within|from|at|on)\b|$)",
            r"\bat\s+(?:the\s+)?(.+?)(?=\s+(?:in|during|within|from|on)\b|$)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                query_lower,
            )

            if match:
                location = match.group(1).strip()

                return {
                    "value": location,
                    "resolved": False,
                }

        return None

    def _extract_time_range(self, query: str) -> dict | None:
        query_lower = query.lower()

        relative_patterns = [
            r"last\s+(minute|hour|day|week)",
            r"past\s+(minute|hour|day|week)",
        ]

        for pattern in relative_patterns:
            match = re.search(
                pattern,
                query_lower,
            )

            if match:
                return {
                    "type": "relative",
                    "value": match.group(0),
                }

        return None

    def _extract_action(self, query: str) -> str | None:
        query_lower = query.lower()

        actions = [
            "pass through",
            "enter",
            "exit",
            "cross",
            "park",
            "leave",
            "appear",
        ]

        for action in actions:
            if action in query_lower:
                return action

        return None

    def _extract_reference(self, query: str) -> dict | None:
        query_lower = query.lower()

        if "previous result" in query_lower:
            return {
                "type": "previous_result",
                "entity_id": None,
            }

        if re.search(
            r"\bthis\s+(car|person|vehicle|object)\b",
            query_lower,
        ):
            return {
                "type": "previous_result",
                "entity_id": None,
            }

        return None