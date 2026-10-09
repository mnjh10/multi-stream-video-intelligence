"""
Temporal Follow-Up Query Engine for ARGUS.
Enables follow-up questions that retrieve earlier or later observations
of the same tracked object within the same camera.
"""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


class TemporalFollowupEngine:
    """
    Resolves conversational temporal follow-up queries
    (e.g., 'Show me this car later', 'Where was this object 10 seconds earlier?')
    by tracking object timelines within camera-local coordinates.
    """

    def __init__(self, metadata_path: str | Path = "data/index/argus_metadata.json"):
        self.metadata_path = Path(metadata_path)
        self._timeline_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._camera_index: dict[str, list[dict]] = defaultdict(list)
        self._loaded = False
        self._last_context: dict[str, Any] | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.metadata_path.exists():
            return

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        for obs in metadata_list:
            cam_id = obs.get("camera_id")
            obj_id = obs.get("object_id")
            if cam_id:
                self._camera_index[cam_id].append(obs)
                if obj_id:
                    self._timeline_index[(cam_id, obj_id)].append(obs)

        # Sort all timelines chronologically
        for key in self._timeline_index:
            self._timeline_index[key].sort(key=lambda o: (o.get("timestamp", 0.0), o.get("frame_index", 0)))
        for cam_id in self._camera_index:
            self._camera_index[cam_id].sort(key=lambda o: (o.get("timestamp", 0.0), o.get("frame_index", 0)))

        self._loaded = True

    def find_observation(
        self,
        camera_id: str,
        object_id: str | None = None,
        timestamp: float | None = None,
        tolerance: float = 1.0,
    ) -> dict[str, Any] | None:
        """
        Find observation for a given camera and optional object nearest to timestamp.
        """
        self._ensure_loaded()
        if object_id:
            timeline = self._timeline_index.get((camera_id, object_id), [])
        elif camera_id in self._camera_index:
            timeline = self._camera_index[camera_id]
        else:
            return None

        if not timeline:
            return None

        if timestamp is None:
            return timeline[0]

        nearest = min(timeline, key=lambda o: abs(o.get("timestamp", 0.0) - timestamp))
        if abs(nearest.get("timestamp", 0.0) - timestamp) <= tolerance:
            return nearest
        return None

    def set_last_context(self, context: dict[str, Any] | None) -> None:
        """Store the context of the most recent query result."""
        self._last_context = dict(context) if context else None

    def get_last_context(self) -> dict[str, Any] | None:
        """Retrieve the context of the most recent query result."""
        return self._last_context

    def is_followup_query(self, query: str, context: dict[str, Any] | None = None) -> bool:
        """
        Determine whether a query is a temporal follow-up.
        """
        q_lower = query.lower().strip()

        # Follow-up indicators
        has_temporal_dir = bool(
            re.search(
                r"\b(earlier|later|before|after|next\s+appearance|previous\s+appearance|prior\s+appearance|prior|subsequent|next|last|where\s+was|at\s+its\s+next)\b",
                q_lower,
            )
        )

        has_reference = bool(
            re.search(
                r"\b(this\s+(?:car|vehicle|person|object|truck)|same\s+(?:car|vehicle|person|object|truck)|it|this|that|the\s+same|the\s+object)\b",
                q_lower,
            )
            or re.search(r"\bobj_cam\d+_\d+\b", q_lower)
        )

        has_offset = bool(
            re.search(
                r"\d+(?:\.\d+)?\s*(?:seconds?|secs?|s|minutes?|mins?|m)\s*(?:earlier|later|before|after|prior|ago)",
                q_lower,
            )
        )

        # If query has an explicit object ID and temporal direction
        if re.search(r"\bobj_cam\d+_\d+\b", q_lower) and has_temporal_dir:
            return True

        # If query has reference + temporal direction or offset
        if (has_reference or has_offset) and has_temporal_dir:
            return True

        # If context is explicitly provided and query asks for temporal relation
        active_ctx = context or self._last_context
        if active_ctx and (has_temporal_dir or has_offset):
            # Check if query is not a fresh general search
            if not re.search(r"\b(find|search|look for|detect)\s+(?:a|an|all)?\s+(?:car|truck|bus|vehicle|person|dog|bicycle)\b", q_lower):
                return True
            if has_reference:
                return True

        return False

    def parse_temporal_intent(self, query: str) -> tuple[str, float | None]:
        """
        Extract temporal direction ('earlier' or 'later') and optional numeric offset in seconds.
        """
        q_lower = query.lower().strip()

        # Check for numeric offset: e.g. "10 seconds earlier", "5s later", "20 secs before"
        offset_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)?\s*(earlier|later|before|after|ago|prior)",
            q_lower,
        )

        direction = "later"
        offset_seconds: float | None = None

        if offset_match:
            val = float(offset_match.group(1))
            unit = (offset_match.group(2) or "s").lower()
            rel = offset_match.group(3).lower()

            mult = 60.0 if "m" in unit else 1.0
            total_seconds = val * mult

            if rel in ("earlier", "before", "ago", "prior"):
                direction = "earlier"
                offset_seconds = -total_seconds
            else:
                direction = "later"
                offset_seconds = total_seconds
            return direction, offset_seconds

        # Check for directional keywords
        if any(w in q_lower for w in ["earlier", "before", "previous", "prior"]):
            direction = "earlier"
        elif any(w in q_lower for w in ["later", "after", "next", "subsequent"]):
            direction = "later"

        return direction, None

    def resolve_followup(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Resolve a follow-up query to earlier or later observations of the target object.
        """
        self._ensure_loaded()
        q_lower = query.lower().strip()

        # Determine target object and camera
        obj_match = re.search(r"\b(obj_cam\d+_\d+)\b", q_lower)
        active_ctx = context or self._last_context or {}

        object_id: str | None = None
        camera_id: str | None = None
        ref_timestamp: float | None = None

        if obj_match:
            object_id = obj_match.group(1)
            # Camera ID can be deduced from object ID format: obj_camXX_YYYYYY
            cam_match = re.search(r"obj_(cam_\d+)_", object_id)
            if cam_match:
                camera_id = cam_match.group(1)
        elif active_ctx.get("object_id"):
            object_id = active_ctx.get("object_id")
            camera_id = active_ctx.get("camera_id")
            ref_timestamp = active_ctx.get("timestamp") or active_ctx.get("best_timestamp")

        if not object_id or not camera_id:
            return {
                "status": "error",
                "message": (
                    "Previous object context is unavailable or ambiguous for temporal follow-up. "
                    "Please search for an object first or specify an object ID."
                ),
                "results": [],
            }

        timeline = self._timeline_index.get((camera_id, object_id), [])
        if not timeline:
            return {
                "status": "no_match",
                "message": f"No observations found for tracked object {object_id} in {camera_id}.",
                "results": [],
            }

        # Determine base reference timestamp if not provided
        if ref_timestamp is None:
            ref_timestamp = timeline[0].get("timestamp", 0.0)

        direction, offset = self.parse_temporal_intent(query)

        # Select candidate observations matching temporal direction/offset
        selected_obs: dict[str, Any] | None = None
        relation_label = ""

        if offset is not None:
            target_time = ref_timestamp + offset
            if offset < 0:
                candidates = [o for o in timeline if o.get("timestamp", 0.0) < ref_timestamp]
                relation_label = f"{abs(offset):.1f}s earlier"
            else:
                candidates = [o for o in timeline if o.get("timestamp", 0.0) > ref_timestamp]
                relation_label = f"{offset:.1f}s later"

            if candidates:
                selected_obs = min(
                    candidates,
                    key=lambda o: abs(o.get("timestamp", 0.0) - target_time),
                )
        else:
            # Directional jump (next or previous distinct appearance)
            if direction == "earlier":
                # Find observations earlier than ref_timestamp by at least 0.1s
                candidates = [o for o in timeline if o.get("timestamp", 0.0) < (ref_timestamp - 0.1)]
                if candidates:
                    selected_obs = max(candidates, key=lambda o: o.get("timestamp", 0.0))
                    time_diff = ref_timestamp - selected_obs.get("timestamp", 0.0)
                    relation_label = f"earlier appearance (-{time_diff:.1f}s)"
            else:
                # Find observations later than ref_timestamp by at least 0.1s
                candidates = [o for o in timeline if o.get("timestamp", 0.0) > (ref_timestamp + 0.1)]
                if candidates:
                    selected_obs = min(candidates, key=lambda o: o.get("timestamp", 0.0))
                    time_diff = selected_obs.get("timestamp", 0.0) - ref_timestamp
                    relation_label = f"next appearance (+{time_diff:.1f}s)"

        if not selected_obs:
            dir_text = "earlier" if direction == "earlier" else "later"
            return {
                "status": "no_match",
                "message": (
                    f"No {dir_text} observation found for object {object_id} in {camera_id} "
                    f"relative to timestamp {ref_timestamp:.1f}s."
                ),
                "results": [],
            }

        # Build formatted QueryResult
        obs_time = float(selected_obs.get("timestamp", 0.0))
        frame_idx = int(selected_obs.get("frame_index", 0))
        bbox = selected_obs.get("bbox", [])
        crop_path = selected_obs.get("crop_path")
        source_video = selected_obs.get("source_video")
        object_type = selected_obs.get("object_type", "object")
        obs_id = selected_obs.get("observation_id", "obs")

        result = {
            "event_id": f"evt_followup_{obs_id}",
            "camera_id": camera_id,
            "timestamp": obs_time,
            "best_timestamp": obs_time,
            "timestamp_start": obs_time,
            "timestamp_end": obs_time,
            "score": 1.0,
            "object_id": object_id,
            "object_type": object_type,
            "evidence": {
                "frame_path": None,
                "clip_path": None,
                "source_video": source_video,
            },
            "bbox": bbox,
            "frame_index": frame_idx,
            "crop_path": crop_path,
            "is_followup": True,
            "followup_relation": relation_label,
        }

        # Update last context to this newly matched observation
        self._last_context = {
            "camera_id": camera_id,
            "object_id": object_id,
            "timestamp": obs_time,
            "best_timestamp": obs_time,
            "event_id": result["event_id"],
            "bbox": bbox,
            "frame_index": frame_idx,
        }

        return {
            "status": "ok",
            "message": f"Retrieved {relation_label} for {object_type} ({object_id}) in {camera_id} at {obs_time:.2f}s.",
            "results": [result],
        }
