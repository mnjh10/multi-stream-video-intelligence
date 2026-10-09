"""
Backend service provider for TemporalFollowupEngine.
Encapsulates timeline-based temporal follow-ups.
"""

from pathlib import Path
from retrieval.temporal_followup import TemporalFollowupEngine

_temporal_engine_instance: TemporalFollowupEngine | None = None


def get_temporal_followup_engine() -> TemporalFollowupEngine:
    """
    Singleton getter for TemporalFollowupEngine.
    """
    global _temporal_engine_instance
    if _temporal_engine_instance is None:
        _temporal_engine_instance = TemporalFollowupEngine()
    return _temporal_engine_instance


def set_temporal_followup_engine(engine: TemporalFollowupEngine) -> None:
    """
    Override temporal engine (useful for testing).
    """
    global _temporal_engine_instance
    _temporal_engine_instance = engine
