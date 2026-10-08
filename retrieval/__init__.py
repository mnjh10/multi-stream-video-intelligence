"""
ARGUS Semantic Retrieval Pipeline.
"""

from retrieval.embeddings import CLIPEmbedder
from retrieval.event_builder import EventBuilder
from retrieval.index_builder import ObservationIndexBuilder
from retrieval.index_storage import IndexStorage
from retrieval.observation import Observation
from retrieval.observation_encoder import ObservationEncoder
from retrieval.observation_loader import iter_observations, load_observations
from retrieval.observation_pipeline import ObservationRetrievalPipeline
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_parser import QueryParser
from retrieval.ranking import ResultRanker
from retrieval.result_builder import ResultBuilder
from retrieval.retrieval_result import RetrievalResult
from retrieval.search import RetrievalEngine
from retrieval.temporal import TemporalEventGrouper
from retrieval.vector_index import VectorIndex

__all__ = [
    "CLIPEmbedder",
    "EventBuilder",
    "IndexStorage",
    "Observation",
    "ObservationEncoder",
    "ObservationIndexBuilder",
    "ObservationRetrievalPipeline",
    "QueryParser",
    "ResultBuilder",
    "ResultRanker",
    "RetrievalEngine",
    "RetrievalPipeline",
    "RetrievalResult",
    "TemporalEventGrouper",
    "VectorIndex",
    "iter_observations",
    "load_observations",
]
