"""
End-to-End Integration Tests for Person 1 (CV Pipeline) × Person 2 (Retrieval Engine).

Validates the complete contract and data flow:
    Real Person 1 Observations (observations.jsonl)
    ↓
    Real Object Crops (data/crops/cam_XX/obs_XXXXXX.jpg)
    ↓
    OpenCLIP ViT-B-32 Visual Embeddings (512-dim, normalized)
    ↓
    FAISS Vector Index (IndexFlatIP with metadata)
    ↓
    Natural Language Query & Structured Parsing
    ↓
    Query-aware Ranking (Semantic + Metadata)
    ↓
    Temporal Event Grouping (Camera + Object + Time Window)
    ↓
    RetrievalResult Contract Output
"""

import json
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch
from PIL import Image

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


# Fixture for sharing CLIPEmbedder across tests to avoid repeated model loading
@pytest.fixture(scope="module")
def shared_embedder():
    return CLIPEmbedder()


@pytest.fixture(scope="module")
def real_observations_slice():
    """Load a representative slice of real Person 1 observations across multiple cameras."""
    obs_cam01 = load_observations(camera_id="cam_01", limit=15)
    obs_cam02 = load_observations(camera_id="cam_02", limit=15)
    obs_cam03 = load_observations(camera_id="cam_03", limit=10)
    combined = obs_cam01 + obs_cam02 + obs_cam03
    assert len(combined) >= 40
    return combined


class TestPerson1Person2Contract:
    """Tests contract alignment and data loading between Person 1 and Person 2."""

    def test_observation_dataclass_contract_exactness(self):
        """Observation dataclass must match Person 1's frozen 11-key schema."""
        expected_keys = {
            "observation_id",
            "camera_id",
            "timestamp",
            "frame_index",
            "object_id",
            "object_type",
            "confidence",
            "bbox",
            "frame_path",
            "crop_path",
            "source_video",
        }
        obs = Observation(
            observation_id="obs_000001",
            camera_id="cam_01",
            timestamp=0.0,
            frame_index=0,
            object_id="obj_cam01_000001",
            object_type="car",
            confidence=0.88,
            bbox=[100.0, 200.0, 300.0, 400.0],
            frame_path="data/frames/cam_01/frame_000000.jpg",
            crop_path="data/crops/cam_01/obs_000001.jpg",
            source_video="data/videos/cam_01/vdo.avi",
        )
        obs_dict = obs.to_dict()
        assert set(obs_dict.keys()) == expected_keys
        assert "detection_id" not in obs_dict

    def test_load_real_observations_from_dataset(self):
        """Real observations from observations.jsonl deserialize cleanly into Observation instances."""
        observations = load_observations(limit=25)
        assert len(observations) == 25

        for obs in observations:
            assert isinstance(obs, Observation)
            assert obs.observation_id.startswith("obs_")
            assert obs.camera_id.startswith("cam_")
            assert isinstance(obs.timestamp, float)
            assert isinstance(obs.frame_index, int)
            assert obs.object_id.startswith(f"obj_{obs.camera_id.replace('_', '')}_")
            assert isinstance(obs.object_type, str) and len(obs.object_type) > 0
            assert 0.0 <= obs.confidence <= 1.0
            assert len(obs.bbox) == 4
            assert Path(obs.crop_path).exists()
            assert Path(obs.frame_path).exists()
            assert Path(obs.source_video).exists()

    def test_iter_observations_streaming(self):
        """iter_observations streams observations lazily without memory explosion."""
        count = 0
        for obs in iter_observations():
            assert isinstance(obs, Observation)
            count += 1
            if count >= 30:
                break
        assert count == 30

    def test_crop_images_are_valid_rgb(self):
        """Referenced crop images exist and can be opened as valid RGB images."""
        observations = load_observations(limit=10)
        for obs in observations:
            crop_path = Path(obs.crop_path)
            assert crop_path.is_file()
            with Image.open(crop_path) as img:
                assert img.mode == "RGB"
                assert img.width > 0
                assert img.height > 0


class TestEmbeddingAndVectorIndex:
    """Tests OpenCLIP visual embedding and FAISS vector indexing."""

    def test_clip_embedding_dimension_and_normalization(self, shared_embedder):
        """Embeddings produced by CLIPEmbedder must be 512-dim and unit normalized."""
        text_emb = shared_embedder.encode_text(["white vehicle turning right"])
        assert text_emb.shape == (1, 512)
        norm = torch.linalg.norm(text_emb, dim=-1).item()
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_observation_encoder_with_real_crops(self, shared_embedder):
        """ObservationEncoder encodes real Person 1 crop and returns normalized tensor + metadata."""
        encoder = ObservationEncoder(embedder=shared_embedder)
        obs = load_observations(limit=1)[0]
        emb, meta = encoder.encode(obs)

        assert isinstance(emb, torch.Tensor)
        assert emb.shape == (1, 512)
        assert pytest.approx(torch.linalg.norm(emb).item(), abs=1e-4) == 1.0
        assert meta["observation_id"] == obs.observation_id
        assert meta["camera_id"] == obs.camera_id
        assert meta["object_id"] == obs.object_id

    def test_faiss_vector_index_add_and_search(self):
        """VectorIndex adds normalized vectors and performs cosine similarity search."""
        index = VectorIndex(dimension=512)
        vectors = np.random.randn(10, 512).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        metadata = [{"id": i, "camera_id": f"cam_{i % 3:02d}"} for i in range(10)]

        index.add(vectors, metadata)
        assert len(index) == 10

        query_vec = vectors[3:4]
        results = index.search(query_vec, top_k=3)
        assert len(results) == 3
        # Top result should be vector 3 with score close to 1.0
        assert results[0]["metadata"]["id"] == 3
        assert pytest.approx(results[0]["score"], abs=1e-4) == 1.0

    def test_faiss_vector_index_metadata_filtering(self):
        """VectorIndex filters candidates by exact metadata match."""
        index = VectorIndex(dimension=512)
        vectors = np.random.randn(6, 512).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        metadata = [
            {"camera_id": "cam_01", "object_type": "car"},
            {"camera_id": "cam_02", "object_type": "car"},
            {"camera_id": "cam_01", "object_type": "truck"},
            {"camera_id": "cam_02", "object_type": "truck"},
            {"camera_id": "cam_01", "object_type": "car"},
            {"camera_id": "cam_03", "object_type": "car"},
        ]
        index.add(vectors, metadata)

        query = vectors[0:1]
        filtered = index.search(query, top_k=10, filters={"camera_id": "cam_01", "object_type": "car"})
        assert len(filtered) == 2
        for r in filtered:
            assert r["metadata"]["camera_id"] == "cam_01"
            assert r["metadata"]["object_type"] == "car"

    def test_index_storage_persistence(self):
        """IndexStorage saves and loads FAISS index and metadata without distortion."""
        index = VectorIndex(dimension=512)
        vectors = np.random.randn(5, 512).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        metadata = [{"obs_id": f"obs_{i:06d}"} for i in range(5)]
        index.add(vectors, metadata)

        storage = IndexStorage()
        with tempfile.TemporaryDirectory() as tmp_dir:
            idx_file = Path(tmp_dir) / "test.index"
            meta_file = Path(tmp_dir) / "test_meta.json"

            storage.save(index, str(idx_file), str(meta_file))
            assert idx_file.exists()
            assert meta_file.exists()

            loaded_index = VectorIndex(dimension=512)
            storage.load(loaded_index, str(idx_file), str(meta_file))
            assert len(loaded_index) == 5
            assert loaded_index.metadata == metadata

    def test_faiss_persistence_mapping_determinism_real_data(self, shared_embedder, real_observations_slice):
        """Explicitly verify FAISS persistence preserves vector positions, metadata ordering, and query determinism."""
        encoder = ObservationEncoder(embedder=shared_embedder)
        builder = ObservationIndexBuilder(encoder=encoder, embedding_dimension=512)
        builder.add_observations(real_observations_slice[:20])

        orig_index = builder.vector_index
        assert orig_index.index.ntotal == len(orig_index.metadata) == 20
        assert orig_index.index.d == 512

        orig_mapping = [meta["observation_id"] for meta in orig_index.metadata]

        # Query original index
        q_emb = shared_embedder.encode_text(["car"]).numpy()
        orig_results = orig_index.search(q_emb, top_k=5)
        orig_result_ids = [r["metadata"]["observation_id"] for r in orig_results]
        orig_result_scores = [r["score"] for r in orig_results]

        storage = IndexStorage()
        with tempfile.TemporaryDirectory() as tmp_dir:
            idx_file = Path(tmp_dir) / "persistent.index"
            meta_file = Path(tmp_dir) / "persistent_meta.json"

            storage.save(orig_index, str(idx_file), str(meta_file))

            reloaded_index = VectorIndex(dimension=512)
            storage.load(reloaded_index, str(idx_file), str(meta_file))

            # Invariant 1 & 2: Dimension preserved
            assert reloaded_index.dimension == 512
            assert reloaded_index.index.d == 512

            # Invariant 3: Vector count preserved
            assert reloaded_index.index.ntotal == orig_index.index.ntotal == 20
            assert len(reloaded_index.metadata) == 20

            # Invariant 4 & 5: Vector position to metadata mapping and observation_id invariant preserved
            reloaded_mapping = [meta["observation_id"] for meta in reloaded_index.metadata]
            assert reloaded_mapping == orig_mapping
            for i in range(len(orig_mapping)):
                assert reloaded_index.metadata[i]["observation_id"] == orig_index.metadata[i]["observation_id"]
                assert reloaded_index.metadata[i]["camera_id"] == orig_index.metadata[i]["camera_id"]

            # Invariant 6: Query determinism preserved after persistence reload
            reloaded_results = reloaded_index.search(q_emb, top_k=5)
            reloaded_result_ids = [r["metadata"]["observation_id"] for r in reloaded_results]
            reloaded_result_scores = [r["score"] for r in reloaded_results]

            assert reloaded_result_ids == orig_result_ids
            for s_orig, s_reload in zip(orig_result_scores, reloaded_result_scores):
                assert pytest.approx(s_orig, abs=1e-6) == s_reload


class TestRetrievalLogic:
    """Tests QueryParser, ResultRanker, TemporalEventGrouper, and ResultBuilder."""

    def test_query_parser_extractions(self):
        """QueryParser extracts object_type, camera_id, and attributes."""
        parser = QueryParser()

        parsed = parser.parse("white car in camera 1")
        assert parsed["object_type"] == "car"
        assert parsed["camera_id"] == "cam_01"
        assert "white" in parsed["attributes"]

        parsed2 = parser.parse("red truck in camera 11")
        assert parsed2["object_type"] == "truck"
        assert parsed2["camera_id"] == "cam_11"
        assert "red" in parsed2["attributes"]

    def test_result_ranker_combines_semantic_and_metadata(self):
        """ResultRanker computes weighted combination of semantic and metadata scores."""
        ranker = ResultRanker(semantic_weight=0.8, metadata_weight=0.2)
        candidates = [
            {
                "score": 0.70,
                "metadata": {"camera_id": "cam_01", "object_type": "car"},
            },
            {
                "score": 0.75,
                "metadata": {"camera_id": "cam_02", "object_type": "truck"},
            },
        ]
        parsed_query = {"camera_id": "cam_01", "object_type": "car"}
        ranked = ranker.rank(candidates, parsed_query)

        # Candidate 0: semantic = 0.70, metadata matches 2/2 = 1.0 => final = 0.8*0.70 + 0.2*1.0 = 0.56 + 0.20 = 0.76
        # Candidate 1: semantic = 0.75, metadata matches 0/2 = 0.0 => final = 0.8*0.75 + 0.2*0.0 = 0.60
        assert ranked[0]["metadata"]["camera_id"] == "cam_01"
        assert pytest.approx(ranked[0]["final_score"], abs=1e-3) == 0.76
        assert pytest.approx(ranked[1]["final_score"], abs=1e-3) == 0.60

    def test_temporal_event_grouper_and_gap_splitting(self):
        """TemporalEventGrouper merges close observations and splits on gap > max_gap."""
        grouper = TemporalEventGrouper(max_gap=2.0)
        observations = [
            {"camera_id": "cam_01", "object_id": "obj_cam01_000001", "timestamp": 10.0, "score": 0.8},
            {"camera_id": "cam_01", "object_id": "obj_cam01_000001", "timestamp": 11.5, "score": 0.9},
            {"camera_id": "cam_01", "object_id": "obj_cam01_000001", "timestamp": 15.0, "score": 0.7},  # gap 3.5 > 2.0
            {"camera_id": "cam_02", "object_id": "obj_cam02_000002", "timestamp": 10.0, "score": 0.85},
        ]

        events = grouper.group(observations)
        # Should produce 3 events: 2 for obj_cam01_000001 (split by gap), 1 for obj_cam02_000002
        assert len(events) == 3

        cam01_events = [e for e in events if e["camera_id"] == "cam_01"]
        assert len(cam01_events) == 2

        # First event: timestamps 10.0 and 11.5
        ev1 = cam01_events[0]
        assert ev1["timestamp_start"] == 10.0
        assert ev1["timestamp_end"] == 11.5
        assert ev1["best_timestamp"] == 11.5
        assert ev1["best_score"] == 0.9

        # Second event: timestamp 15.0
        ev2 = cam01_events[1]
        assert ev2["timestamp_start"] == 15.0
        assert ev2["timestamp_end"] == 15.0
        assert ev2["best_score"] == 0.7

    def test_result_builder_converts_events_to_retrieval_results(self):
        """ResultBuilder produces valid RetrievalResult contract objects."""
        builder = ResultBuilder()
        events = [
            {
                "event_id": "evt_000001",
                "camera_id": "cam_01",
                "object_id": "obj_cam01_000001",
                "timestamp_start": 5.0,
                "timestamp_end": 7.0,
                "best_timestamp": 6.0,
                "best_score": 0.92,
                "observations": [
                    {
                        "score": 0.92,
                        "timestamp": 6.0,
                        "object_type": "car",
                        "source_video": "data/videos/cam_01/vdo.avi",
                    }
                ],
            }
        ]
        results = builder.build(events)
        assert len(results) == 1
        res = results[0]
        assert isinstance(res, RetrievalResult)
        assert res.event_id == "evt_000001"
        assert res.camera_id == "cam_01"
        assert res.object_id == "obj_cam01_000001"
        assert res.timestamp_start == 5.0
        assert res.timestamp_end == 7.0
        assert res.best_timestamp == 6.0
        assert res.score == 0.92
        assert res.object_type == "car"
        assert res.source_video == "data/videos/cam_01/vdo.avi"

        d = res.to_dict()
        assert d["event_id"] == "evt_000001"


class TestEndToEndPipeline:
    """End-to-End integration tests using real Person 1 observations and crops."""

    def test_phase9_candidate_pool_expansion_and_event_ordering(self, shared_embedder, real_observations_slice):
        """Candidate pool is expanded before temporal grouping, returning top_k ranked events."""
        encoder = ObservationEncoder(embedder=shared_embedder)
        engine = RetrievalEngine(embedder=shared_embedder, vector_index=None)
        pipeline = ObservationRetrievalPipeline(
            encoder=encoder,
            retrieval_pipeline=RetrievalPipeline(retrieval_engine=engine),
            max_gap=2.0,
        )
        pipeline.index_observations(real_observations_slice)
        assert len(pipeline) == len(real_observations_slice)

        results = pipeline.search("car on road", top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

        # Verify results are sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_end_to_end_search_with_camera_filter(self, shared_embedder, real_observations_slice):
        """End-to-end search parses camera filter and isolates results to requested camera."""
        encoder = ObservationEncoder(embedder=shared_embedder)
        engine = RetrievalEngine(embedder=shared_embedder, vector_index=None)
        pipeline = ObservationRetrievalPipeline(
            encoder=encoder,
            retrieval_pipeline=RetrievalPipeline(retrieval_engine=engine),
            max_gap=2.0,
        )
        pipeline.index_observations(real_observations_slice)

        # Query cam_02 specifically
        results = pipeline.search("vehicle in camera 2", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.camera_id == "cam_02"
            assert r.object_id.startswith("obj_cam02_")
            assert Path(r.source_video).exists()
            assert r.timestamp_start <= r.best_timestamp <= r.timestamp_end

    def test_end_to_end_search_determinism(self, shared_embedder, real_observations_slice):
        """Running the same query multiple times yields identical results."""
        encoder = ObservationEncoder(embedder=shared_embedder)
        engine = RetrievalEngine(embedder=shared_embedder, vector_index=None)
        pipeline = ObservationRetrievalPipeline(
            encoder=encoder,
            retrieval_pipeline=RetrievalPipeline(retrieval_engine=engine),
            max_gap=2.0,
        )
        pipeline.index_observations(real_observations_slice)

        res1 = pipeline.search("car", top_k=4)
        res2 = pipeline.search("car", top_k=4)

        assert len(res1) == len(res2)
        for r1, r2 in zip(res1, res2):
            assert r1.event_id == r2.event_id
            assert r1.camera_id == r2.camera_id
            assert r1.object_id == r2.object_id
            assert pytest.approx(r1.score, abs=1e-5) == r2.score


class TestEvaluationFramework:
    """Tests evaluation metrics from Person 2 evaluation framework."""

    def test_recall_and_reciprocal_rank(self):
        from evaluation.evaluate import evaluate_dataset, evaluate_query, recall_at_k, reciprocal_rank

        results = ["evt_000001", "evt_000002", "evt_000003"]
        assert recall_at_k(results, "evt_000001", k=1) == 1.0
        assert recall_at_k(results, "evt_000002", k=1) == 0.0
        assert recall_at_k(results, "evt_000002", k=2) == 1.0

        assert reciprocal_rank(results, "evt_000001") == 1.0
        assert reciprocal_rank(results, "evt_000002") == 0.5
        assert reciprocal_rank(results, "evt_000003") == 1.0 / 3.0
        assert reciprocal_rank(results, "evt_000099") == 0.0

        single_eval = evaluate_query(results, "evt_000002")
        assert single_eval["recall_at_1"] == 0.0
        assert single_eval["recall_at_5"] == 1.0
        assert single_eval["reciprocal_rank"] == 0.5

        dataset_eval = evaluate_dataset([
            {"results": results, "ground_truth": "evt_000001"},
            {"results": results, "ground_truth": "evt_000002"},
        ])
        assert dataset_eval["recall_at_1"] == 0.5
        assert dataset_eval["recall_at_5"] == 1.0
        assert pytest.approx(dataset_eval["mrr"], abs=1e-4) == 0.75
