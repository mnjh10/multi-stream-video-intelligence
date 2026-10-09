"""
End-to-End Backend Integration Tests for ARGUS.
Tests:
1. RealRetrievalProvider contract and functionality.
2. FastAPI endpoints: /query, /memory, /evidence, /health, /openapi.json.
3. Semantic Memory resolution (alias -> camera_id filter).
4. EvidenceService extraction with real CityFlowV2 videos.
5. Architectural encapsulation (zero FAISS/OpenCLIP/Torch imports in API routes).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.retrieval import (
    RetrievalProvider,
    RealRetrievalProvider,
    MockRetrievalProvider,
    get_retrieval_provider,
)
from backend.services.evidence import EvidenceService


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def shared_provider():
    # Uses pre-persisted FAISS index in data/index/
    return get_retrieval_provider()


class TestBackendContract:
    def test_provider_subclasses_retrieval_provider(self, shared_provider):
        assert isinstance(shared_provider, RetrievalProvider)
        assert isinstance(shared_provider, RealRetrievalProvider)

    def test_search_output_contract_keys(self, shared_provider):
        results = shared_provider.search("car on road", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

        expected_keys = {
            "event_id",
            "camera_id",
            "timestamp",
            "best_timestamp",
            "timestamp_start",
            "timestamp_end",
            "score",
            "object_id",
            "object_type",
            "evidence",
        }
        for item in results:
            assert expected_keys.issubset(set(item.keys()))
            assert isinstance(item["camera_id"], str)
            assert item["camera_id"].startswith("cam_")
            assert isinstance(item["score"], float)
            assert "source_video" in item["evidence"]
            assert Path(item["evidence"]["source_video"]).exists()


class TestFastAPIQueryEndpoint:
    def test_query_success(self, client):
        response = client.post("/query", json={"query": "white car", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["query"] == "white car"
        assert len(data["results"]) <= 3
        assert len(data["results"]) > 0

        first = data["results"][0]
        assert first["camera_id"].startswith("cam_")
        assert first["score"] is not None

    def test_query_determinism(self, client):
        r1 = client.post("/query", json={"query": "vehicle on road", "top_k": 3})
        r2 = client.post("/query", json={"query": "vehicle on road", "top_k": 3})
        assert r1.status_code == 200
        assert r2.status_code == 200
        res1 = r1.json()["results"]
        res2 = r2.json()["results"]
        assert len(res1) == len(res2)
        for item1, item2 in zip(res1, res2):
            assert item1["event_id"] == item2["event_id"]
            assert item1["camera_id"] == item2["camera_id"]
            assert pytest.approx(item1["score"], abs=1e-4) == item2["score"]

    def test_query_empty_error_handling(self, client):
        response = client.post("/query", json={"query": ""})
        assert response.status_code in (400, 422)

    def test_query_whitespace_error_handling(self, client):
        response = client.post("/query", json={"query": "   "})
        assert response.status_code in (400, 422)


class TestSemanticMemoryIntegration:
    def test_store_and_resolve_semantic_memory(self, client):
        store_res = client.post(
            "/memory",
            json={
                "key": "toll booth north",
                "value": "Northern toll gate plaza",
                "camera_id": "cam_02",
                "region": "toll_area",
            },
        )
        assert store_res.status_code == 200
        data = store_res.json()
        assert data["key"] == "toll booth north"
        assert data["camera_id"] == "cam_02"

        get_res = client.get("/memory/toll booth north")
        assert get_res.status_code == 200
        assert get_res.json()["camera_id"] == "cam_02"

    def test_query_resolves_semantic_memory_to_camera_filter(self, client):
        client.post(
            "/memory",
            json={
                "key": "main entrance",
                "value": "Main gate entrance",
                "camera_id": "cam_01",
            },
        )

        response = client.post(
            "/query",
            json={"query": "car near main entrance", "top_k": 5},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) > 0
        for item in results:
            assert item["camera_id"] == "cam_01"


class TestEvidenceExtraction:
    def test_extract_evidence_real_video(self, client):
        response = client.post(
            "/evidence",
            json={
                "source_video": "data/videos/cam_01/vdo.avi",
                "timestamp": 0.5,
                "camera_id": "cam_01",
                "event_id": "evt_test_001",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["camera_id"] == "cam_01"
        assert data["timestamp"] == 0.5
        assert Path(data["frame_path"]).exists()
        assert Path(data["frame_path"]).stat().st_size > 10000

    def test_extract_evidence_missing_video_returns_404(self, client):
        response = client.post(
            "/evidence",
            json={
                "source_video": "data/videos/cam_nonexistent/vdo.avi",
                "timestamp": 1.0,
            },
        )
        assert response.status_code == 404


class TestArchitectureEncapsulation:
    def test_routes_do_not_import_faiss_openclip_torch(self):
        routes_dir = Path("backend/api")
        forbidden = ["faiss", "open_clip", "torch"]
        for py_file in routes_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8").lower()
            for token in forbidden:
                assert token not in content, f"Forbidden import '{token}' found in {py_file}"


class TestHealthAndSwaggerDocs:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/query" in paths
        assert "/memory" in paths
        assert "/memory/{key}" in paths
        assert "/evidence" in paths
        assert "/health" in paths
