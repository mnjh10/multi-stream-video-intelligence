"""
Focused tests for ARGUS targeted improvements:
1. Temporal Follow-Up Queries (same camera, tracked object timeline)
2. Annotated Evidence Frames (OpenCV bounding boxes, raw frame fallback)
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.temporal import get_temporal_followup_engine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestTemporalFollowup:
    def test_baseline_query_and_followup_earlier(self, client):
        # 1. Baseline query: Find a car in camera 6
        r1 = client.post("/query", json={"query": "Find a car in camera 6", "top_k": 3})
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["status"] == "ok"
        assert len(data1["results"]) > 0
        top = data1["results"][0]
        assert top["camera_id"] == "cam_06"
        assert top["object_id"] is not None

        # 2. Follow-up query asking for 10 seconds earlier
        r2 = client.post(
            "/query",
            json={
                "query": "Where was this object 10 seconds earlier?",
                "context": {
                    "camera_id": top["camera_id"],
                    "object_id": top["object_id"],
                    "timestamp": top["best_timestamp"],
                },
            },
        )
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["status"] == "ok"
        assert len(data2["results"]) > 0
        followup_res = data2["results"][0]
        assert followup_res["camera_id"] == "cam_06"
        assert followup_res["object_id"] == top["object_id"]
        assert followup_res["is_followup"] is True
        assert followup_res["best_timestamp"] < top["best_timestamp"]
        assert followup_res["bbox"] is not None
        assert len(followup_res["bbox"]) == 4

    def test_followup_next_appearance(self, client):
        # Explicit context at an earlier timestamp (e.g. 190.0s)
        r = client.post(
            "/query",
            json={
                "query": "Show me this same car at its next appearance.",
                "context": {
                    "camera_id": "cam_06",
                    "object_id": "obj_cam06_000004",
                    "timestamp": 190.0,
                },
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["results"]) > 0
        res = data["results"][0]
        assert res["camera_id"] == "cam_06"
        assert res["object_id"] == "obj_cam06_000004"
        assert res["is_followup"] is True
        assert res["best_timestamp"] > 190.0

    def test_followup_without_context_returns_clear_message(self, client):
        # Reset server's last context to None for this test
        engine = get_temporal_followup_engine()
        engine.set_last_context(None)

        r = client.post("/query", json={"query": "Show me this car later."})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("error", "no_match")
        assert "context is unavailable" in data["message"].lower() or "ambiguous" in data["message"].lower()
        assert len(data["results"]) == 0


class TestAnnotatedEvidenceFrames:
    def test_annotated_evidence_generation(self, client):
        # Request evidence with camera, object, and bbox
        r = client.post(
            "/evidence",
            json={
                "source_video": "data/videos/cam_06/vdo.avi",
                "timestamp": 207.7,
                "camera_id": "cam_06",
                "object_id": "obj_cam06_000004",
                "event_id": "evt_test_anno",
                "annotate": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["annotated"] is True
        assert data["bbox"] is not None
        assert len(data["bbox"]) == 4
        assert Path(data["frame_path"]).exists()
        assert Path(data["raw_frame_path"]).exists()

        # Check metadata
        meta = data["metadata"]
        assert meta["width"] > 0
        assert meta["height"] > 0
        # Verify bounding box fits within frame
        x1, y1, x2, y2 = data["bbox"]
        assert 0 <= x1 <= meta["width"]
        assert 0 <= y1 <= meta["height"]
        assert 0 <= x2 <= meta["width"]
        assert 0 <= y2 <= meta["height"]

    def test_evidence_fallback_when_target_missing(self, client):
        # Request evidence where target object is nonexistent
        r = client.post(
            "/evidence",
            json={
                "source_video": "data/videos/cam_06/vdo.avi",
                "timestamp": 10.0,
                "camera_id": "cam_06",
                "object_id": "obj_nonexistent_999999",
                "annotate": True,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["annotated"] is False
        assert Path(data["frame_path"]).exists()
        assert data["message"] is not None
        assert "could not be resolved" in data["message"].lower() or "not found" in data["message"].lower()

    def test_evidence_bbox_matches_returned_observation(self, client):
        # Retrieve verified observation
        r_query = client.post("/query", json={"query": "Find a red car in camera 05", "top_k": 1})
        assert r_query.status_code == 200
        top = r_query.json()["results"][0]
        assert top["bbox"] is not None
        assert top["frame_index"] is not None

        # Request evidence with exact observation parameters
        r_ev = client.post(
            "/evidence",
            json={
                "source_video": top["evidence"]["source_video"],
                "timestamp": top["best_timestamp"],
                "camera_id": top["camera_id"],
                "object_id": top["object_id"],
                "frame_index": top["frame_index"],
                "bbox": top["bbox"],
                "annotate": True,
            },
        )
        assert r_ev.status_code == 200
        ev_data = r_ev.json()
        assert ev_data["status"] == "ok"
        assert ev_data["annotated"] is True
        assert ev_data["bbox"] == top["bbox"]
        meta = ev_data["metadata"]
        x1, y1, x2, y2 = ev_data["bbox"]
        assert 0 <= x1 <= meta["width"]
        assert 0 <= y1 <= meta["height"]
        assert 0 <= x2 <= meta["width"]
        assert 0 <= y2 <= meta["height"]
        assert Path(ev_data["frame_path"]).exists()
        assert Path(ev_data["raw_frame_path"]).exists()


class TestQueryConstrainedEventRetrieval:
    @pytest.mark.parametrize(
        "query_variant",
        [
            "Find a red car in camera 05",
            "Find a red car in camera 5",
            "Find a red car in cam_05",
            "Find a red car in camera five",
        ],
    )
    def test_camera_normalization_and_filtering(self, client, query_variant):
        r = client.post("/query", json={"query": query_variant, "top_k": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["results"]) > 0
        for item in data["results"]:
            assert item["camera_id"] == "cam_05"

    def test_object_type_exclusion(self, client):
        r_cars = client.post("/query", json={"query": "Find a car in camera 05", "top_k": 5})
        assert r_cars.status_code == 200
        data_cars = r_cars.json()
        assert data_cars["status"] == "ok"
        assert len(data_cars["results"]) > 0
        for item in data_cars["results"]:
            assert item["object_type"] == "car"

        r_person = client.post("/query", json={"query": "Find a person in camera 05", "top_k": 3})
        assert r_person.status_code == 200
        data_person = r_person.json()
        assert data_person["status"] == "ok"
        assert len(data_person["results"]) > 0
        for item in data_person["results"]:
            assert item["object_type"] == "person"

    def test_honest_attribute_verification_status(self, client):
        r = client.post("/query", json={"query": "Find a red car in camera 05", "top_k": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["results"]) > 0
        for item in data["results"]:
            assert item["verification_status"] == "attribute_verified"
            assert item["attribute_details"] is not None
            assert "red" in item["attribute_details"]
            assert item["attribute_details"]["red"]["verified"] is True
            assert item["crop_path"] is not None
            assert Path(item["crop_path"]).exists()

    def test_generic_query_returns_visual_similarity(self, client):
        r = client.post("/query", json={"query": "Find a car in camera 6", "top_k": 3})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert len(data["results"]) > 0
        for item in data["results"]:
            assert item["verification_status"] == "visual_similarity"
            assert item["attribute_details"] is None

    def test_unverified_attributes_return_no_match(self, client):
        r_orange = client.post("/query", json={"query": "Find an orange car in camera 05", "top_k": 3})
        assert r_orange.status_code == 200
        data_orange = r_orange.json()
        assert data_orange["status"] == "no_match"
        assert len(data_orange["results"]) == 0
        assert data_orange["message"] is not None
        assert "orange car" in data_orange["message"]

        r_bus = client.post("/query", json={"query": "Find a bus in camera 05", "top_k": 3})
        assert r_bus.status_code == 200
        data_bus = r_bus.json()
        assert data_bus["status"] == "no_match"
        assert len(data_bus["results"]) == 0
        assert data_bus["message"] is not None
        assert "bus" in data_bus["message"]

    def test_multi_turn_conversational_followup_preserves_object(self, client):
        # Turn 1: Initial retrieval
        r1 = client.post("/query", json={"query": "Find a car in camera 6", "top_k": 1})
        assert r1.status_code == 200
        top1 = r1.json()["results"][0]
        cam_id = top1["camera_id"]
        obj_id = top1["object_id"]
        t1 = top1["best_timestamp"]

        # Turn 2: Follow-up asking for earlier
        r2 = client.post(
            "/query",
            json={
                "query": "Where was this object 10 seconds earlier?",
                "context": {
                    "camera_id": cam_id,
                    "object_id": obj_id,
                    "timestamp": t1,
                },
            },
        )
        assert r2.status_code == 200
        top2 = r2.json()["results"][0]
        assert top2["camera_id"] == cam_id
        assert top2["object_id"] == obj_id
        assert top2["is_followup"] is True
        t2 = top2["best_timestamp"]
        assert t2 < t1

        # Turn 3: Follow-up asking for later
        r3 = client.post(
            "/query",
            json={
                "query": "Show me this car later.",
                "context": {
                    "camera_id": cam_id,
                    "object_id": obj_id,
                    "timestamp": t2,
                },
            },
        )
        assert r3.status_code == 200
        top3 = r3.json()["results"][0]
        assert top3["camera_id"] == cam_id
        assert top3["object_id"] == obj_id
        assert top3["is_followup"] is True
        t3 = top3["best_timestamp"]
        assert t3 > t2

