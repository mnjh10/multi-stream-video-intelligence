import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("ARGUS TARGETED IMPROVEMENTS — LIVE SMOKE TEST")
print("=" * 60)

# 1. Health check
res = requests.get(f"{BASE_URL}/health")
print(f"1. Health check: HTTP {res.status_code} -> {res.json()}")
assert res.status_code == 200

# 2. Baseline query
print("\n2. Baseline Query: 'Find a car in camera 6'")
res = requests.post(f"{BASE_URL}/query", json={"query": "Find a car in camera 6", "top_k": 3})
print(f"HTTP {res.status_code}")
data = res.json()
assert data["status"] == "ok"
assert len(data["results"]) > 0
top = data["results"][0]
print(f"Top result: event={top['event_id']}, camera={top['camera_id']}, object={top['object_id']}, timestamp={top['best_timestamp']}s, bbox={top.get('bbox')}")
assert top["camera_id"] == "cam_06"
cam_id = top["camera_id"]
obj_id = top["object_id"]
ts = top["best_timestamp"]

# 3. Conversational Follow-Up: 10 seconds earlier
print("\n3. Temporal Follow-Up: 'Where was this object 10 seconds earlier?'")
res = requests.post(
    f"{BASE_URL}/query",
    json={
        "query": "Where was this object 10 seconds earlier?",
        "context": {
            "camera_id": cam_id,
            "object_id": obj_id,
            "timestamp": ts,
        },
    },
)
print(f"HTTP {res.status_code}")
data = res.json()
assert data["status"] == "ok"
assert len(data["results"]) > 0
earlier_res = data["results"][0]
print(f"Earlier result: message='{data.get('message')}'")
print(f"event={earlier_res['event_id']}, camera={earlier_res['camera_id']}, object={earlier_res['object_id']}, timestamp={earlier_res['best_timestamp']}s, relation='{earlier_res.get('followup_relation')}', bbox={earlier_res.get('bbox')}")
assert earlier_res["camera_id"] == cam_id
assert earlier_res["object_id"] == obj_id
assert earlier_res["best_timestamp"] < ts

# 4. Conversational Follow-Up: Show me this car later
print("\n4. Temporal Follow-Up: 'Show me this car later.'")
res = requests.post(
    f"{BASE_URL}/query",
    json={
        "query": "Show me this car later.",
        "context": {
            "camera_id": earlier_res["camera_id"],
            "object_id": earlier_res["object_id"],
            "timestamp": earlier_res["best_timestamp"],
        },
    },
)
print(f"HTTP {res.status_code}")
data = res.json()
assert data["status"] == "ok"
assert len(data["results"]) > 0
later_res = data["results"][0]
print(f"Later result: message='{data.get('message')}'")
print(f"event={later_res['event_id']}, camera={later_res['camera_id']}, object={later_res['object_id']}, timestamp={later_res['best_timestamp']}s, relation='{later_res.get('followup_relation')}'")
assert later_res["camera_id"] == cam_id
assert later_res["object_id"] == obj_id
assert later_res["best_timestamp"] > earlier_res["best_timestamp"]

# 5. Annotated Evidence Generation
print("\n5. Annotated Evidence Extraction")
res = requests.post(
    f"{BASE_URL}/evidence",
    json={
        "source_video": f"data/videos/{cam_id}/vdo.avi",
        "timestamp": earlier_res["best_timestamp"],
        "camera_id": earlier_res["camera_id"],
        "event_id": earlier_res["event_id"],
        "object_id": earlier_res["object_id"],
        "bbox": earlier_res.get("bbox"),
        "annotate": True,
    },
)
print(f"HTTP {res.status_code}")
ev_data = res.json()
assert ev_data["status"] == "ok"
assert ev_data["annotated"] is True
print(f"Annotated Frame: {ev_data['frame_path']} (exists: {Path(ev_data['frame_path']).exists()}, size: {Path(ev_data['frame_path']).stat().st_size} bytes)")
print(f"Raw Frame: {ev_data['raw_frame_path']} (exists: {Path(ev_data['raw_frame_path']).exists()})")
print(f"BBox: {ev_data.get('bbox')}")
print(f"Resolution: {ev_data['metadata']['width']}x{ev_data['metadata']['height']}")

# 6. Unannotated Fallback
print("\n6. Fallback Unannotated Evidence when target missing")
res = requests.post(
    f"{BASE_URL}/evidence",
    json={
        "source_video": f"data/videos/{cam_id}/vdo.avi",
        "timestamp": 15.0,
        "camera_id": cam_id,
        "object_id": "obj_cam99_999999",
        "annotate": True,
    },
)
print(f"HTTP {res.status_code}")
ev_fallback = res.json()
assert ev_fallback["status"] == "ok"
assert ev_fallback["annotated"] is False
print(f"Fallback Frame: {ev_fallback['frame_path']} (annotated: {ev_fallback['annotated']})")

print("\n" + "=" * 60)
print("ALL LIVE SMOKE TESTS PASSED PERFECTLY!")
print("=" * 60)
