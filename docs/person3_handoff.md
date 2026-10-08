# ARGUS — Person 3 Integration Handoff Specification

**Project:** ARGUS — Conversational Multi-Camera Video Intelligence  
**Role:** Person 1 (CV Pipeline) & Person 2 (Semantic Retrieval Engine)  
**Target Audience:** Person 3 (Backend API, Conversational Agent, & Evidence Coordination)  
**Branch:** `cv-pipeline` (Merged to `main`)  
**Status:** FULLY INTEGRATED & VERIFIED (168 / 168 tests passing)

---

## 1. Executive Summary

This document formally specifies the integrated **Person 1 (Computer Vision Pipeline)** and **Person 2 (Semantic Retrieval Engine)** deliverables for **Person 3** (Backend API, Database, Persistent Semantic Memory, and Evidence Generation).

Person 3 does **not** need to manage raw video frames, YOLO detection, ByteTrack tracking, or low-level FAISS vector calculations directly. Instead, Person 3 consumes the unified high-level retrieval interface [`ObservationRetrievalPipeline`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/retrieval/observation_pipeline.py) or loads structured observation records from [`data/observations/observations.jsonl`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/data/observations/observations.jsonl).

---

## 2. Integrated System Architecture

```text
 ┌────────────────────────────────────────────────────────┐
 │           PERSON 1: CV OBSERVATION PIPELINE            │
 │                                                        │
 │   11 Recorded CityFlowV2 Videos (data/videos/)         │
 │                  ↓ (cv2 / 3.0 FPS native)              │
 │   7,287 Sampled Frames (data/frames/)                  │
 │                  ↓ (Ultralytics YOLO11x, conf=0.25)    │
 │   68,906 Bounding Boxes (COCO classes)                 │
 │                  ↓ (ByteTrack tracking)                │
 │   61,039 Track Records across 933 Local Trajectories   │
 │                  ↓ (Pillow safe boundary clamping)     │
 │   61,039 Object Crops (data/crops/cam_XX/*.jpg)        │
 │                  ↓                                     │
 │   61,039 Observations (data/observations.jsonl)        │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │        PERSON 2: SEMANTIC RETRIEVAL ENGINE             │
 │                                                        │
 │   Observation Ingestion (retrieval.observation_loader) │
 │                  ↓                                     │
 │   OpenCLIP ViT-B-32 Image Encoder (512-dim L2 unit)    │
 │                  ↓                                     │
 │   FAISS Vector Index (IndexFlatIP + metadata map)      │
 │                  ↓                                     │
 │   Natural Language Query ("white car in camera 1")     │
 │                  ↓ (QueryParser + Metadata Filters)    │
 │   Hybrid Ranking (0.8 Semantic + 0.2 Metadata)         │
 │                  ↓ (Candidate Pool Expansion)          │
 │   Temporal Event Grouping (max_gap=2.0s per object)    │
 │                  ↓ (ResultBuilder)                     │
 │   RetrievalResult Objects (Structured Timelines)       │
 └──────────────────────────┬─────────────────────────────┘
                            │
             ════ FROZEN HANDOFF BOUNDARY ════
                            │
 ┌──────────────────────────▼─────────────────────────────┐
 │        PERSON 3: BACKEND API & CONVERSATIONAL UI       │
 │                                                        │
 │   FastAPI Endpoints (/query, /evidence, /cameras)      │
 │   PostgreSQL + Persistent Semantic Memory              │
 │     ("main gate" ↔ "cam_01" permanent alias mapping)   │
 │   Visual Evidence Serving (Frame JPEGs & Video Clips)  │
 │   Conversational LLM / LangChain / WebSocket Delivery  │
 └────────────────────────────────────────────────────────┘
```

---

## 3. Authoritative Data & Artifact Locations

All required artifacts reside at fixed, project-relative paths:

| Artifact | Path | Format | Record Count | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Observation Records** | `data/observations/observations.jsonl` | JSONL | 61,039 | Authoritative CV observations |
| **Object Crops** | `data/crops/cam_01/` .. `cam_11/` | JPEG | 61,039 | Individual cropped object images |
| **Sampled Frames** | `data/frames/cam_01/` .. `cam_11/` | JPEG | 7,287 | Full sampled source frames |
| **Source Videos** | `data/videos/cam_01/` .. `cam_11/` | AVI | 11 | Original recorded video streams |
| **Frame Index** | `data/frames/frame_index.jsonl` | JSONL | 7,287 | Mapping of frames to timestamps & indices |
| **Handoff Manifest** | `data/observations/person1_handoff_manifest.json` | JSON | 1 | Complete Stage 8 provenance manifest |

---

## 4. Contract Specifications for Person 3

### Contract 1: The Observation Contract (Input to Retrieval Engine)
Every observation in `data/observations/observations.jsonl` contains exactly these 11 keys:

```json
{
  "observation_id": "obs_000001",
  "camera_id": "cam_01",
  "timestamp": 0.0,
  "frame_index": 0,
  "object_id": "obj_cam01_000001",
  "object_type": "car",
  "confidence": 0.7413,
  "bbox": [1097.04, 256.91, 1175.53, 322.65],
  "frame_path": "data/frames/cam_01/frame_000000.jpg",
  "crop_path": "data/crops/cam_01/obs_000001.jpg",
  "source_video": "data/videos/cam_01/vdo.avi"
}
```

* **Timestamp Semantics:** Source-video elapsed time in seconds (`frame_index / fps`). Synchronization offsets (`offset_seconds`) are excluded.
* **Camera IDs:** `cam_01` through `cam_11`.
* **Object IDs:** `obj_<cam>_<seq>`, strictly camera-local. No cross-camera re-ID is claimed.

---

### Contract 2: The RetrievalResult Contract (Output to Person 3)
When Person 3 queries the retrieval engine, it receives a list of [`RetrievalResult`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/retrieval/retrieval_result.py#L4-L33) objects:

```json
{
  "event_id": "evt_000001",
  "camera_id": "cam_02",
  "object_id": "obj_cam02_000008",
  "timestamp_start": 0.0,
  "timestamp_end": 0.3,
  "best_timestamp": 0.3,
  "score": 0.39638,
  "object_type": "car",
  "source_video": "data/videos/cam_02/vdo.avi"
}
```

#### Field Specifications:
* `event_id`: Unique temporal event identifier (`evt_XXXXXX`).
* `camera_id`: Camera where the event occurred (`cam_01` to `cam_11`).
* `object_id`: Local track identifier of the target vehicle.
* `timestamp_start`: Beginning of the event in source-video seconds.
* `timestamp_end`: End of the event in source-video seconds.
* `best_timestamp`: Moment of maximum visual similarity to the user query.
* `score`: Retrieval confidence score [0.0, 1.0] combining visual similarity and metadata matching.
* `object_type`: Detected category (`car`, `truck`, `bus`).
* `source_video`: Path to the video file for evidence clipping.

---

## 5. How Person 3 Consumes the Retrieval Engine

### Pattern A: Initializing and Searching via Python API

```python
from retrieval import (
    CLIPEmbedder,
    ObservationEncoder,
    ObservationRetrievalPipeline,
    RetrievalPipeline,
    RetrievalEngine,
    load_observations,
)

# 1. Initialize embedder and encoder (cached OpenCLIP ViT-B-32)
embedder = CLIPEmbedder()
encoder = ObservationEncoder(embedder)

# 2. Build retrieval pipeline
pipeline = ObservationRetrievalPipeline(
    encoder=encoder,
    retrieval_pipeline=RetrievalPipeline(
        retrieval_engine=RetrievalEngine(embedder=embedder, vector_index=None)
    ),
    max_gap=2.0,  # Temporal clustering gap threshold in seconds
)

# 3. Index Person 1 observations
# (Can index a representative sample or the entire dataset)
observations = load_observations(limit=1000)
pipeline.index_observations(observations)

# 4. Search using natural language queries
results = pipeline.search("white car turning right", top_k=5)

for r in results:
    print(r.to_dict())
```

---

### Pattern B: Supporting Persistent Semantic Memory (Alias Resolution)

Person 3's core responsibility is **"Clarify once, remember permanently"**:
* User: *"Did a truck enter through the main gate?"*
* System maps `"main gate"` $\rightarrow$ `cam_01` (from PostgreSQL database).
* Person 3 formats query as: `"truck in camera 1"`.
* The retrieval engine automatically extracts:
  * `object_type = "truck"`
  * `camera_id = "cam_01"`
* FAISS filters candidates to `cam_01` trucks only, and ranks by visual similarity.

```python
# In FastAPI endpoint:
alias_mapping = {"main gate": "cam_01", "north intersection": "cam_03"}
user_query = "white car near main gate"

# Substitute stored persistent memory alias:
resolved_query = user_query.replace("main gate", "camera 1")

# Execute search
results = pipeline.search(resolved_query, top_k=5)
```

---

### Pattern C: Generating Visual Evidence for the Frontend

Person 3 can locate and serve visual evidence using the fields in `RetrievalResult`:

1. **Representative Frame / Evidence Image:**
   * Find observation where `timestamp == result.best_timestamp` and `object_id == result.object_id`.
   * Return `crop_path` (e.g. `data/crops/cam_01/obs_000001.jpg`) or `frame_path` (e.g. `data/frames/cam_01/frame_000000.jpg`).

2. **Video Clip Snippet:**
   * Extract video snippet from `source_video` using `timestamp_start` and `timestamp_end` with `ffmpeg` or `opencv`:
   ```bash
   ffmpeg -ss {timestamp_start} -to {timestamp_end} -i {source_video} -c copy clip.mp4
   ```

---

## 6. Pre-Packaged Index Storage for Fast Backend Startup

Person 3 should **not** re-encode 61,039 crops on every server restart. Use `IndexStorage` to save and load the pre-computed FAISS index:

```python
from retrieval import IndexStorage

# Save index once during build:
pipeline.index_builder.save(
    index_path="data/index/argus.index",
    metadata_path="data/index/argus_metadata.json"
)

# Load index instantly in FastAPI startup event:
pipeline.index_builder.load(
    index_path="data/index/argus.index",
    metadata_path="data/index/argus_metadata.json"
)
```

---

## 7. Verification & Test Suite

Person 3 can verify the complete integration by running:
```bash
python -m pytest tests/test_p1_p2_integration.py -v
```
All 18 tests will execute in ~30 seconds, verifying:
* Contract exactness
* OpenCLIP 512-dim unit normalization
* FAISS persistence determinism
* Temporal event grouping
* Candidate pool expansion
* End-to-end natural-language and camera-filtered queries
