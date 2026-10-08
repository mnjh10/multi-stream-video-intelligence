# ARGUS — Conversational Multi-Camera Video Intelligence

**Track:** HNX26EPS05 — Multi-Stream Video Intelligence with Conversational Query  
**Status:** Person 1 (CV Pipeline) × Person 2 (Semantic Retrieval Engine) **FULLY INTEGRATED** (168 / 168 tests passing)  
**Primary Branches:** `main` (integrated), `cv-pipeline` (CV & integration)

---

## 1. Overview

**ARGUS** is an explainable, open-vocabulary, temporally indexed multi-camera video intelligence system that allows operators to search recorded surveillance footage using natural-language queries.

Instead of manually reviewing hours of disjointed CCTV video streams, users query ARGUS conversationally:

> *"Find the white car near camera 3 turning right."*  
> *"Find the red truck in the north intersection."*  
> *"Where did this vehicle appear next?"*

ARGUS processes multi-camera video streams, tracks objects with camera-local identifiers, extracts safe object crops, indexes visual appearances using OpenCLIP embeddings into FAISS, and dynamically groups search candidates into temporal events with grounded visual evidence.

---

## 2. End-to-End System Pipeline

```text
[Recorded CityFlowV2 Multi-Camera Streams (cam_01 .. cam_11)]
                        │
                        ▼ (Person 1 CV Pipeline — Stages 0–8)
           [Sampling @ 3.0 FPS Native Heterogeneous]
                        │
                        ▼
             [YOLO11 Object Detection]
                        │ (Ultralytics YOLO11x, conf=0.25)
                        ▼
           [ByteTrack Local ID Tracking]
                        │ (933 camera-local trajectories)
                        ▼
       [Deterministic Safe JPEG Crops (61,039)]
                        │ (Quality 95, safe coordinate clamping)
                        ▼
     [Authoritative observations.jsonl (61,039)]
 ═══════════════════════╪══════════════════════════════════════════════════════
                        │  (Integration Interface Boundary)
                        ▼
         [retrieval/observation_loader.py]
                        │
                        ▼
         [retrieval/observation.py: Observation Dataclass]
                        │
                        ▼
     [retrieval/observation_encoder.py + OpenCLIP ViT-B-32]
                        │ (512-dim unit-normalized visual embeddings)
                        ▼
       [retrieval/vector_index.py: FAISS IndexFlatIP]
                        │
   Natural Language Query ──► [retrieval/query_parser.py]
                        │            │
                        │            ▼
                        │      Structured Filters + Metadata
                        ▼            │
         [retrieval/search.py + ranking.py]
                        │ (Semantic similarity + metadata score)
                        ▼
          [Candidate Pool Expansion (Phase 9 Fix)]
                        │
                        ▼
         [retrieval/temporal.py: TemporalEventGrouper]
                        │ (Gap-based temporal clustering per object/cam)
                        ▼
         [retrieval/result_builder.py: ResultBuilder]
                        │
                        ▼
       [retrieval/retrieval_result.py: RetrievalResult Contract]
                        │
 ═══════════════════════╪══════════════════════════════════════════════════════
                        │  (Person 3 Boundary)
                        ▼
         [FastAPI Endpoints + Visual Evidence + Memory]
```

---

## 3. Data Contracts & Schemas

### Contract 1: Observation Record (Person 1 → Person 2)
Stored in [`data/observations/observations.jsonl`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/data/observations/observations.jsonl) (61,039 records):

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

* **Timestamp Rule:** Source-video elapsed time in seconds (`frame_index / fps`). Synchronization offsets are strictly excluded.
* **Prohibited Fields:** No downstream fields (`detection_id`, embeddings, etc.) exist in public records.

---

### Contract 2: RetrievalResult (Person 2 → Person 3)
Produced by [`ObservationRetrievalPipeline`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/retrieval/observation_pipeline.py) for the backend:

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

* **`event_id`**: Structured event identifier (`evt_XXXXXX`).
* **`best_timestamp`**: Moment of peak visual match within the temporal window `[timestamp_start, timestamp_end]`.
* **`source_video`**: Video file path for visual evidence extraction and clip generation.

---

## 4. Quick Start & Usage

### Setup & Requirements
```bash
pip install torch torchvision open-clip-torch faiss-cpu ultralytics opencv-python pytest pillow
```

### Running Retrieval via Python
```python
from retrieval import (
    CLIPEmbedder,
    ObservationEncoder,
    ObservationRetrievalPipeline,
    RetrievalPipeline,
    RetrievalEngine,
    load_observations,
)

# 1. Initialize OpenCLIP ViT-B-32 model
embedder = CLIPEmbedder()
encoder = ObservationEncoder(embedder)

# 2. Build retrieval pipeline
pipeline = ObservationRetrievalPipeline(
    encoder=encoder,
    retrieval_pipeline=RetrievalPipeline(
        retrieval_engine=RetrievalEngine(embedder=embedder, vector_index=None)
    ),
    max_gap=2.0,
)

# 3. Index observations
observations = load_observations(limit=500)
pipeline.index_observations(observations)

# 4. Search with natural language
results = pipeline.search("white car turning right", top_k=5)
for r in results:
    print(r.to_dict())
```

---

## 5. Verification & Testing

Run the full unified test suite (168 tests):

```bash
python -m pytest tests/ -v
```

### Test Breakdown:
* **Person 1 Regression Suite (150 tests):**
  * Stage 1 Ingestion (14 tests)
  * Stage 2 Dataset Discovery (6 tests)
  * Stage 3 Frame Sampling (27 tests)
  * Stage 4 YOLO11 Detection (22 tests)
  * Stage 5 ByteTrack Tracking (24 tests)
  * Stage 6 Observations & Crops (27 tests)
  * Stage 7 Multi-Camera Audit (13 tests)
  * Stage 8 Handoff Freeze (17 tests)
* **Person 1 × Person 2 Integration Suite (18 tests):**
  * Contract exactness & referential integrity
  * OpenCLIP 512-dim unit-normalized visual embeddings
  * FAISS persistence determinism & mapping invariants
  * Query parser, structured filters, and hybrid ranking
  * Temporal event grouping & candidate pool expansion
  * Real end-to-end queries (unfiltered and camera-filtered)

**Result:** `168 passed in 83.5s (100% pass rate)`.

---

## 6. Directory Structure

```text
├── backend/                  # Person 1 CV Pipeline
│   └── cv/                   # Video ingestion, sampling, detection, tracking, crops
├── retrieval/                # Person 2 Semantic Retrieval Engine
│   ├── observation.py        # Observation dataclass
│   ├── observation_loader.py # Streaming observation loader
│   ├── observation_encoder.py# OpenCLIP crop image encoder
│   ├── embeddings.py         # OpenCLIP ViT-B-32 embedder
│   ├── vector_index.py       # FAISS IndexFlatIP wrapper
│   ├── index_builder.py      # Observation index builder
│   ├── index_storage.py      # FAISS binary + JSON metadata persistence
│   ├── query_parser.py       # Natural-language query parser
│   ├── ranking.py            # Hybrid semantic + metadata ranker
│   ├── search.py             # Retrieval engine coordinator
│   ├── temporal.py           # Temporal event grouper
│   ├── event_builder.py      # Event builder
│   ├── result_builder.py     # Result contract builder
│   ├── retrieval_result.py   # RetrievalResult dataclass
│   └── observation_pipeline.py # End-to-end ObservationRetrievalPipeline
├── evaluation/               # Person 2 Evaluation Framework
│   ├── evaluate.py           # Recall@k, MRR metrics
│   ├── baseline.py           # Full-frame retrieval baseline
│   └── ground_truth.py       # Ground truth schemas
├── docs/                     # Specifications & Handoff Documents
│   ├── person1_handoff.md    # Person 1 CV Handoff
│   ├── person1_stage8_handoff.md # Person 1 Stage 8 Freeze
│   └── person3_handoff.md    # Person 3 Backend & API Handoff
├── tests/                    # Unified Automated Test Suite
│   ├── test_p1_p2_integration.py # Person 1 × Person 2 integration tests (18 tests)
│   └── test_stage*.py        # Person 1 regression tests (150 tests)
└── data/                     # Authoritative Dataset Artifacts
    ├── observations/         # observations.jsonl (61,039 records)
    ├── crops/                # Object crops (61,039 JPEGs)
    ├── frames/               # Sampled frames (7,287 JPEGs) & frame_index.jsonl
    ├── tracks/               # tracks.jsonl (61,039 records)
    ├── detections/           # detections.jsonl (68,906 records)
    └── videos/               # 11 CityFlowV2 raw videos
```
