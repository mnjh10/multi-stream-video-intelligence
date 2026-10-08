# ARGUS — Person 1 Computer Vision Pipeline & Integration Handoff

## 1. Purpose

This document formally describes the completed Person 1 Computer Vision & Video Intelligence pipeline for ARGUS and defines the exact interface that the downstream integration/retrieval components must consume.

The Person 1 pipeline is **complete and frozen through Stage 8**.

Downstream developers must consume the generated artifacts through the contracts described below and must **not modify, reinterpret, or regenerate the frozen Person 1 outputs** unless a new stage is explicitly agreed upon by the team.

---

# 2. ARGUS System Position

The overall intended pipeline is:

```text
Recorded Multi-Camera Video
        ↓
Video Ingestion
        ↓
Frame Sampling
        ↓
YOLO11 Object Detection
        ↓
Object Tracking
        ↓
Object Crops
        ↓
Observation Records
        ↓
Visual Embeddings
        ↓
Temporal Events
        ↓
Retrieval / Ranking
        ↓
Camera + Timestamp
        ↓
Evidence
```

Person 1 owns the pipeline up to and including the **observation + crop handoff**.

The downstream integration/retrieval side begins from the frozen observation records and crops.

---

# 3. Person 1 Responsibility

Person 1 completed:

* video ingestion
* camera inventory
* dataset acquisition and validation
* frame sampling
* frame indexing
* timestamp generation
* YOLO11 object detection
* bounding-box validation
* ByteTrack object tracking
* per-camera object IDs
* object crop generation
* observation serialization
* multi-camera reconciliation
* deterministic output validation
* pipeline quality control
* performance measurement
* downstream handoff contract

Person 1 does **not** own:

* visual embeddings
* OpenCLIP feature extraction
* FAISS
* query parsing
* semantic retrieval
* ranking
* semantic memory
* frontend
* FastAPI
* PostgreSQL application schema
* WebSocket
* cross-camera re-identification
* final retrieval evaluation

---

# 4. Dataset

The frozen dataset is:

**AI City Challenge 2022 Track 1 — CityFlowV2**

Selected subset:

* 11 cameras
* 2 scenarios: S01 and S03
* all selected cameras from the official train split
* approximately 40.47 minutes aggregate video
* 23,798 source frames
* primarily 10 FPS
* cam_11 operates at 8 FPS

Selected camera mapping:

| ARGUS Camera | Original Camera | Scenario | FPS |
| ------------ | --------------- | -------- | --: |
| cam_01       | c001            | S01      |  10 |
| cam_02       | c002            | S01      |  10 |
| cam_03       | c003            | S01      |  10 |
| cam_04       | c004            | S01      |  10 |
| cam_05       | c005            | S01      |  10 |
| cam_06       | c010            | S03      |  10 |
| cam_07       | c011            | S03      |  10 |
| cam_08       | c012            | S03      |  10 |
| cam_09       | c013            | S03      |  10 |
| cam_10       | c014            | S03      |  10 |
| cam_11       | c015            | S03      |   8 |

Important:

`offset_seconds` from CityFlow is synchronization metadata.

It is **not added to observation timestamps**.

Observation timestamps represent elapsed time within the source video.

---

# 5. Frozen Pipeline Results

The completed pipeline produced:

```text
11 cameras
        ↓
7,287 sampled frames
        ↓
68,906 detections
        ↓
61,039 tracked observations
        ↓
933 per-camera trajectories
        ↓
61,039 object crops
        ↓
61,039 serialized observations
```

Final artifacts:

* 11 source videos
* 7,287 sampled frames
* 68,906 detections
* 61,039 tracked records
* 933 trajectories
* 61,039 observations
* 61,039 JPEG crops

No missing observations or crops were found during the final handoff audit.

---

# 6. Frame Sampling Contract

Default sampling:

```text
sample_fps = 3.0
JPEG quality = 95
```

Frames are saved under:

```text
data/frames/<camera_id>/
```

Example:

```text
data/frames/cam_01/frame_004271.jpg
```

Frame indices are recorded in:

```text
data/frames/frame_index.jsonl
```

The sampling process is deterministic.

Frame index calculation:

```text
frame_index_k = round(k × source_fps / sample_fps)
```

Sampling uses the actual FPS of each source video.

---

# 7. Detection Contract

Detector:

```text
Ultralytics YOLO11
Model: yolo11n.pt
```

Configuration:

```text
confidence threshold = 0.25
device = cuda:0
batch size = 32
```

Bounding boxes use:

```text
[x1, y1, x2, y2]
```

with:

* origin at top-left
* x increasing to the right
* y increasing downward

Bounding boxes were clipped and validated before downstream use.

The detection stage does not claim object-detection accuracy because no matching ground-truth evaluation was performed.

---

# 8. Tracking Contract

Tracker:

```text
ByteTrack
```

Frozen configuration:

```text
track_high_thresh = 0.25
track_low_thresh = 0.10
new_track_thresh = 0.25
track_buffer = 30 frames
match_thresh = 0.80
fuse_score = True
```

Tracking is **camera-local**.

Object IDs must never be interpreted as globally consistent identities across cameras.

Example:

```text
obj_cam01_000034
obj_cam02_000034
```

These are different camera-local identities.

No cross-camera re-identification has been performed.

---

# 9. Frozen Observation Schema

The authoritative downstream interface is:

```json
{
  "observation_id": "obs_000001",
  "camera_id": "cam_01",
  "timestamp": 142.37,
  "frame_index": 4271,
  "object_id": "obj_cam01_000034",
  "object_type": "car",
  "confidence": 0.94,
  "bbox": [120, 80, 530, 310],
  "frame_path": "data/frames/cam_01/frame_004271.jpg",
  "crop_path": "data/crops/cam_01/obs_000001.jpg",
  "source_video": "data/videos/cam_01/video.mp4"
}
```

The exact observation keys are:

```text
observation_id
camera_id
timestamp
frame_index
object_id
object_type
confidence
bbox
frame_path
crop_path
source_video
```

The schema is frozen.

Do not add downstream fields directly into this contract.

---

# 10. Observation Semantics

### observation_id

Globally unique deterministic identifier.

Current range:

```text
obs_000001 → obs_061039
```

### camera_id

Deterministic camera identifier:

```text
cam_01
...
cam_11
```

### timestamp

Elapsed time in the original source video, in seconds.

It is:

```text
frame_index / source_video_fps
```

It is **not**:

* wall-clock time
* CityFlow synchronization offset
* cross-camera synchronized time

### frame_index

Zero-based source-video frame position.

### object_id

Stable within a camera only.

Format:

```text
obj_camXX_NNNNNN
```

Do not treat it as a global identity.

### object_type

Generic detected object class.

Examples include:

```text
car
truck
bus
person
motorcycle
bicycle
traffic light
```

The pipeline is not restricted to vehicles.

### confidence

YOLO detection confidence retained with the observation.

### bbox

Validated object bounding box:

```text
[x1, y1, x2, y2]
```

### frame_path

Project-relative path to the source sampled frame.

### crop_path

Project-relative path to the corresponding object crop.

### source_video

Project-relative path to the source video.

---

# 11. Crop Contract

Crops are stored under:

```text
data/crops/<camera_id>/
```

Example:

```text
data/crops/cam_01/obs_000001.jpg
```

Properties:

* JPEG
* quality 95
* RGB
* 0 px padding
* native crop resolution
* deterministic association with observation ID

There are:

```text
61,039 observations
61,039 crops
```

Each observation must resolve to exactly one crop.

---

# 12. Important Prohibited Fields

The frozen Person 1 observation interface must NOT contain downstream-derived fields such as:

```text
detection_id
embedding
vector
clip_embedding
dino_embedding
faiss_id
similarity_score
query
ranking_score
database_id
reid_id
global_object_id
```

These belong to downstream processing if required.

---

# 13. Downstream Integration Instructions

The integration/retrieval developer should begin from:

```text
data/observations/observations.jsonl
```

and:

```text
data/crops/
```

The observations file should be streamed line-by-line rather than loaded unnecessarily into memory as one large structure.

For every observation:

1. Read the JSON record.
2. Resolve `crop_path` relative to the repository root.
3. Load the crop.
4. Generate the required visual embedding.
5. Store the embedding in the downstream vector system.
6. Preserve the mapping:

```text
vector_position / vector_id
        ↔
observation_id
```

The downstream system must always be able to recover:

```text
vector
→ observation_id
→ camera_id
→ timestamp
→ object_type
→ confidence
→ bbox
→ frame_path
→ crop_path
→ source_video
```

---

# 14. Recommended Embedding Handoff

The next downstream stage may use OpenCLIP or the team's selected embedding model.

The embedding pipeline must:

* use the crop image as input
* apply the model's required preprocessing
* generate a deterministic embedding
* preserve observation ID mapping
* avoid changing the original crop
* avoid modifying the frozen observation records

If FAISS is used:

```text
FAISS vector index
        +
vector-position → observation_id mapping
        ↓
observation metadata
```

The FAISS index itself is a downstream artifact.

It must not replace the observation records.

---

# 15. Retrieval / Ranking Interface

A user query may eventually produce:

```text
query
   ↓
query interpretation
   ↓
candidate retrieval
   ↓
vector similarity
   ↓
metadata filtering
   ↓
temporal / camera reasoning
   ↓
ranking
   ↓
evidence
```

Person 1 provides the metadata needed for this process:

```text
camera_id
timestamp
frame_index
object_type
confidence
bbox
frame_path
crop_path
source_video
observation_id
object_id
```

Person 1 does not define the final retrieval or ranking algorithm.

---

# 16. Cross-Camera Identity Warning

The integration layer must not infer:

```text
obj_cam01_000034
=
obj_cam07_000034
```

or any other cross-camera identity relationship.

Person 1 only establishes:

```text
stable identity within a single camera
```

Cross-camera re-identification, if later required, is a separate downstream problem.

---

# 17. Evidence Generation

The final system should be able to return evidence using the frozen observation metadata.

A retrieved result should be traceable to:

```text
observation_id
        ↓
camera_id
        ↓
timestamp
        ↓
frame_path / crop_path
```

This allows the final ARGUS response to identify:

* what was detected
* where it was detected
* when it was detected
* which image/frame supports the result

---

# 18. Validation Requirements Before Integration

Before declaring downstream integration successful, verify:

### Observation integrity

```text
61,039 observations
61,039 unique observation IDs
0 missing crops
0 duplicate observations
0 schema violations
```

### Camera integrity

```text
11 cameras
no cross-camera object-ID leakage
```

### Path integrity

All paths must remain project-relative.

Do not convert the interface into machine-specific absolute paths.

### Determinism

The frozen Person 1 artifacts must remain unchanged.

Do not regenerate them simply to integrate another component.

---

# 19. Frozen Person 1 Artifacts

Important artifacts:

```text
data/videos/
data/frames/
data/frames/frame_index.jsonl
data/detections/
data/tracks/
data/crops/
data/observations/observations.jsonl
data/observations/observation_manifest.json
data/observations/person1_handoff_manifest.json
```

Supporting documentation includes the Person 1 stage reports and contract documentation under:

```text
docs/
```

---

# 20. Final Person 1 Validation

Final validation status:

```text
Stage 1   — COMPLETE
Stage 2   — COMPLETE
Stage 3   — COMPLETE
Stage 4   — COMPLETE
Stage 5   — COMPLETE
Stage 6   — COMPLETE
Stage 7   — COMPLETE
Stage 8   — COMPLETE
```

Final automated test status:

```text
150 / 150 tests passed
```

Stage 8 handoff validation:

```text
0 missing observations
0 duplicate observations
0 missing crops
0 invalid crops
0 schema mismatches
0 prohibited fields
0 cross-camera identity leaks
0 absolute paths
```

No embedding or retrieval processing was performed as part of Person 1.

---

# 21. Integration Rule

**Person 1's pipeline is frozen.**

The integration developer should consume the outputs rather than modify the pipeline.

If an integration requirement cannot be satisfied using the frozen interface, the correct action is to raise the requirement for team-level discussion rather than silently changing the observation schema or regenerating the dataset.

---

# 22. One-Line Handoff

The downstream team receives:

> **61,039 deterministic, camera-local object observations with validated metadata and corresponding JPEG crops across 11 CityFlowV2 cameras, ready for embedding, vector indexing, retrieval, ranking, and final evidence generation.**

Person 1's responsibility ends at this interface.
