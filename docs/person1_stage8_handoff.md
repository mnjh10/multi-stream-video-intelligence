# Person 1 Stage 8 Specification: Downstream Handoff & Interface Freeze

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Stage**: Stage 8 — Person 1 Downstream Handoff & Interface Freeze  
**Active Branch**: `cv-pipeline`  
**Status**: FROZEN & AUDITED  

---

> **Authoritative Handoff Statement**:  
> Person 1 provides camera-local object observations and image crops. Person 1 does not provide cross-camera identity, visual embeddings, semantic retrieval, vector indexing, or query ranking.

---

## 1. Stage 8 Purpose & Architectural Boundary

Stage 8 formally freezes and packages the completed Person 1 Computer Vision pipeline deliverables for downstream consumption by **Person 2** (Multimodal Retrieval, OpenCLIP embeddings, and FAISS indexing) and **Person 3** (Backend API and Query Coordination).

### Pipeline Ownership Division:

```text
┌────────────────────────────────────────────────────────┐
│               PERSON 1 RESPONSIBILITY                  │
│                                                        │
│   Raw CityFlowV2 Video Feeds (11 Cameras)              │
│                     ↓                                  │
│   Deterministic Frame Sampling (3.0 FPS)               │
│                     ↓                                  │
│   YOLO11 Object Detection                              │
│                     ↓                                  │
│   ByteTrack Camera-Local Tracking                      │
│                     ↓                                  │
│   Object Crop Extraction (JPEG Quality 95)             │
│                     ↓                                  │
│   Observation Serialization (JSONL)                    │
└──────────────────────────┬─────────────────────────────┘
                           │
             ═══ FROZEN HANDOFF INTERFACE ═══
                           │
┌──────────────────────────▼─────────────────────────────┐
│          PERSON 2 / DOWNSTREAM RESPONSIBILITY          │
│                                                        │
│   Visual Embedding Generation (OpenCLIP / DINO)        │
│                     ↓                                  │
│   FAISS Vector Index Construction                      │
│                     ↓                                  │
│   Multi-Camera Natural Language Semantic Retrieval     │
│                     ↓                                  │
│   Ranking, Confidence Fusion & Event Windowing         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Authoritative Deliverables & Directory Layout

Downstream consumers must read strictly from these authoritative paths:

### Primary Deliverables:
1. **Observation Records**: [`data/observations/observations.jsonl`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/data/observations/observations.jsonl)
   - 61,039 line-delimited JSON objects.
   - 21.30 MB on disk.
2. **Object Crops Directory**: [`data/crops/`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/data/crops/)
   - 61,039 JPEG image files organized into 11 camera subdirectories (`cam_01/` through `cam_11/`).
   - 651.75 MB total storage.
3. **Handoff Manifest**: [`data/observations/person1_handoff_manifest.json`](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/data/observations/person1_handoff_manifest.json)
   - Deterministic dataset-level summary and validation proof.

### Supporting Provenance Artifacts:
- **Sampled Frames**: `data/frames/` (7,287 source frames referenced by observations).
- **Source Videos**: `data/videos/` (11 raw videos for provenance auditing).
- **Frame Index**: `data/frames/frame_index.jsonl` (authoritative frame mapping).

---

## 3. Authoritative Observation Contract

Every line in `data/observations/observations.jsonl` contains **exactly** the 11 frozen Stage 0 contract keys in JSON format:

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

### Prohibited Fields:
Downstream systems must not expect, and Person 1 does not serialize:
- `detection_id`
- `embedding` / `vector` / `clip_embedding` / `dino_embedding`
- `faiss_id` / `similarity_score`
- `query` / `ranking_score`
- `database_id`
- `reid_id` / `global_object_id`

---

## 4. Field Semantics & Downstream Consumption Mapping

| Field | Type | Description | Downstream (Person 2) Purpose |
|---|---|---|---|
| `observation_id` | `string` | Unique deterministic identifier `obs_{index:06d}` | Primary key for FAISS vector-to-metadata lookup |
| `camera_id` | `string` | Canonical camera ID (`cam_01` .. `cam_11`) | Multi-camera filter, provenance, spatial constraint |
| `timestamp` | `float` | Source playback seconds ($t = \text{frame\_index} / \text{source\_fps}$) | Temporal sorting, windowing, trajectory duration |
| `frame_index` | `integer` | 0-indexed position within source video | Frame-level synchronization and provenance |
| `object_id` | `string` | Camera-local trajectory ID (`obj_camXX_NNNNNN`) | Local trajectory aggregation (within-camera only) |
| `object_type` | `string` | Object category (e.g. `car`, `truck`, `person`) | Lexical / metadata filtering prior to vector search |
| `confidence` | `float` | YOLO11 detector confidence score | Filtering low-confidence detections, result ranking |
| `bbox` | `list[float]` | Original pixel coordinates `[x1, y1, x2, y2]` | Spatial location and visual bounding box overlay |
| `frame_path` | `string` | Project-relative path to source frame JPEG | Contextual frame visualization / UI display |
| `crop_path` | `string` | Project-relative path to object crop JPEG | **Direct input image for OpenCLIP embedding model** |
| `source_video` | `string` | Project-relative path to source raw video | Video stream provenance |

---

## 5. Crop Specification & Visual Contract

1. **Format**: Standard JPEG (`.jpg`).
2. **Quality Factor**: 95 (preserves visual texture, fine vehicle details, and license plate regions for OpenCLIP).
3. **Color Space**: RGB (standard 3-channel 8-bit unsigned integer).
4. **Padding**: 0 pixels (native tightly-bounded box).
5. **Dimensions**: Derived directly from the bounding box with safe coordinate clamping ($\text{width} \ge 1$, $\text{height} \ge 1$).
6. **Downstream Resizing**: Crops retain native pixel dimensions. Person 2 must apply model-specific resizing (e.g., standard bicubic interpolation to $224 \times 224$ or $336 \times 336$ via OpenCLIP `preprocess`) during embedding ingestion.

---

## 6. Temporal Contract

- **Definition**: `timestamp` represents **elapsed playback time in seconds** relative to the start of the source video:
  $$\text{timestamp} = \frac{\text{frame\_index}}{\text{source\_video\_fps}}$$
- **Not Wall-Clock Time**: Timestamps do not represent absolute calendar time.
- **CityFlow Synchronization Offsets**: The AI City Challenge `offset_seconds` is synchronization metadata recorded in `data/dataset_manifest.json`. It is **NOT** added to the observation `timestamp`. Downstream multi-camera temporal alignment should apply dataset synchronization offsets explicitly if cross-camera timeline alignment is desired.

---

## 7. Multi-Camera & Identity Semantics

- **Strict Camera-Local Scope**: Object IDs follow the format `obj_camXX_NNNNNN` (e.g. `obj_cam01_000001`, `obj_cam02_000001`).
- **No Shared Identities**: The same numeric suffix across different cameras does **NOT** indicate the same vehicle.
- **Cross-Camera Re-ID Responsibility**: Any vehicle re-identification across camera feeds must be performed by downstream retrieval models (e.g. appearance embedding cosine similarity in Person 2).

---

## 8. Path & Portability Contract

- All paths are serialized using **project-relative POSIX forward slashes** (`data/...`).
- Zero machine-specific absolute drive letters (`C:`, `D:`) or temporary paths exist.
- Downstream code resolves files relative to the project root:
  ```python
  from pathlib import Path
  repo_root = Path(__file__).resolve().parents[2] # Or current working directory
  abs_crop_path = repo_root / observation["crop_path"]
  ```

---

## 9. Verified Dataset Totals

These exact figures are frozen and verified across the pipeline:

| Metric | Frozen Total |
|---|---|
| Participating Cameras | **11** (`cam_01` .. `cam_11`) |
| Scenarios | **2** (S01, S03) |
| Sampled Frames | **7,287** |
| Raw YOLO11 Detections | **68,906** |
| ByteTrack Tracked Records | **61,039** |
| Unique Camera-Local Trajectories | **933** |
| Output Observations | **61,039** |
| Verified Crops on Disk | **61,039** |
| Failed Crops / Observations | **0** |

### Per-Camera Verification Breakdown:
| Camera ID | Scenario | Frames | Tracks | Observations | Verified Crops | Unique Trajectories |
|---|---|---|---|---|---|---|
| `cam_01` | S01 | 587 | 3,956 | 3,956 | 3,956 | 93 |
| `cam_02` | S01 | 633 | 6,063 | 6,063 | 6,063 | 106 |
| `cam_03` | S01 | 599 | 5,324 | 5,324 | 5,324 | 143 |
| `cam_04` | S01 | 633 | 4,308 | 4,308 | 4,308 | 143 |
| `cam_05` | S01 | 633 | 5,267 | 5,267 | 5,267 | 119 |
| `cam_06` | S03 | 643 | 4,357 | 4,357 | 4,357 | 58 |
| `cam_07` | S03 | 684 | 6,104 | 6,104 | 6,104 | 71 |
| `cam_08` | S03 | 727 | 5,949 | 5,949 | 5,949 | 29 |
| `cam_09` | S03 | 725 | 1,810 | 1,810 | 1,810 | 31 |
| `cam_10` | S03 | 700 | 5,948 | 5,948 | 5,948 | 56 |
| `cam_11` | S03 | 723 | 11,953 | 11,953 | 11,953 | 84 |
| **Total** | — | **7,287** | **61,039** | **61,039** | **61,039** | **933** |

---

## 10. Prohibited Downstream Assumptions

1. **Do not assume `object_id` is global**: `obj_cam01_000001` and `obj_cam02_000001` are independent entities.
2. **Do not assume crops are uniform in size**: Crops reflect native detector boxes. Preprocessing must resize them appropriately.
3. **Do not assume observations are sorted by timestamp across cameras**: Observations are ordered deterministically by `(camera_id, frame_index, object_id)`. Temporal cross-camera queries must explicitly index or sort by `timestamp`.
4. **Do not modify the observation schema**: Downstream systems must store embeddings and search indices in their own storage artifacts (e.g., FAISS index files or vector store metadata databases).

---

## 11. Downstream Consumption Code Example (Person 2)

```python
"""Example of downstream consumption by Person 2 (OpenCLIP & FAISS)."""

import json
from pathlib import Path
from PIL import Image

def load_observations(observations_file: Path):
    """Load and validate observation stream."""
    with open(observations_file, "r", encoding="utf-8") as f:
        for line in f:
            obs = json.loads(line)
            obs_id = obs["observation_id"]
            crop_path = Path(obs["crop_path"])
            
            # Load crop for OpenCLIP embedding
            with Image.open(crop_path) as img:
                # person2_model.encode_image(preprocess(img))
                pass
            
            yield obs_id, obs
```

---

## 12. Honest Scientific & System Limitations

1. **Camera-Local Scope**: Cross-camera vehicle continuity is not determined by Person 1.
2. **Bounding-Box Sizing**: Bounding boxes range from small distant vehicles ($32\text{px}$) to large foreground vehicles ($600\text{px}$). Downstream embedding models should account for resolution variance.
3. **No Visual Embeddings Provided**: Person 1's role ends strictly at the crop and observation record level.

---

## 13. Final Verdict

> **STAGE 8 — SAFE TO HAND OFF**
