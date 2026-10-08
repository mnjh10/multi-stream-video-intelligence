# Person 1 Role, Scope & Observation Contract Specification

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Team ID**: HNX26EPS05  
**Active Branch**: `cv-pipeline`  
**Baseline Commit**: `7989699ea795cd3af33ecf3fd8e103abc3a7b479`  
**Status**: STAGE 0 LOCKED  

---

## 1. Frozen Scope & Responsibilities

### In-Scope (Person 1 Ownership)
Person 1 is strictly responsible for the end-to-end computer vision and video observation pipeline:
1. **Video Ingestion**: Reading multi-camera video streams/files reliably via OpenCV / FFmpeg.
2. **Camera Inventory**: Discovering and validating multi-camera sources with deterministic IDs (`cam_01`, `cam_02`, ...).
3. **Frame Sampling**: Sampling frames from source videos according to inspected video properties.
4. **Timestamp & Frame-Index Generation**: Deterministic mapping of elapsed source-video time (seconds) and exact source frame indices.
5. **YOLO11 Object Detection**: Running object detection to identify target entities (vehicles, persons, etc.).
6. **Object Tracking**: Associating detections across consecutive frames within each camera to maintain stable local track IDs.
7. **Bounding-Box Validation**: Clipping, format verification (`[x1, y1, x2, y2]`), minimum dimension checks, and boundary enforcement.
8. **Object Crop Generation**: Extracting localized visual crops for each detected object from the sampled frame.
9. **Observation Generation**: Generating structured observation records adhering strictly to the contract schema.
10. **Multi-Camera Processing**: Supporting multiple independent camera feeds without identity confusion across feeds.
11. **Observation Serialization**: Writing observation manifests and visual assets to disk in standardized layouts.
12. **Quality-Control Validation**: Asserting observations pass strict schema, bbox coordinate, and file reference checks.
13. **CV-Pipeline Benchmarking**: Measuring ingestion FPS, inference latency, tracking latency, and memory footprint.
14. **Robustness & Failure Handling**: Graceful recovery from corrupt frames, EOF, dropped frames, or missing files.
15. **Downstream Handoff Contract**: Producing visual assets (frames, crops) and metadata observations consumed by Person 2 (retrieval) and Person 3 (backend).

### Explicitly Excluded (Out of Scope for Person 1)
The following tasks belong to other team members or later integration stages and are strictly out of scope for Person 1:
- OpenCLIP / DINO or other visual/multimodal embedding generation (Person 2)
- FAISS index construction or vector database management (Person 2)
- Natural language query parsing, tokenization, or expansion (Person 2)
- Natural-language semantic or metadata retrieval (Person 2)
- Ranking, scoring, or fusion algorithms (Person 2)
- Temporal grouping, sliding windows, or event aggregation (Person 2)
- Persistent semantic memory or clarification memory (Person 2 & Person 3)
- FastAPI endpoints, database schemas (PostgreSQL / Alembic), API routers (Person 3)
- WebSockets, streaming servers, or frontend UI (Person 3 & Person 4)
- Cross-camera global re-identification (stretch goal / multi-camera fusion stage)
- Final hackathon retrieval evaluation benchmarks (Person 2)

---

## 2. Frozen Observation Contract

Every observation produced by Person 1 must conform strictly to the following JSON structure:

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

### Concept Definitions & Invariants
- `observation_id` *(string)*: Deterministic, globally unique observation identifier formatted sequentially (e.g. `obs_000001`). Unique across the complete multi-camera dataset.
- `camera_id` *(string)*: Deterministic camera identifier (e.g. `cam_01`, `cam_02`). Stable across runs. No cross-camera identity assumption.
- `timestamp` *(float)*: Elapsed playback / source-video time in seconds (`>= 0.0`), computed from source video metadata and frame index. **NOT** wall-clock time.
- `frame_index` *(integer)*: 0-indexed position of the frame in the original source video (`>= 0`).
- `object_id` *(string)*: Stable tracking identifier within a single camera stream, prefixed by camera ID (e.g. `obj_cam01_000034`). An object appearing across multiple cameras must **never** share an ID automatically.
- `object_type` *(string)*: Lowercase detector class name (e.g. `car`, `person`, `bus`, `truck`, `motorcycle`, `bicycle`).
- `confidence` *(float)*: Detector confidence score bounded within `[0.0, 1.0]`.
- `bbox` *(list of 4 integers)*: `[x1, y1, x2, y2]`.
  - Origin `(0, 0)` is at top-left.
  - `x` increases rightward, `y` increases downward.
  - Strict ordering: `x1 < x2` and `y1 < y2`.
  - Coordinates are pixel coordinates clipped within frame boundaries: `0 <= x1 < x2 <= frame_width` and `0 <= y1 < y2 <= frame_height`.
  - Verified to match Ultralytics `Boxes.xyxy` format representation.
- `frame_path` *(string)*: Relative POSIX path to the stored sampled frame image file (e.g. `data/frames/cam_01/frame_004271.jpg`).
- `crop_path` *(string)*: Relative POSIX path to the stored cropped object image file (e.g. `data/crops/cam_01/obs_000001.jpg`).
- `source_video` *(string)*: Relative POSIX path to the source video file (e.g. `data/videos/cam_01/video.mp4`).

### Contract Boundary Prohibitions
Do **NOT** add any extra fields (such as `color`, `action`, `make`, `model`, `location_semantics`, `embedding`, `event_id`, or `track_history`) unless an explicit upstream contract change is ratified.

---

## 3. Frozen Directory & Output Layout

The designated layout for persistent visual assets and serialized observations:

```text
data/
├── videos/
│   ├── cam_01/
│   │   └── video.mp4
│   └── cam_02/
│       └── video.mp4
├── frames/
│   ├── cam_01/
│   │   ├── frame_000000.jpg
│   │   └── ...
│   └── cam_02/
│       ├── frame_000000.jpg
│       └── ...
├── crops/
│   ├── cam_01/
│   │   ├── obs_000001.jpg
│   │   └── ...
│   └── cam_02/
│       ├── obs_000050.jpg
│       └── ...
└── observations/
    ├── cam_01_observations.json
    └── multi_camera_observations.json
```

---

## 4. Provisional Decisions (Pending Real Video Inspection)

The following pipeline parameters are kept provisional and will be finalized once real CCTV videos are inspected:

1. **Frame Sampling Strategy**:
   - Provisional baseline: Fixed time stride (e.g. 1 frame per second or 2 frames per second) or frame stride based on source FPS.
   - Requires inspection of video FPS, vehicle/pedestrian velocity, scene motion, and total frame budget.
2. **YOLO11 Model Checkpoint**:
   - Candidates: `yolo11n.pt` (high speed / low VRAM), `yolo11s.pt` (balanced), `yolo11m.pt` (higher accuracy).
   - Selection depends on input video resolution, target processing FPS, and RTX 4060 GPU VRAM constraints.
3. **Detector Confidence Threshold**:
   - Provisional candidate range: `0.25` to `0.45` to balance precision and recall.
4. **Inference Image Size (`imgsz`)**:
   - Provisional default: `640` or native resolution scaled to standard YOLO stride (multiple of 32).
5. **Tracker Engine & Configuration**:
   - Candidates available in Ultralytics: ByteTrack (`bytetrack.yaml`) or BoT-SORT (`botsort.yaml`). ByteTrack is provisional favorite due to high speed and `lap` package availability.
6. **Processing Batch Size**:
   - Provisional: 4 to 16 frames per batch depending on GPU VRAM occupancy during inference.
7. **Observation File Format**:
   - Serialized as structured JSON array (`observations.json`) or line-delimited JSON (`observations.jsonl`).

---

## 5. Unresolved Decisions (Require Real Video Inspection)

1. **Real Input Videos Availability**:
   - Status: Currently **NO REAL INPUT VIDEOS PRESENT** in the repository or parent workspace.
2. **Video Specifications**:
   - Resolution, frame rate, container format, video codec, and duration are unknown until real files are provided.
3. **Timestamp Synchronization**:
   - Whether multi-camera videos share synchronized start times or have relative time offsets.
4. **Quality-Control Rejection Thresholds**:
   - Minimum pixel dimensions for bounding boxes / crops before discarding as noise (e.g. `< 20x20` pixels).
5. **Downstream Throughput Target**:
   - Real-time (1x playback speed) vs offline batch extraction speed requirements.

---

## 6. Environment Verification Record

Inspection performed without modifications or package installations:

- **Operating System**: Windows (AMD64)
- **Python**: 3.12.10
- **PyTorch**: 2.5.1+cu121 (CUDA 12.1)
- **CUDA Device**: NVIDIA GeForce RTX 4060 Laptop GPU (1 Device available)
- **Ultralytics**: 8.4.143
  - Verified trackers: ByteTrack (`bytetrack.yaml`), BoT-SORT (`botsort.yaml`), DeepOCSORT, FastTrack, OCSORT, TrackTrack
  - Bounding box convention verified: `Boxes.xyxy` outputs `[x1, y1, x2, y2]`
- **OpenCV**: 5.0.0
- **FFmpeg**: 9.0.1 (gyan.dev build with full CUDA / NVDEC / NVENC acceleration)
- **NumPy**: 2.2.6
- **Pandas**: 2.3.2
- **Tracking Dependencies**: `lap` 0.5.13, `scipy` 1.16.2
