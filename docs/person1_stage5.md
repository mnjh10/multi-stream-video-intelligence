# Person 1 Stage 5 Specification: Object Tracking & Stable Per-Camera IDs

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Stage**: Stage 5 — Multi-Object Tracking & Stable Per-Camera IDs  
**Active Branch**: `cv-pipeline`  
**Status**: COMPLETED & VERIFIED  

---

## 1. Overview & Objectives

Stage 5 consumes the frozen Stage 4 YOLO11 object detections (`data/detections/detections.jsonl`) and executes deterministic per-camera temporal association using ByteTrack. The primary objective is to link generic object detections into stable camera-local trajectories (`object_id`) across consecutive sampled frames without mixing cameras or performing cross-camera re-identification.

### Key Architectural Invariants:
1. **Camera-Local Identity**: Object identities are strictly local to individual cameras (`obj_<camera_id>_<numeric_id>`). There is no cross-camera association or identity linking.
2. **Offline Detection Consumption**: Stage 5 operates entirely offline on the existing 68,906 Stage 4 detections. No neural network inference or frame reprocessing is performed.
3. **Monotonic Frame Ordering**: For every tracked object, `frame_index` increases strictly across its trajectory.
4. **Referential Integrity**: Every tracked observation corresponds to an exact, validated Stage 4 detection record.
5. **Deterministic Serialization**: Object IDs begin at `000001` for each camera and are assigned deterministically based on trajectory onset timestamp. Output records are strictly sorted by `(frame_index, object_id)`.
6. **Strict Person 1 Scope Preservation**: Stage 5 produces tracked detection records only. No object crops, embeddings, FAISS indices, or retrieval logic are implemented.

---

## 2. Tracker Selection & Execution Environment

### Tracker Specification
- **Tracker Algorithm**: ByteTrack (`BYTETracker`)
- **Implementation**: Built-in `ultralytics.trackers.byte_tracker`
- **Ultralytics Version**: `8.4.143`
- **Association Framework**: Two-stage bipartite matching via Hungarian algorithm (`lap` linear sum assignment) with Kalman filter state estimation (`KalmanFilterXYAH`).
- **Input Adapter**: Lightweight `OfflineDetections` wrapper exposing native Stage 4 bounding boxes and confidences to the tracker interface without image overhead.

### Hardware & Environment
- **Platform**: Windows 11
- **Python**: `3.12.10`
- **PyTorch**: `2.5.1+cu121`
- **Execution Mode**: Offline CPU/GPU tensor arithmetic

---

## 3. Configuration & Parameters

Tracking parameters are defined in `TrackingConfig` ([backend/cv/tracking.py](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/backend/cv/tracking.py)):

| Parameter | Default | Value Range | Description |
|---|---|---|---|
| `track_high_thresh` | `0.25` | `(0.0, 1.0]` | First-stage association score threshold |
| `track_low_thresh` | `0.10` | `[0.0, high]` | Second-stage association threshold for low-score detections |
| `new_track_thresh` | `0.25` | `(0.0, 1.0]` | Minimum score required to initialize a new tracklet |
| `track_buffer` | `30` | `int >= 1` | Maximum frames to maintain lost tracklets across occlusions |
| `match_thresh` | `0.80` | `(0.0, 1.0]` | Cost ceiling for IoU association matching |
| `fuse_score` | `True` | `bool` | Whether to fuse detection confidence with spatial IoU |
| `input_detections` | `data/detections/detections.jsonl` | `str / Path` | Path to Stage 4 detections file |
| `output_dir` | `data/tracks` | `str / Path` | Directory for serialized tracking artifacts |
| `tracks_file` | `data/tracks/tracks.jsonl` | `str / Path` | Serialized line-delimited tracks JSONL |
| `manifest_file` | `data/tracks/track_manifest.json` | `str / Path` | Summary tracking manifest |

CLI invocation:
```bash
python -m backend.cv.tracking --detections data/detections/detections.jsonl --output-dir data/tracks
```

---

## 4. Association Mechanism & Candidate Confirmation

### Two-Stage Matching (ByteTrack)
1. **First-Stage Association**: High-confidence detections ($\text{conf} \ge 0.25$) are matched against active and lost track pools using Kalman filter predicted bounding box positions and spatial IoU distance.
2. **Second-Stage Association**: Remaining unmatched tracks are matched against lower-confidence detections ($\text{conf} \in [0.10, 0.25)$) to recover tracklets during motion blur or partial occlusion.
3. **Tracklet Confirmation**:
   - In accordance with standard ByteTrack design, a candidate tracklet initialized on frame $t$ is confirmed once it associates with a detection on a subsequent frame.
   - For offline tracking, once a tracklet is confirmed, its initial detection on frame $t$ is preserved and linked to the track.
   - Spurious, isolated single-frame detections that never associate with any subsequent frame within `track_buffer` are discarded as detector noise.

---

## 5. Camera-Local Object ID Format

Object IDs follow the strict camera-local convention:
```text
obj_<camera_id>_<numeric_id>
```

### Formatting Rules:
- Camera identifier prefix without underscores: `obj_cam01_`, `obj_cam02_`, ..., `obj_cam11_`.
- Monotonically increasing zero-padded 6-digit integer starting at `000001` per camera.
- Trajectories are sorted deterministically by trajectory onset `(first_frame_index, first_detection_id)` before ID assignment.
- Examples:
  - `obj_cam01_000001`
  - `obj_cam01_000034`
  - `obj_cam11_000084`

---

## 6. Output Schemas & Data Contracts

### 6.1 Track Record Schema (`data/tracks/tracks.jsonl`)
Each record is a self-contained JSON object:
```json
{
  "camera_id": "cam_01",
  "frame_index": 0,
  "timestamp": 0.0,
  "object_id": "obj_cam01_000001",
  "object_type": "car",
  "class_id": 2,
  "confidence": 0.7413,
  "bbox": [1097.04, 256.91, 1175.53, 322.65],
  "frame_path": "data/frames/cam_01/frame_000000.jpg",
  "source_video": "data/videos/cam_01/vdo.avi",
  "detection_id": "det_000001"
}
```

| Field | Type | Description |
|---|---|---|
| `camera_id` | `string` | Canonical camera identifier (`cam_01` .. `cam_11`) |
| `frame_index` | `integer` | 0-indexed position in source video |
| `timestamp` | `float` | Source playback seconds ($t = \text{frame\_index} / \text{source\_fps}$) |
| `object_id` | `string` | Camera-local unique object identifier (`obj_camXX_000001`) |
| `object_type` | `string` | Object classification label (e.g. `car`, `truck`, `traffic light`) |
| `class_id` | `integer` | COCO integer class ID |
| `confidence` | `float` | Detector confidence score |
| `bbox` | `list[float]` | Native coordinate box `[x1, y1, x2, y2]` |
| `frame_path` | `string` | Repo-relative path to sampled frame JPEG |
| `source_video` | `string` | Repo-relative path to source video file |
| `detection_id` | `string` | Upstream Stage 4 detection identifier |

### 6.2 Track Manifest Schema (`data/tracks/track_manifest.json`)
Contains complete dataset-level accounting, per-camera tracking counts, and execution metrics:
- `schema_version`: `"1.0"`
- `stage`: `"Stage 5 — Multi-Object Tracking & Stable Per-Camera IDs"`
- `tracker_name`: `"ByteTrack"`
- `input_detection_count`: `68906`
- `camera_count`: `11`
- `frame_count`: `7287`
- `tracked_detection_count`: `61039`
- `discarded_detection_count`: `7867`
- `unique_object_count`: `933`
- `objects_per_camera`: Per-camera unique trajectory counts
- `tracked_per_camera`: Per-camera tracked detection counts
- `discarded_per_camera`: Per-camera unconfirmed detection counts
- `frames_per_camera`: Authoritative frame counts per camera
- `class_distribution`: Frequency distribution of tracked object types
- `performance`: Runtime benchmarks

---

## 7. Execution Results & Statistics

### Per-Camera Tracking Breakdown
| Camera ID | Scenario | Frames | Input Detections | Tracked Detections | Tracking Rate | Discarded (Unconfirmed) | Unique Objects |
|---|---|---|---|---|---|---|---|
| `cam_01` | S01 | 587 | 4,535 | 3,956 | 87.2% | 579 | 93 |
| `cam_02` | S01 | 633 | 7,296 | 6,063 | 83.1% | 1,233 | 106 |
| `cam_03` | S01 | 599 | 6,422 | 5,324 | 82.9% | 1,098 | 143 |
| `cam_04` | S01 | 633 | 5,254 | 4,308 | 82.0% | 946 | 143 |
| `cam_05` | S01 | 633 | 6,532 | 5,267 | 80.6% | 1,265 | 119 |
| `cam_06` | S03 | 643 | 4,908 | 4,357 | 88.8% | 551 | 58 |
| `cam_07` | S03 | 684 | 6,796 | 6,104 | 89.8% | 692 | 71 |
| `cam_08` | S03 | 727 | 6,355 | 5,949 | 93.6% | 406 | 29 |
| `cam_09` | S03 | 725 | 2,052 | 1,810 | 88.2% | 242 | 31 |
| `cam_10` | S03 | 700 | 6,332 | 5,948 | 93.9% | 384 | 56 |
| `cam_11` | S03 | 723 | 12,424 | 11,953 | 96.2% | 471 | 84 |
| **Total** | — | **7,287** | **68,906** | **61,039** | **88.6%** | **7,867** | **933** |

### Tracked Class Distribution
| Object Class | Tracked Records | Percentage |
|---|---|---|
| `car` | 46,572 | 76.30% |
| `traffic light` | 8,424 | 13.80% |
| `truck` | 3,283 | 5.38% |
| `person` | 1,747 | 2.86% |
| `fire hydrant` | 589 | 0.97% |
| `bus` | 386 | 0.63% |
| `motorcycle` | 18 | 0.03% |
| `bicycle` | 17 | 0.03% |
| `suitcase` | 3 | <0.01% |
| **Total** | **61,039** | **100.00%** |

### Discarded Detections Accounting
- **Total Discarded**: 7,867 detections (11.42% of input detections).
- **Explanation**: In accordance with ByteTrack's 2-stage association and confirmation mechanism, detections that only occur in an isolated single frame and never match any subsequent detection within the 30-frame track buffer are treated as unconfirmed false positives and discarded to maintain clean, stable trajectories.

---

## 8. Performance Benchmarks

- **Total Execution Time**: 11.965 s
- **Total Frames Processed**: 7,287
- **Total Detections Evaluated**: 68,906
- **Processing Rate**: 609.0 frames per second
- **Average Latency**: 1.64 ms per frame

---

## 9. Verification & Automated Test Suite

A dedicated test suite was implemented in [tests/test_stage5_tracking.py](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/tests/test_stage5_tracking.py), covering:
1. `test_tracking_config_defaults`: Verifies default configuration parameters.
2. `test_tracking_config_custom_values`: Verifies custom configuration parameters.
3. `test_tracking_config_rejects_invalid_high_thresh`: Verifies out-of-range thresholds are rejected.
4. `test_tracking_config_rejects_invalid_buffer`: Verifies non-positive buffer values are rejected.
5. `test_format_object_id`: Verifies exact format `obj_<camera_id>_<numeric_id>`.
6. `test_object_id_pattern_valid`: Verifies regex against valid camera-local IDs.
7. `test_object_id_pattern_invalid`: Verifies regex rejection of non-conforming IDs.
8. `test_validate_track_record_valid`: Verifies valid track record schema.
9. `test_validate_track_record_missing_key`: Flags missing schema keys.
10. `test_validate_track_record_camera_mismatch`: Ensures camera ID and object ID camera prefix match.
11. `test_offline_detections_adapter`: Tests array conversion, coordinates, and slicing in adapter.
12. `test_tracking_determinism_on_synthetic_sequence`: Proves repeated runs produce byte-identical output.
13. `test_stage5_track_manifest_structure`: Asserts all fields, totals, and camera maps in `track_manifest.json`.
14. `test_stage5_tracks_jsonl_invariants`: Evaluates all 61,039 lines in `tracks.jsonl`:
    - Asserts 100% camera isolation (0 cross-camera tracks).
    - Asserts ID uniqueness per frame (at most 1 detection per object per frame).
    - Asserts strictly monotonic frame order per trajectory.
    - Asserts valid, non-degenerate bounding boxes.
    - Asserts timestamp agreement with Stage 3.
    - Confirms numbering begins at `000001` for each camera.

**Test Suite Results**:
- Stage 5 Tests: **24 / 24 PASSED**
- Full Regression Suite (Stages 1–5): **93 / 93 PASSED**

---

## 10. Honest Scientific Limitations

1. **No Ground-Truth Tracking Evaluation**: No MOTA, IDF1, or HOTA scores are claimed because ground-truth vehicle ID associations from AI City Challenge were not evaluated against these tracks.
2. **Camera-Local Scope**: Tracks end when vehicles exit a camera's field of view. There is zero cross-camera tracking or vehicle re-identification.
3. **Occlusion Buffer Limit**: Occlusions longer than 30 sampled frames (~10 seconds) will result in a new object ID being assigned when the vehicle reappears.
