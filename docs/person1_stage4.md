# Person 1 Stage 4 Specification: YOLO11 Object Detection & Bounding-Box Validation

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Stage**: Stage 4 — YOLO11 Object Detection & Bounding-Box Validation  
**Active Branch**: `cv-pipeline`  
**Status**: COMPLETED & VERIFIED  

---

## 1. Overview & Objectives

Stage 4 executes object detection across all 7,287 sampled frames generated in Stage 3 from the 11 CityFlowV2 CCTV cameras. It employs the Ultralytics YOLO11 architecture (`yolo11n.pt`) to identify objects (vehicles, pedestrians, traffic infrastructure), validate and clip bounding boxes to native image boundaries, enforce schema compliance, and record all detections into a deterministic line-delimited JSON (`detections.jsonl`) accompanied by a summary manifest (`detection_manifest.json`).

### Key Principles & Invariants:
1. **Generic Object-Oriented Scope**: Detects relevant objects (vehicles, persons, traffic lights, etc.) using pretrained COCO classes without restricting detection to arbitrary subsets or introducing manual remapping.
2. **Native Resolution Bounding Boxes**: Bounding boxes are formatted strictly as `[x1, y1, x2, y2]` in the native image resolution coordinates of each camera (e.g. 1920×1080, 1280×960, 2560×1920).
3. **Strict Box Validation & Boundary Clipping**: Coordinates extending slightly beyond frame dimensions are clipped to `[0, width]` and `[0, height]`. Degenerate boxes ($x_1 \ge x_2$, $y_1 \ge y_2$, or width/height $< 1.0\text{ px}$) are rejected.
4. **Zero-Drift Timestamp Traceability**: Timestamps and frame paths are sourced directly from the authoritative Stage 3 `frame_index.jsonl`. No CityFlowV2 synchronization offsets are injected into observation timestamps.
5. **Deterministic Serialization**: Within each frame, detections are sorted deterministically by `(class_id, bbox[0], bbox[1], bbox[2], bbox[3], -confidence)` and assigned monotonically increasing IDs (`det_000001`, `det_000002`, ...).
6. **Strict Person 1 Scope Preservation**: Stage 4 produces raw object detections ONLY. No tracking IDs, persistent identity linking, object crops, embeddings, FAISS indices, or retrieval logic are implemented in Stage 4.

---

## 2. Model Selection & Execution Environment

### Detection Model
- **Architecture**: Ultralytics YOLO11 Nano (`yolo11n.pt`)
- **Version**: Ultralytics 8.4.143
- **Pretrained Weights**: Standard official PyTorch weights loaded directly from local cache or verified repository path.
- **Input Resolution**: Standard dynamic batching (640×640 internal network scale with automatic letterboxing and coordinate rescaling back to original native dimensions via `box.xyxy`).

### Hardware & Acceleration
- **Inference Device**: `cuda:0` (NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM)
- **PyTorch Version**: 2.5.1+cu121
- **Python Version**: 3.12.10

---

## 3. Configuration & CLI Parameters

Detection configuration is encapsulated in `DetectionConfig` ([backend/cv/detection.py](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/backend/cv/detection.py)):

| Parameter | Default | Range / Type | Description |
|---|---|---|---|
| `model_name` | `yolo11n.pt` | `str / Path` | Model weight file or name |
| `conf_threshold` | `0.25` | `float (0.0, 1.0]` | Minimum detection confidence threshold |
| `batch_size` | `32` | `int >= 1` | GPU inference batch size |
| `device` | `None` (auto `cuda:0` / `cpu`) | `str` | PyTorch device string |
| `clip_boxes` | `True` | `bool` | Whether to clip coordinates to frame bounds |
| `min_box_size` | `1.0` | `float >= 0.0` | Minimum width and height for valid detections |
| `frames_root` | `data/frames` | `str / Path` | Root directory containing sampled camera frames |
| `frame_index_path` | `data/frames/frame_index.jsonl` | `str / Path` | Path to Stage 3 authoritative frame index |
| `output_dir` | `data/detections` | `str / Path` | Directory for serialized detection artifacts |
| `output_jsonl` | `data/detections/detections.jsonl` | `str / Path` | Output line-delimited detections file |
| `manifest_file` | `data/detections/detection_manifest.json` | `str / Path` | Output detection summary manifest |

The detection pipeline can be executed via the CLI:
```bash
python -m backend.cv.detection --conf 0.25 --batch-size 32 --model yolo11n.pt
```

---

## 4. Bounding Box Policy & Validation Pipeline

### Coordinate Format
Bounding boxes are formatted as 4-element float arrays:
$$\text{bbox} = [x_1, y_1, x_2, y_2]$$
where $(x_1, y_1)$ represents the top-left coordinate, and $(x_2, y_2)$ represents the bottom-right coordinate in pixels.

### Clipping & Validation Function
Implemented in `validate_and_clip_bbox()`:
```python
def validate_and_clip_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
    clip: bool = True,
    min_box_size: float = 1.0,
) -> tuple[bool, list[float] | None, str | None]
```
1. **Dimension Check**: The list must contain exactly 4 finite numerical values ($x_1, y_1, x_2, y_2$). Non-finite values (`NaN`, `Inf`, `-Inf`) are immediately rejected.
2. **Boundary Clipping**: If `clip=True`, coordinates are clipped:
   $$x_1 = \max(0.0, \min(x_1, W)), \quad x_2 = \max(0.0, \min(x_2, W))$$
   $$y_1 = \max(0.0, \min(y_1, H)), \quad y_2 = \max(0.0, \min(y_2, H))$$
3. **Geometry Validation**:
   - $x_1 < x_2$ and $y_1 < y_2$
   - $(x_2 - x_1) \ge \text{min\_box\_size}$
   - $(y_2 - y_1) \ge \text{min\_box\_size}$
4. **Precision**: Coordinates are rounded to 2 decimal places to ensure compact, deterministic serialization without precision drift.

---

## 5. Output Schemas & Data Contracts

### 5.1 Detection Record Schema (`detections.jsonl`)
Every line in `data/detections/detections.jsonl` is an independent JSON object:

```json
{
  "camera_id": "cam_01",
  "frame_index": 0,
  "timestamp": 0.0,
  "class_id": 2,
  "class_name": "car",
  "confidence": 0.7413,
  "bbox": [1097.04, 256.91, 1175.53, 322.65],
  "frame_path": "data/frames/cam_01/frame_000000.jpg",
  "source_video": "data/videos/cam_01/vdo.avi",
  "detection_id": "det_000001"
}
```

| Field | Type | Description |
|---|---|---|
| `detection_id` | `string` | Globally unique deterministic identifier (`det_000001` ..) |
| `camera_id` | `string` | Canonical camera identifier (`cam_01` .. `cam_11`) |
| `frame_index` | `integer` | 0-indexed position in source video |
| `timestamp` | `float` | Elapsed playback seconds from source video ($t = \text{frame\_index} / \text{fps}$) |
| `class_id` | `integer` | COCO class integer ID |
| `class_name` | `string` | Human-readable class name (e.g., `car`, `person`, `traffic light`) |
| `confidence` | `float` | Detection confidence score $\in [0.25, 1.0]$ rounded to 4 decimals |
| `bbox` | `list[float]` | Native-resolution box coordinates `[x1, y1, x2, y2]` rounded to 2 decimals |
| `frame_path` | `string` | Repo-relative path to sampled frame JPEG |
| `source_video` | `string` | Repo-relative path to source video file |

### 5.2 Summary Manifest Schema (`detection_manifest.json`)
Saved at `data/detections/detection_manifest.json`:
- `schema_version`: `"1.0"`
- `model_name`: `"yolo11n.pt"`
- `ultralytics_version`: `"8.4"`
- `torch_version`: `"2.5.1+cu121"`
- `device`: `"cuda:0"`
- `gpu_name`: `"NVIDIA GeForce RTX 4060 Laptop GPU"`
- `conf_threshold`: `0.25`
- `batch_size`: `32`
- `total_input_frames`: `7287`
- `processed_frames`: `7287`
- `failed_frames`: `0`
- `total_detections`: `68906`
- `confidence_statistics`: `{"min": 0.25, "max": 0.9575, "mean": 0.5329}`
- `frames_per_camera`: Camera frame count dictionary
- `detections_per_camera`: Camera detection count dictionary
- `class_distribution`: Class frequency dictionary
- `performance`: Performance metrics dictionary
- `detections_file`: `"data/detections/detections.jsonl"`

---

## 6. Execution Results & Statistics

The full dataset detection run completed with **0 errors across all 7,287 frames**:

### Per-Camera Detection Breakdown
| Camera ID | Scenario | Resolution | Source FPS | Sampled Frames | Detections Count | Detections / Frame |
|---|---|---|---|---|---|---|
| `cam_01` | S01 | 1920×1080 | 10.0 | 587 | 4,535 | 7.73 |
| `cam_02` | S01 | 1920×1080 | 10.0 | 633 | 7,296 | 11.53 |
| `cam_03` | S01 | 1920×1080 | 10.0 | 599 | 6,422 | 10.72 |
| `cam_04` | S01 | 1920×1080 | 10.0 | 633 | 5,254 | 8.30 |
| `cam_05` | S01 | 1280×960 | 10.0 | 633 | 6,532 | 10.32 |
| `cam_06` | S03 | 1920×1080 | 10.0 | 643 | 4,908 | 7.63 |
| `cam_07` | S03 | 2560×1920 | 10.0 | 684 | 6,796 | 9.94 |
| `cam_08` | S03 | 2560×1920 | 10.0 | 727 | 6,355 | 8.74 |
| `cam_09` | S03 | 2560×1920 | 10.0 | 725 | 2,052 | 2.83 |
| `cam_10` | S03 | 1920×1080 | 10.0 | 700 | 6,332 | 9.05 |
| `cam_11` | S03 | 1920×1080 | 8.0 | 723 | 12,424 | 17.18 |
| **Total** | — | — | — | **7,287** | **68,906** | **9.46** |

### Class Distribution
| Object Class | Detections Count | Percentage |
|---|---|---|
| `car` | 52,811 | 76.64% |
| `traffic light` | 8,616 | 12.50% |
| `truck` | 3,853 | 5.59% |
| `person` | 2,261 | 3.28% |
| `fire hydrant` | 597 | 0.87% |
| `bus` | 432 | 0.63% |
| `motorcycle` | 158 | 0.23% |
| `bicycle` | 130 | 0.19% |
| `stop sign` | 35 | 0.05% |
| `suitcase` | 7 | 0.01% |
| `parking meter` | 2 | <0.01% |
| `skateboard` | 2 | <0.01% |
| `airplane` | 1 | <0.01% |
| `backpack` | 1 | <0.01% |
| **Total** | **68,906** | **100.00%** |

---

## 7. Performance Benchmarks

Inference was performed on GPU `cuda:0` with batch size 32:
- **Total Input Frames**: 7,287
- **Model Load Time**: 0.064 s
- **Total Inference Time**: 296.96 s (~4.95 minutes)
- **Throughput**: 24.54 frames per second
- **Average Time per Frame**: 40.75 ms
- **Failed Frames**: 0 (100% success rate)

---

## 8. Verification & Test Suite

The test suite thoroughly verifies all requirements of Stage 4 across 22 dedicated test cases in [tests/test_stage4_detection.py](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex/Hackathon/tests/test_stage4_detection.py):

1. **Configuration Tests**: Default parameter verification, custom parameter assignment, boundary and negative value rejection.
2. **Bounding Box Validation**: Valid box passing, clipping out-of-bounds coordinates, rejection of zero-area/inverted/degenerate boxes, rejection of `NaN`/`Inf` coordinates, length enforcement.
3. **Record Schema Tests**: Acceptance of compliant detection records, rejection of records missing mandatory schema keys.
4. **Real Inference & Determinism**: Loading `yolo11n.pt`, performing inference on a real frame (`data/frames/cam_01/frame_000000.jpg`), and verifying that repeated runs on the same frame yield identical detections and bounding boxes.
5. **Full Dataset Manifest Verification**: Confirming manifest schema, camera count (11/11), detection totals (68,906), and class distribution consistency.
6. **Full Dataset JSONL Verification**: Parsing all 68,906 lines, verifying coordinate boundaries, ensuring zero missing keys, cross-validating timestamps against `frame_index.jsonl`, and verifying absence of `track_id`.

**Overall Test Suite Status**:
- Stage 4 Tests: **22 / 22 PASSED**
- All Project Tests (Stages 1–4): **69 / 69 PASSED**

---

## 9. Downstream Handoff to Stage 5

Stage 4 outputs provide clean, verified object detections for **Stage 5: Single-Camera Object Tracking & Crop Generation**.

### Provided Artifacts:
1. `data/detections/detections.jsonl`: 68,906 validated object detections with native pixel coordinates.
2. `data/detections/detection_manifest.json`: Verified summary manifest.
3. `backend/cv/detection.py`: Reusable detection module, box clipping and validation utility functions.

### Upstream Guarantees:
- Every detection corresponds to an existing JPEG frame in `data/frames/`.
- Every timestamp matches the authoritative `frame_index.jsonl` exactly.
- All bounding box coordinates are strictly within $[0, \text{width}]$ and $[0, \text{height}]$.
- No degenerate or negative-area boxes exist.
- Person 1 boundary is strictly preserved: no persistent tracking IDs, no crops, no embeddings, and no vector search elements have been introduced.
