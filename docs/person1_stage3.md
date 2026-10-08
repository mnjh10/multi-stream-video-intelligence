# Person 1 Stage 3 Specification: Deterministic Frame Sampling & Timestamp Indexing

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Stage**: Stage 3 — Frame Sampling & Timestamp Indexing  
**Active Branch**: `cv-pipeline`  
**Status**: COMPLETED & VERIFIED  

---

## 1. Overview & Objectives

Stage 3 extracts video frames from the 11 acquired CityFlowV2 CCTV cameras at a configurable sampling rate (default `sample_fps = 3.0`), indexing each sampled frame with exact 0-based source frame positions and source-video elapsed timestamps.

Key tenets:
1. **Source Resolution Preservation**: Frames are extracted at their exact native resolutions (1280×960, 1920×1080, 2560×1920). No downscaling or resizing is performed during Stage 3.
2. **0-Indexed Frame Positions**: Filenames and metadata reflect the true source video frame index (e.g. `frame_000030.jpg` for source frame 30), never arbitrary consecutive sample numbers.
3. **Strict Source-Video Timestamps**: Timestamps represent elapsed playback seconds from the camera's timeline. CityFlowV2 cross-camera synchronization offsets are strictly excluded from observation timestamps.
4. **Per-Video Heterogeneous FPS**: Dynamic framerate handling ensures `cam_01`–`cam_10` (10.0 FPS) and `cam_11` (8.0 FPS) compute accurate timestamps without global assumptions.
5. **Full Determinism**: Identical runs on identical inputs produce byte-for-byte and metadata-identical outputs with zero volatile timestamps.

---

## 2. Sampling Target & Configuration

Sampling parameters are encapsulated in `SamplingConfig` ([backend/cv/sampling.py](file:///D:/yasmin%20programs/PROJECT_FOLDER/HackNex\Hackathon/backend/cv/sampling.py)):

| Parameter | Default | Range / Type | Description |
|---|---|---|---|
| `sample_fps` | `3.0` | `float > 0.0` | Target sampling rate in frames per second |
| `jpeg_quality` | `95` | `int [1, 100]` | JPEG image compression quality |
| `output_dir` | `data/frames` | `str / Path` | Directory root for sampled frames |
| `index_file` | `data/frames/frame_index.jsonl` | `str / Path` | Path to line-delimited frame index |
| `manifest_file` | `data/frames/sampling_manifest.json` | `str / Path` | Path to summary sampling manifest |

The sampling rate can be configured via CLI:
```bash
python -m backend.cv.sampling --sample-fps 3.0 --jpeg-quality 95 --output-dir data/frames
```

---

## 3. Sampling Algorithm & Mathematical Schedule

To prevent cumulative floating-point drift and guarantee strict reproducibility across diverse platforms:
Given:
- $F = \text{source\_video\_fps}$ (native capture rate, e.g. 10.0 or 8.0)
- $S = \text{sample\_fps}$ (target rate, e.g. 3.0)
- $N = \text{total\_source\_frames}$

### Schedule Derivation
1. If $S \ge F$: all available source frames $k \in [0, N-1]$ are sampled without duplicate indices.
2. If $S < F$:
   Desired sampling time instances:
   $$t_k = \frac{k}{S} \quad \text{for } k = 0, 1, 2, \dots$$
   Corresponding 0-indexed source frame position:
   $$\text{frame\_index}_k = \text{int}\left(\text{round}\left(k \cdot \frac{F}{S}\right)\right)$$
   Sampling terminates when $\text{frame\_index}_k \ge N$.
3. Deduplication and ordering:
   $$\text{indices} = \text{sorted}(\text{list}(\text{set}(\text{indices})))$$

### Schedule Examples
- **10.0 FPS Source @ 3.0 FPS Sample Rate** (`cam_01`–`cam_10`):
  $k \cdot (10/3) \implies [0, 3, 7, 10, 13, 17, 20, 23, 27, 30, \dots]$
  Yields exactly 3 sampled frames per second uniformly across playback.
- **8.0 FPS Source @ 3.0 FPS Sample Rate** (`cam_11`):
  $k \cdot (8/3) \implies [0, 3, 5, 8, 11, 13, 16, 19, 21, 24, \dots]$
  Yields exactly 3 sampled frames per second on the 8.0 FPS capture timeline.

---

## 4. Timestamp Semantics & Synchronization Offset Separation

### Frozen Stage 0 Formula
Per the frozen Stage 0 contract:
```text
timestamp = frame_index / source_video_fps
```
Computed with 4-decimal precision (`round(float(frame_index) / float(source_fps), 4)`).

### Internal Usage vs. Serialization Precision
- **Internal Storage**: `compute_timestamp()` returns a float rounded to 4 decimal places (`round(raw_ts, 4)`). This rounded float is stored directly in the in-memory frame record (`record["timestamp"]`), preventing floating-point representation noise (such as `0.30000000000000004` from IEEE 754 binary division of `3 / 10.0`).
- **Decoupled Sampling**: The sampling algorithm itself (`compute_sample_indices`) does not rely on floating-point timestamps or incremental time additions; it computes 0-based source frame indices directly via integer rounding $k \cdot (F / S)$, ensuring zero accumulated drift.
- **Serialization**: When written to `data/frames/frame_index.jsonl`, `json.dumps()` emits the clean, exact decimal values (`0.0`, `0.3`, `0.7`, `0.375`, etc.).

> **Contract Precision Note**:  
> Stage 3 timestamps are strictly source-video elapsed timestamps. CityFlowV2 synchronization offsets (`offset_seconds`) are **NOT** added to the observation timestamp.

### Separation of Concerns
- **Observation `timestamp`**: Elapsed playback duration from video start (e.g. `frame 30` @ 10 FPS = `3.0s`; `frame 3` @ 8 FPS = `0.375s`).
- **CityFlow `offset_seconds`**: Stored exclusively in dataset manifest metadata (`data/dataset_manifest.json`) for downstream cross-camera temporal alignment. It is never included in individual frame observation records.

---

## 5. Output Directory Structure & File Naming

Frames are stored using 0-indexed source frame numbers:
```
data/frames/
├── cam_01/
│   ├── frame_000000.jpg
│   ├── frame_000003.jpg
│   ├── frame_000007.jpg
│   └── ...
├── cam_02/
│   └── ...
...
├── cam_11/
│   ├── frame_000000.jpg
│   ├── frame_000003.jpg
│   ├── frame_000005.jpg
│   └── ...
├── frame_index.jsonl
└── sampling_manifest.json
```

### Filename Semantics
- Format: `frame_{frame_index:06d}.jpg`
- `frame_index` directly matches the 0-based frame number in the source AVI video container.
- Frames are never numbered sequentially (e.g. `1, 2, 3`) because doing so would obscure the source temporal anchor.

---

## 6. Manifest & Index Formats

### 1. `data/frames/frame_index.jsonl`
Line-delimited JSON format for rapid streaming and downstream indexing. Each line contains:
```json
{
  "camera_id": "cam_01",
  "frame_index": 30,
  "timestamp": 3.0,
  "frame_path": "data/frames/cam_01/frame_000030.jpg",
  "source_video": "data/videos/cam_01/vdo.avi"
}
```

### 2. `data/frames/sampling_manifest.json`
Deterministic summary manifest containing aggregate sampling statistics without volatile timestamps (`generated_at`):
```json
{
  "schema_version": "1.0",
  "sample_fps": 3.0,
  "jpeg_quality": 95,
  "output_directory": "data/frames",
  "frame_index_file": "data/frames/frame_index.jsonl",
  "total_cameras": 11,
  "total_source_frames": 23798,
  "total_sampled_frames": 7287,
  "total_frames_disk_bytes": 5767349712,
  "cameras": [ ... ]
}
```

---

## 7. Performance & Verification Metrics

Execution across all 11 cameras on the development workstation:

| Camera ID | Source Path | Native Resolution | Source FPS | Source Frames | Sampled Frames | Disk Footprint | Processing Time | Decode Speed |
|---|---|---|---|---|---|---|---|---|
| `cam_01` | `data/videos/cam_01/vdo.avi` | 1920×1080 | 10.0 | 1,955 | 587 | 348.3 MB | 16.15s | 121.0 fps |
| `cam_02` | `data/videos/cam_02/vdo.avi` | 1920×1080 | 10.0 | 2,110 | 633 | 355.8 MB | 18.55s | 113.8 fps |
| `cam_03` | `data/videos/cam_03/vdo.avi` | 1920×1080 | 10.0 | 1,996 | 599 | 292.2 MB | 17.48s | 114.2 fps |
| `cam_04` | `data/videos/cam_04/vdo.avi` | 1920×1080 | 10.0 | 2,110 | 633 | 272.1 MB | 18.01s | 117.1 fps |
| `cam_05` | `data/videos/cam_05/vdo.avi` | 1280×960  | 10.0 | 2,110 | 633 | 180.5 MB |  9.85s | 214.2 fps |
| `cam_06` | `data/videos/cam_06/vdo.avi` | 1920×1080 | 10.0 | 2,141 | 643 | 337.0 MB | 17.74s | 120.7 fps |
| `cam_07` | `data/videos/cam_07/vdo.avi` | 2560×1920 | 10.0 | 2,279 | 684 | 629.0 MB | 36.74s |  62.0 fps |
| `cam_08` | `data/videos/cam_08/vdo.avi` | 2560×1920 | 10.0 | 2,422 | 727 | 1,325.8 MB | 45.48s |  53.3 fps |
| `cam_09` | `data/videos/cam_09/vdo.avi` | 2560×1920 | 10.0 | 2,415 | 725 | 1,064.8 MB | 43.31s |  55.8 fps |
| `cam_10` | `data/videos/cam_10/vdo.avi` | 1920×1080 | 10.0 | 2,332 | 700 | 404.3 MB | 19.51s | 119.6 fps |
| `cam_11` | `data/videos/cam_11/vdo.avi` | 1920×1080 | 8.0  | 1,928 | 723 | 557.5 MB | 19.13s | 100.8 fps |
| **Total** | **11 cameras** | — | — | **23,798** | **7,287** | **~5.50 GB** | **263.57s** | **90.3 avg fps** |

---

## 8. Error Handling & Robustness

The sampling engine incorporates defensive error handling:
1. **Missing or Inaccessible Files**: Raises explicit `FileNotFoundError` with absolute path references.
2. **Video Open Failures**: Raises `RuntimeError` if OpenCV fails to open container or initialize codec.
3. **Invalid Parameter Checks**:
   - `sample_fps <= 0`: Raises `ValueError`.
   - `jpeg_quality not in [1, 100]`: Raises `ValueError`.
   - `source_fps <= 0`: Raises `ValueError`.
4. **Frame Grab & Decode Failures**: Verifies `cap.grab()` and `cap.retrieve()` return valid frames. Any premature end-of-stream or unreadable frame aborts with explicit camera ID, frame number, and failure message.
5. **Resolution Integrity**: Asserts that every decoded frame matches expected native resolution dimensions ($W \times H$).
6. **Disk Write Verification**: Confirms `cv2.imwrite` returns True and target file exists with non-zero byte size before registering the frame record.

---

## 9. Downstream Handoff Considerations for Stage 4 (YOLO11)

1. **Resolution Variation**:
   - Stage 4 detector must accept native frames of varying resolutions ($1280 \times 960$, $1920 \times 1080$, $2560 \times 1920$) and map YOLO letterbox coordinates back to source pixel coordinates $[x_1, y_1, x_2, y_2]$.
2. **Sequential Batching**:
   - `data/frames/frame_index.jsonl` enables deterministic, sequential feeding into YOLO11 without touching raw video decoders again.
3. **Object Crops**:
   - Stage 5 crop generation will crop directly from the high-fidelity native resolution frames stored in `data/frames/`.
