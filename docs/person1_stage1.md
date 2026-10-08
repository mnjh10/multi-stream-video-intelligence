# Person 1 Stage 1 Specification: Video Ingestion & Camera Inventory

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Stage**: Stage 1 — Video Ingestion & Camera Inventory  
**Active Branch**: `cv-pipeline`  
**Status**: COMPLETE  

---

## 1. Input Video Convention

Video sources must be placed within `data/videos/`.

Two layout styles are natively supported and deterministically discovered:

### Layout A: Canonical Camera Subdirectories (Recommended)
```text
data/videos/
├── cam_01/
│   └── video.mp4
├── cam_02/
│   └── video.mp4
└── ...
```
- Each subdirectory represents an independent camera feed.
- Exactly one video file in a camera directory: processed normally as the camera feed.
- Multiple video files in a camera directory: flagged as an ambiguity error with all candidate files recorded. No file is silently discarded.

### Layout B: Direct Files in Video Root
```text
data/videos/
├── cam_01.mp4
├── cam_02.mp4
└── entrance_feed.mp4
```
- Direct video files residing in `data/videos/` are treated as independent camera feeds.

---

## 2. Supported Formats & File Handling

### Supported File Extensions
Case-insensitive matching for standard video container formats:
- `.mp4`
- `.avi`
- `.mkv`
- `.mov`
- `.wmv`
- `.flv`
- `.webm`
- `.m4v`

### Ingestion & Stream Decoding
- Primary decoding and stream inspection is handled via OpenCV (`cv2.VideoCapture`).
- Supplementary technical metadata (such as exact stream container format and video codec) is inspected via `ffprobe` when available.
- All file paths are strictly converted to repository-relative POSIX format (`data/videos/cam_01/video.mp4`), ensuring portability across Windows, Linux, and macOS. Machine-specific absolute paths are never persisted.

---

## 3. Camera ID Assignment Rule

Camera IDs strictly follow the canonical format:
`cam_01`, `cam_02`, `cam_03`, ...

### Assignment Logic
1. **Explicit Canonical Names**:
   If a folder name or file stem matches the pattern `^cam[_-]?(\d+)$` (case-insensitive, e.g. `cam_01`, `cam02`, `CAM_3`), the ID is normalized directly to `cam_{num:02d}` (e.g. `cam_01`, `cam_02`, `cam_03`).
2. **Arbitrary / Semantic Names**:
   For sources with names like `entrance`, `gate_north`, or `parking`, canonical IDs are assigned sequentially starting from `cam_01` based on deterministic sorted order.
3. **Collision Avoidance**:
   Any auto-generated IDs automatically skip IDs already claimed by explicit matches. Every camera is guaranteed a distinct, stable, collision-free identifier.

---

## 4. Deterministic Ordering Rule

To ensure consistent behavior across different operating systems and filesystems:
- File paths are collected and normalized to repository-relative POSIX paths.
- All sources are sorted using lowercase alphanumeric collation with a case-sensitive tie-breaker:
  `sort_key = (path.lower(), path)`
- Directory iteration order, creation timestamps, and filesystem inode ordering have zero impact on the resulting camera inventory or ID assignment.

---

## 5. Metadata Fields

For each discovered camera stream, the following metadata fields are recorded:

| Field | Type | Description |
|---|---|---|
| `camera_id` | string | Canonical camera ID (e.g. `"cam_01"`). |
| `source_video` | string | Repository-relative POSIX path to the video. |
| `fps` | float \| null | Frame rate (frames per second). |
| `frame_count` | int \| null | Total frame count in the source video. |
| `duration_seconds` | float \| null | Video playback duration in elapsed seconds. |
| `width` | int \| null | Frame width in pixels. |
| `height` | int \| null | Frame height in pixels. |
| `codec` | string \| null | Video codec FourCC or name (e.g. `"h264"`, `"avc1"`). |
| `container` | string \| null | Container format extension (e.g. `"mp4"`). |
| `file_size_bytes` | int \| null | Exact file size on disk in bytes. |
| `readable` | bool | `true` if OpenCV successfully decoded the initial frame. |
| `duration_derived` | bool | `true` if duration was calculated via `frame_count / fps`. |
| `validation_status` | string | `"valid"` or `"invalid"`. |
| `error_message` | string \| null | Error details if validation failed; otherwise `null`. |

---

## 6. Validation Rules

A camera video is classified as **`VALID`** if and only if all of the following conditions pass:
1. File exists and is a regular file.
2. File size is greater than 0 bytes (`file_size_bytes > 0`).
3. Video stream can be opened by OpenCV (`cap.isOpened() == True`).
4. Initial frame decodes successfully (`readable == True` with non-empty frame data).
5. Frame dimensions are positive integers (`width > 0`, `height > 0`).
6. Framerate is a positive finite number (`fps > 0`, not NaN, not Inf).
7. Frame count is a positive integer (`frame_count > 0`).
8. Duration is a positive finite number (`duration_seconds > 0`).

If any condition fails, the entry is classified as **`INVALID`** with a clear, descriptive `error_message`. The inventory process never crashes due to a corrupt or unreadable video file.

---

## 7. Manifest Format

The camera inventory manifest is serialized to:
`data/observations/camera_inventory.json`

To guarantee strict determinism across identical runs, the manifest omits dynamic execution timestamps.

### Schema Structure
```json
{
  "schema_version": "1.0",
  "video_directory": "data/videos",
  "total_cameras": 0,
  "valid_cameras": 0,
  "invalid_cameras": 0,
  "cameras": []
}
```

---

## 8. Current Empty-Input Behavior

Because **NO REAL INPUT VIDEOS ARE CURRENTLY PRESENT** in the repository:
- The discovery logic scans `data/videos/` and detects 0 video files.
- The inventory builder outputs a clean manifest with:
  - `"total_cameras": 0`
  - `"valid_cameras": 0`
  - `"invalid_cameras": 0`
  - `"cameras": []`
- No fake, synthetic, or mock video entries are fabricated.
- The pipeline completes cleanly with exit code 0 and logs:
  `"NO REAL INPUT VIDEOS DISCOVERED IN data/videos"`.

---

## 9. Known Limitations

1. **Absence of Real CCTV Footage**: Pipeline cannot benchmark playback decoding FPS or verify hardware NVDEC acceleration until real video streams are added.
2. **Variable Frame Rate (VFR)**: For VFR video containers, `frame_count / fps` provides an approximation; container-level timestamp inspection (via FFmpeg/PyAV) will be evaluated in Stage 2 if real videos use VFR.
3. **Multi-File Camera Segments**: Currently indexes the primary video file per camera folder; multi-part segmented clips (`part1.mp4`, `part2.mp4`) will require continuous sequence concatenation if introduced in future stages.
