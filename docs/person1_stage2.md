# Person 1 Stage 2 Specification: CityFlowV2 Acquisition & Dataset Lock

**Project**: Multi-Stream Video Intelligence (ARGUS)  
**Role**: Person 1 (CV & Video Observation Pipeline)  
**Stage**: Stage 2 — CityFlowV2 Acquisition, Inspection & Dataset Lock  
**Active Branch**: `cv-pipeline`  
**Status**: ACQUIRED & LOCKED  

---

## A. Dataset Identity & Source

- **Dataset Name**: AI City Challenge 2022 Track 1 — CityFlowV2
- **Official Documentation**: [https://www.aicitychallenge.org/2022-data-and-evaluation/](https://www.aicitychallenge.org/2022-data-and-evaluation/)
- **Official Download Portal**: [https://www.aicitychallenge.org/2022-track1-download/](https://www.aicitychallenge.org/2022-track1-download/)
- **Official Archive File**: `AICity22_Track1_MTMC_Tracking.zip` (16.91 GB / 15.75 GiB, Google Drive ID `13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC`)
- **Dataset Year**: 2022 (5th / 6th edition AI City Challenge benchmark)
- **License**: AI City Challenge Dataset License Agreement (Academic & Research Use)

---

## B. Official Dataset Characteristics

From the official documentation (`ReadMe.txt` and Challenge specifications):
- **Total Cameras**: 46 cameras across 16 real-world intersections in a mid-sized U.S. city.
- **Maximum Distance**: 4 km between furthest simultaneous cameras.
- **Total Scenarios**: 6 scenarios:
  - `train/S01` (5 cameras, c001–c005)
  - `validation/S02` (4 cameras, c006–c009)
  - `train/S03` (6 cameras, c010–c015)
  - `train/S04` (25 cameras, c016–c040)
  - `validation/S05` (19 cameras, c010, c016–c029, c033–c036)
  - `test/S06` (6 cameras, c041–c046)
- **Total Duration**: 3.58 hours (215.03 minutes)
  - Training: 58.43 minutes
  - Validation: 136.60 minutes
  - Testing: 20.00 minutes
- **Resolutions**: At least 960p (predominantly 1280x960 and 1920x1080).
- **Framerate**: Majority 10 FPS (except camera c015 in S03 at 8 FPS).
- **Annotations**: 313,931 bounding boxes for 880 annotated vehicle identities across cameras (in `gt/gt.txt`), plus homography calibration matrices (`calibration.txt`), ROI masks (`roi.jpg`), and baseline detections (`det/`).

---

## C. Selected Subset for ARGUS

To satisfy the target of **8–12 cameras** while covering multiple intersections/scenarios without mixing splits, we lock the following **11-camera subset** composed of Scenario 1 (S01) and Scenario 3 (S03) from the `train` split:

| Canonical Camera ID | Original Path | Scenario | Original Cam | Resolution | FPS | Frame Count | Video Duration (s) | Starting Offset (s) |
|---|---|---|---|---|---|---|---|---|
| `cam_01` | `train/S01/c001/vdo.avi` | S01 | c001 | 1920x1080 | 10.0 | 1955 | 195.5 | 0.000 |
| `cam_02` | `train/S01/c002/vdo.avi` | S01 | c002 | 1920x1080 | 10.0 | 2110 | 211.0 | 1.640 |
| `cam_03` | `train/S01/c003/vdo.avi` | S01 | c003 | 1920x1080 | 10.0 | 1996 | 199.6 | 2.049 |
| `cam_04` | `train/S01/c004/vdo.avi` | S01 | c004 | 1920x1080 | 10.0 | 2110 | 211.0 | 2.177 |
| `cam_05` | `train/S01/c005/vdo.avi` | S01 | c005 | 1280x960 | 10.0 | 2110 | 211.0 | 2.235 |
| `cam_06` | `train/S03/c010/vdo.avi` | S03 | c010 | 1920x1080 | 10.0 | 2141 | 214.1 | 8.715 |
| `cam_07` | `train/S03/c011/vdo.avi` | S03 | c011 | 2560x1920 | 10.0 | 2279 | 227.9 | 8.457 |
| `cam_08` | `train/S03/c012/vdo.avi` | S03 | c012 | 2560x1920 | 10.0 | 2422 | 242.2 | 5.879 |
| `cam_09` | `train/S03/c013/vdo.avi` | S03 | c013 | 2560x1920 | 10.0 | 2415 | 241.5 | 0.000 |
| `cam_10` | `train/S03/c014/vdo.avi` | S03 | c014 | 1920x1080 | 10.0 | 2332 | 233.2 | 5.042 |
| `cam_11` | `train/S03/c015/vdo.avi` | S03 | c015 | 1920x1080 | 8.0 | 1928 | 241.0 | 8.492 |

### Subset Summary
- **Total Selected Cameras**: 11
- **Total Scenarios**: 2 (`S01`: intersection network; `S03`: roadway / corridor network)
- **Dataset Split**: `train` (100% within the training split; no split contamination or mixing)
- **Total Multi-Camera Footage**: ~40.5 minutes (2,428.0 seconds aggregate video playback)
- **Total Disk Footprint**: ~1.81 GB (1,898,397,880 bytes)

---

## D. Selection Rationale

1. **Target Range Alignment**: Exactly 11 cameras fits directly inside the 8–12 camera target window.
2. **Multi-Intersection Diversity**: Includes two distinct geographic environments:
   - Scenario 1 (center 42.525678, -90.723601): tightly clustered multi-camera intersection traffic.
   - Scenario 3 (center 42.498780, -90.686393): extended corridor with vehicles transitioning across sequential cameras.
3. **Data Completeness & Ground Truth**: Both S01 and S03 contain ground truth vehicle trajectories (`gt.txt`), enabling downstream retrieval and tracking evaluation.
4. **Clean Split Isolation**: Both scenarios are from `train`, strictly respecting the requirement not to mix train, validation, and test splits.
5. **Practical Computational Budget**: 1.77 GB allows rapid experimentation and full CV pipeline execution on laptop hardware (NVIDIA RTX 4060) without exhausting VRAM or disk space.

---

## E. Reproduction Instructions

To reproduce this exact dataset lock from scratch without downloading the full 16.9 GB archive:
1. The official archive `AICity22_Track1_MTMC_Tracking.zip` supports HTTP Range requests on Google Drive.
2. The acquisition script (`scratch/acquire_cityflow_subset.py`) reads the Central Directory entries for:
   - `train/S01/c001` through `c005`
   - `train/S03/c010` through `c015`
3. Each video stream is decompressed via zlib and written to `data/videos/cam_01/vdo.avi` through `data/videos/cam_11/vdo.avi`.
4. The manifest is deterministically locked at `data/dataset_manifest.json`.

---

## F. Validation & Stage 1 Integration

Running the Stage 1 camera inventory against `data/videos/` with these 11 real videos:
- Discovers all 11 cameras with canonical IDs `cam_01` to `cam_11`.
- Identifies each stream as `VALID` with verified frame counts, positive FPS, non-zero dimensions (1280x960 to 2560x1920), and readable frames.
- Generates deterministic `data/observations/camera_inventory.json` matching `data/dataset_manifest.json`.

---

## G. Limitations

1. **Stationary Cameras**: CityFlowV2 CCTV cameras are static pole-mounted traffic cameras; handheld or pan-tilt-zoom motion is not present.
2. **Framerate Heterogeneity**: 10 of the 11 cameras operate at 10.0 FPS, while `cam_11` (c015) operates at 8.0 FPS. The sampling pipeline in Stage 3+ must compute time-based timestamps using per-camera FPS rather than assuming a universal fixed framerate.
3. **Vehicle-Centric Annotations**: The ground-truth annotations in CityFlowV2 focus on vehicles (cars, SUVs, vans, buses, trucks). Pedestrian annotations are sparse in these highway/intersection feeds.

---

## H. Temporal & Timestamp Semantics (Frozen Contract Compliance)

### 1. Source Timestamp Semantics
Our frozen Stage 0 observation contract defines:
```text
timestamp = elapsed source-video seconds
```
Therefore, for every future sampled frame:
```text
timestamp = frame_position / source_video_fps
```
computed using the actual FPS of that specific source video. CityFlowV2 synchronization offsets must **NOT** be added to this observation `timestamp`.

### 2. CityFlow Synchronization Offset — Separate Field
The CityFlowV2 `offset_seconds` field represents the dataset's camera synchronization offset. It is **NOT** the source-video elapsed timestamp:
```text
source timestamp:
frame_index / source_fps

synchronization offset:
CityFlowV2 metadata used for cross-camera temporal alignment
```
`offset_seconds` is maintained strictly as dataset/video-level metadata in `data/dataset_manifest.json` and `camera_inventory.json`. It is **NEVER** placed into the Stage 0 observation `timestamp` field. No synchronized timestamp formula is introduced into the observation contract.

### 3. Preservation of Observation Contract
The Stage 0 observation contract remains frozen and unmodified. Future observations record:
```json
"timestamp": 142.37
```
where `142.37` strictly denotes elapsed seconds from the individual source video's timeline. The field is neither renamed nor redefined.

### 4. Per-Video FPS Handling
The selected dataset has heterogeneous framerates:
- `cam_01` through `cam_10`: 10.0 FPS
- `cam_11`: 8.0 FPS

Stage 3 frame sampling must dynamically fetch and utilize the specific `source_video_fps` for each camera stream. The pipeline must **NOT** globally assume 10.0 FPS.

---

## I. Object Scope & Detector Generality

While CityFlowV2 is a vehicle-tracking benchmark, ARGUS's Person 1 detector is **NOT** permanently restricted to vehicles:
1. **Generic Observation Contract**: The Stage 0 observation schema (`object_type: string`) remains completely generic (e.g., `car`, `person`, `bus`, `truck`, `motorcycle`, `bicycle`).
2. **Architecture Compatibility**: The future YOLO11 detection stage remains compatible with the generic ARGUS multi-class observation contract.
3. **Scope Preservation**: The observation schema is not altered to vehicle-only, nor are unsupported classes synthesized into CityFlowV2. CityFlowV2 provides the realistic video streaming substrate, while the detector architecture remains open and generic.

