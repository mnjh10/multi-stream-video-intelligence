# ARGUS: Multi-Camera Conversational Video Intelligence
### Executive & Technical Project Document

---

## 1. What is ARGUS? (The What)

**ARGUS** is an explainable, open-vocabulary, temporally indexed multi-camera surveillance intelligence platform. It replaces manual CCTV scrubbing with conversational, natural-language search and multi-turn temporal exploration across recorded multi-camera feeds.

* **The Core Problem:** Security and traffic operators monitoring dozens of surveillance feeds cannot quickly or reliably answer complex operational questions such as:
  * *"Find a red car in camera 05"*
  * *"Where was this object 10 seconds earlier?"*
  * *"Show me this same car at its next appearance."*
* **The Solution:** ARGUS ingests heterogeneous surveillance streams, tracks objects locally with persistent camera IDs, embeds appearance features into a high-capacity vector space (61,039 frozen observations), enforces query constraints with deterministic attribute verification, supports multi-turn conversational follow-up queries, and generates frame-accurate visual evidence with target-specific bounding boxes.

---

## 2. System Architecture & Core Subsystems (The How)

ARGUS is engineered into four decoupled, production-grade subsystems:

### A. Computer Vision & Observation Pipeline (Frozen Baseline)
* **Dataset:** CityFlowV2 real-world multi-camera traffic benchmark (11 synchronized cameras: `cam_01` through `cam_11`).
* **Frame Sampling:** Deterministic 3.0 FPS sampling balancing temporal continuity and processing throughput (7,287 sampled frames).
* **Object Detection:** **YOLO11x** detector (`ultralytics`) filtering vehicle and pedestrian classes (`conf >= 0.25`, 68,906 raw detections).
* **Local Trajectory Tracking:** **ByteTrack** for resilient camera-local track continuity (933 trajectories).
* **Crop Extraction:** Boundary-clamped, aspect-preserving JPEG crops (`Quality=95`) stored systematically in `data/crops/{camera_id}/`.
* **Authoritative Records:** Emits 61,039 observations into `data/observations/observations.jsonl` with strict schema validation (`observation_id`, `camera_id`, `object_id`, `timestamp`, `frame_index`, `bbox`, `crop_path`).

### B. Full-Scale Semantic Retrieval & Attribute Verification Engine
* **Multimodal Embeddings:** **OpenCLIP (`ViT-B-32`)** encodes visual crops into 512-dimensional L2-normalized vector representations.
* **Vector Index:** Full **FAISS `IndexFlatIP`** covering all 61,039 observations across all 11 cameras (`data/index/argus.index`), paired with deterministic index metadata (`data/index/argus_metadata.json`).
* **Query Parser & Constraint Extraction:** Parses natural language queries into structured parameters:
  * Camera normalization: Supports `"camera 5"`, `"camera 05"`, `"cam_05"`, `"cam 5"`, `"cam05"`, and `"camera five"`.
  * Object classification: Identifies vehicle types (`car`, `truck`, `bus`, `person`, etc.) and plurals.
  * Visual attributes: Recognizes colors (`red`, `blue`, `green`, `yellow`, `white`, `black`, `purple`, `orange`, `gray`, `silver`).
* **Calibrated HSV Attribute Verifier (`AttributeVerifier`):**
  * Evaluates vehicle color on object crops using empirical HSV distribution thresholds (18% chromatic, 25% achromatic).
  * **Achromatic Dominance & Brake-Light Rejection:** Inspects crop central ROI and rejects white/silver cars whose illuminated brake lights produce localized red pixels.
  * Assigns `verification_status="attribute_verified"` with per-attribute statistics when verified, and `verification_status="visual_similarity"` for generic queries.
  * Returns `status="no_match"` with a clear explanatory message when constraints cannot be satisfied, preventing irrelevant candidates from populating `top_k`.
* **Temporal Event Clustering (`EventBuilder`):** Groups discrete frame observations into coherent temporal events using trajectory IDs and time proximity (`max_gap = 2.0s`).

### C. Conversational Temporal Follow-Up Engine
* **Conversational Context Tracking:** Preserves `camera_id`, camera-local `object_id`, `timestamp`, `frame_index`, and `bbox` across query turns.
* **Relative & Absolute Temporal Navigation:**
  * Relative offsets: *"Where was this object 10 seconds earlier?"* (locates nearest observation to `target_time = current_time - offset`).
  * Directional queries: *"Show me this car later"* / *"next appearance"*.
* **Honest Continuity Guarantee:** Strictly treats object IDs as camera-local without fabricating cross-camera tracking identities. Returns clear notifications when context is unavailable or ambiguous.

### D. Backend API, Semantic Memory & Grounded Evidence Service
* **FastAPI Service:** Asynchronous REST API serving `/query`, `/evidence`, `/memory`, and `/health`.
* **Architectural Encapsulation:** Zero imports of FAISS, OpenCLIP, or PyTorch inside API route handlers; all ML compute is mediated through service provider interfaces.
* **Semantic Memory (`SemanticMemory`):** Relational storage mapping operator spatial nicknames (e.g., *"main gate"* ↔ `cam_01`, *"north exit"* ↔ `cam_04`).
* **Frame-Accurate Evidence Extractor (`EvidenceService`):**
  * Seeks directly to `frame_index` via OpenCV (`CAP_PROP_POS_FRAMES`), eliminating keyframe seek drift inherent in `.avi` timestamps.
  * Draws high-contrast bounding boxes around target objects, validated against actual frame dimensions (`1920x1080`).
  * Provides seamless fallback to raw unannotated frames with descriptive diagnostic messages if an object target cannot be resolved.
* **Modern Intelligence Console (React + Vite + TypeScript):**
  * Live search with natural language query suggestions and conversational follow-up pills (`"10s earlier"`, `"Show later"`, `"Next appearance"`).
  * Multi-camera directory, health monitoring, and system metrics.
  * Evidence Inspector modal with toggle between annotated bounding boxes and raw source frames.
  * Transparent verification badges (`✓ Verified: red` vs `Visual Similarity`).

---

## 3. End-to-End System Flow

```
[CityFlowV2 Multi-Camera Streams (cam_01 .. cam_11)]
                         │
                         ▼
        [Deterministic Frame Sampling @ 3.0 FPS]
                         │
                         ▼
            [YOLO11x Object Detection]
                         │
                         ▼
        [ByteTrack Camera-Local Trajectories]
                         │
                         ▼
        [Safe Object Cropping & Clamping]
                         │
                         ▼
   [authoritative observations.jsonl (61,039 records)]
                         │
                         ▼
       [OpenCLIP ViT-B-32 Vector Embeddings (512-dim)]
                         │
                         ▼
         [FAISS IndexFlatIP Index (61k vectors)]
                         │
                         ▼
    User Query ("Find a red car in camera 05" / "10s earlier")
                         │
                         ▼
        [QueryParser & Semantic Memory Resolution]
         (Camera: cam_05 | Object: car | Attr: red)
                         │
                         ▼
       [Candidate Retrieval + Temporal Event Clustering]
                         │
                         ▼
       [AttributeVerifier (HSV + Brake-Light Filter)]
             ├── Match Found ──► verification_status="attribute_verified"
             └── No Match    ──► status="no_match" (Clear rejection reason)
                         │
                         ▼
        [Temporal Follow-Up Context Engine]
         (Preserves camera_id & object_id across turns)
                         │
                         ▼
        [FastAPI Endpoint (/query & /evidence)]
                         │
                         ▼
        [OpenCV Evidence Extraction (Frame-Accurate)]
         (Draws target BBox or falls back to raw frame)
                         │
                         ▼
       [ARGUS Intelligence Console UI (React/TS)]
```

---

## 4. Architectural Trade-offs & Design Decisions (Gives and Takes)

| Component / Decision | Gives (Pros / Benefits) | Takes (Cons / Trade-offs) |
| :--- | :--- | :--- |
| **Object Crop Embeddings vs. Full Frame** | Isolates vehicles/objects; eliminates background clutter; dramatically increases retrieval accuracy for color and vehicle attributes. | Loses global scene contextual semantics (e.g., surrounding road infrastructure or broad perspective). |
| **ByteTrack Local Tracking vs. Cross-Cam Global Re-ID** | High tracking fidelity within cameras; robust against angle and illumination shifts; zero brittle cross-camera Re-ID training. | Cross-camera journeys are correlated via semantic search and temporal windows rather than an automated unified tracking ID. |
| **3.0 FPS Sampling vs. Full Video FPS (30 FPS)** | 90% reduction in storage, frame extraction I/O, and vector indexing compute while preserving traffic event continuity. | Microsecond anomalies or extreme transient motion between sampled frames may not be captured. |
| **FAISS `IndexFlatIP` vs. Approximate ANN (IVF/HNSW)** | 100% exact cosine recall; zero quantization loss; deterministic query outputs and straightforward on-disk persistence. | Linear $O(N)$ vector scan time; optimal for mid-scale (~61k vectors), but requires sharding/clustering for tens of millions of vectors. |
| **Hybrid HSV Color Verification + OpenCLIP** | Eliminates visual hallucinations; rejects white cars with brake lights; provides transparent, deterministic verification badges. | Requires empirical color calibration and crop-level pixel analysis alongside vector similarity. |
| **Frame-Index Seeking (`CAP_PROP_POS_FRAMES`)** | Frame-perfect synchronization between observation bounding boxes and extracted evidence frames on `.avi` files. | Slightly higher seek overhead compared to imprecise time-based keyframe jumping. |
| **Decoupled CV + Vector DB vs. Heavy End-to-End VLM** | Sub-100ms search latency, deterministic coordinate bounds, zero hallucinated bounding boxes, offline runnable on edge hardware. | Requires a structured pre-indexing pipeline rather than direct zero-shot raw video prompting. |

---

## 5. Technology Stack & Tools

### Computer Vision & Processing
* **YOLO11x (`ultralytics`)**: High-accuracy object detection for vehicles and pedestrians.
* **ByteTrack**: Low-latency association and camera-local trajectory tracking.
* **OpenCV (`opencv-python`)**: Video decoding, frame-accurate seeking, and bounding-box annotation.
* **Pillow (PIL)**: Safe boundary clamping and JPEG compression for object crops.
* **PyTorch & Torchvision**: GPU/CPU tensor compute engine.

### Semantic Search & Attribute Verification
* **OpenCLIP (`open-clip-torch`)**: `ViT-B-32` model generating 512-dimensional multimodal vector embeddings.
* **FAISS (`faiss-cpu`)**: Inner-product vector indexing (`IndexFlatIP`) for exact vector similarity.
* **NumPy**: Fast vectorized HSV mask computation and distribution analysis.

### Backend & Storage
* **FastAPI**: Asynchronous high-performance REST API.
* **Uvicorn**: ASGI web server.
* **SQLAlchemy & Alembic**: Relational schema persistence and query audit logging.
* **SQLite / PostgreSQL**: Storage for camera metadata, queries, and semantic aliases.
* **Pydantic v2**: Strict request and response contract validation.

### Frontend & Operator UI
* **React 18 & TypeScript**: Component-driven surveillance user interface.
* **Vite**: Modern build tooling and hot-module replacement.
* **Tailwind CSS**: Dark-mode intelligence console styling.
* **Lucide React**: Icon suite for surveillance telemetry and actions.

### Testing & Datasets
* **CityFlowV2 Dataset**: 11-camera real-world traffic intersections benchmark.
* **PyTest**: Automated test suite covering contracts, integration, query constraints, temporal follow-ups, and evidence generation.

---

## 6. Verification Metrics & Current State

* **Total Observations Indexed:** 61,039 observations (100% of frozen Person 1 dataset across 11 cameras).
* **Vector Index Capacity:** 61,039 vectors (512 dimensions, FAISS IndexFlatIP).
* **Test Suite Status:** 28/28 targeted improvement and integration tests passing (`100% pass rate`).
* **Frontend Compilation:** TypeScript type-checked and production-built with zero warnings.
* **System Latency:**
  * Vector Search & Event Clustering: `< 45ms`
  * Attribute Verification (HSV): `< 15ms` per candidate pool
  * Frame-Accurate Evidence Extraction: `< 120ms` per frame request
