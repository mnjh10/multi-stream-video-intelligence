# ARGUS

## Conversational Multi-Camera Video Intelligence

ARGUS is a multi-camera video intelligence system that allows users to search recorded surveillance footage using natural-language queries.

Instead of manually reviewing hours of CCTV footage, a user can ask questions such as:

> "Find the red car near the main gate."

> "Find the person carrying a backpack."

> "Where did the same vehicle appear next?"

ARGUS processes multiple recorded camera streams, indexes visual events, retrieves relevant evidence, and returns the camera, timestamp, and supporting visual evidence.

The project is being developed as a hackathon prototype with an emphasis on **grounded retrieval, temporal localization, persistent semantic memory, and explainable evidence**.

---

## Problem

Traditional CCTV systems require users to manually inspect individual camera feeds and search through large amounts of recorded footage.

ARGUS aims to provide a conversational interface where users can describe what they are looking for in natural language.

The system should be able to answer questions involving:

- Objects
- Visual attributes
- Cameras
- Locations
- Time ranges
- Temporal events
- Cross-camera appearances
- References to previously identified objects
- Persistent semantic locations

For example:

```text
User:
"Find the person carrying a backpack near the entrance."

ARGUS:
Camera: CAM-01
Timestamp: 02:31
Evidence: [frame / video clip]
```

---

# Core Idea

The fundamental design principle is:

> **The language model interprets the user's intent; the retrieval system finds the evidence.**

ARGUS should not rely on an LLM to simply describe video and generate an answer.

Instead, the system creates a searchable representation of the visual data and grounds the final answer in actual observations from the recorded footage.

---

# Architecture

The current architecture is:

```text
                    RECORDED VIDEOS
                           │
                           ▼
                  Video Ingestion
                           │
                           ▼
                    Frame Sampling
                           │
                           ▼
                     YOLO11
                  Object Detection
                           │
                           ▼
                 Object Observations
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
              Object Crops    Metadata
                    │             │
                    ▼             ▼
                OpenCLIP      PostgreSQL
                    │
                    ▼
             Visual Embeddings
                    │
                    ▼
                   FAISS
                    │
                    │
Natural Language ──► Query Parser
                    │
                    ▼
             Structured Query
                    │
                    ▼
             Candidate Retrieval
                    │
                    ▼
             Query-Aware Ranking
                    │
                    ▼
              Temporal Events
                    │
                    ▼
            Camera + Timestamp
                    │
                    ▼
             Evidence Generation
                    │
                    ▼
                  Backend
                    │
                    ▼
                 Frontend
```

---

# Dataset

## Primary Dataset

The project uses the:

**AI City Challenge / CityFlowV2**

multi-camera video dataset.

CityFlowV2 provides real multi-camera surveillance footage and vehicle tracking information across multiple cameras and scenarios.

We will use a selected subset rather than processing the entire dataset.

### Target evaluation scale

Approximately:

- 8–12 cameras
- Multiple scenarios
- Longer temporal windows
- Multiple object/event types where supported by the footage

A smaller subset may be used during early development to reduce processing time.

---

# Object Scope

ARGUS is designed as a **generic object-oriented video intelligence system**, not a vehicle-only system.

The detection pipeline should support whatever relevant everyday objects are present in the selected footage.

Examples include:

```text
person
car
bicycle
motorcycle
bus
truck
backpack
handbag
suitcase
umbrella
bottle
dog
cat
laptop
cell phone
...
```

The exact detectable categories depend on the selected footage and detector.

The architecture therefore uses generic concepts such as:

```text
object_id
object_type
```

rather than designing the system exclusively around vehicles.

---

# Technology Stack

## Programming

```text
Python 3.11
```

## Computer Vision

```text
PyTorch
Ultralytics YOLO11
OpenCV
FFmpeg
NumPy
Pandas
```

## Vision-Language Retrieval

```text
OpenCLIP
Transformers
```

## Vector Search

```text
FAISS
```

## Database

```text
PostgreSQL
SQLAlchemy
Alembic
psycopg
```

## Backend

```text
FastAPI
Uvicorn
Pydantic
python-dotenv
```

## Testing

```text
pytest
```

## Frontend

```text
React
```

Frontend development and backend/frontend integration will be handled later.

---

# Repository Structure

The three developers are working on independent branches.

The internal folder structure of each branch may differ.

The important requirement is that the shared interfaces and data contracts remain compatible.

A possible overall structure is:

```text
ARGUS/
│
├── backend/
│
├── frontend/
│
├── data/
│   ├── videos/
│   ├── frames/
│   ├── crops/
│   └── embeddings/
│
├── models/
│
├── evaluation/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── .env.example
└── README.md
```

The exact structure will evolve as development progresses.

---

# Git Branches

The current development branches are:

```text
main
│
├── cv-pipeline
├── retrieval-engine
└── backend-foundation
```

### `main`

Stable project branch.

No experimental work should be performed directly on `main`.

### `cv-pipeline`

Owned by Person 1.

Responsible for:

- Video ingestion
- Frame sampling
- YOLO11 detection
- Object observations
- Object crops
- Tracking where practical
- Timestamp handling
- CV preprocessing

### `retrieval-engine`

Owned by Person 2.

Responsible for:

- OpenCLIP embeddings
- FAISS
- Query understanding
- Metadata filtering
- Semantic retrieval
- Ranking
- Temporal retrieval
- Baseline implementation
- Evaluation

### `backend-foundation`

Owned by Person 3.

Responsible for:

- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Persistent semantic memory
- Backend schemas
- Evidence generation utilities
- Mock retrieval interfaces
- Backend testing

Backend/frontend integration will happen later.

---

# Shared Data Contract

The internal implementation of each branch may differ, but the interfaces between components must remain stable.

## Person 1 → Person 2

Person 1 produces observations approximately following:

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

Required concepts:

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

---

# Person 2 → Person 3

The retrieval engine eventually returns semantic results approximately following:

```json
{
  "event_id": "evt_001",
  "camera_id": "cam_01",
  "object_id": "obj_cam01_000034",
  "timestamp_start": 140.2,
  "timestamp_end": 148.7,
  "best_timestamp": 144.2,
  "score": 0.91,
  "object_type": "car",
  "source_video": "cam_01/video.mp4"
}
```

The backend should depend on this conceptual result rather than FAISS-specific implementation details.

---

# Backend → Frontend

The eventual API should expose information such as:

```text
query
results
camera
timestamp
event
score
evidence frame
evidence clip
memory/clarification state
```

The exact API contract will be finalized during integration.

---

# Persistent Semantic Memory

ARGUS must support **clarify once, remember permanently**.

Example:

```text
User:
"Did someone enter through the main gate?"

ARGUS:
"Which camera should I associate with the main gate?"

User:
"Camera 1."
```

The system stores:

```text
main gate → cam_01
```

After restarting the application:

```text
User:
"Did someone enter through the main gate?"
```

ARGUS should already understand:

```text
main gate → cam_01
```

without requesting clarification again.

Persistent memory is stored in PostgreSQL.

---

# Retrieval Approach

ARGUS will compare two retrieval approaches.

## Baseline

```text
Sampled frames
      ↓
OpenCLIP embeddings
      ↓
FAISS
      ↓
Top-K results
```

## Proposed approach

```text
Object detection
      ↓
Object-level crops
      ↓
OpenCLIP embeddings
      ↓
Metadata filtering
      ↓
FAISS
      ↓
Temporal aggregation
      ↓
Query-aware ranking
```

This allows us to evaluate whether object-aware and temporally structured retrieval improves natural-language video search.

---

# Research Question

The primary research question is:

> **Does combining object-aware semantic indexing and temporal event aggregation improve natural-language multi-camera video retrieval compared with frame-level embedding retrieval?**

The evaluation will be based on actual experimental results rather than predetermined claims.

---

# Evaluation

Planned metrics include:

### Recall@1

Whether the correct result is ranked first.

### Recall@5

Whether the correct result appears in the top five.

### Mean Reciprocal Rank

Measures ranking quality.

### Camera Accuracy

Whether the correct camera is identified.

### Timestamp Error

Difference between the returned timestamp and ground-truth event timestamp.

### Query Latency

Time required to retrieve results.

---

# Query Categories

The evaluation set will contain several classes of queries.

### Object

```text
"Find the car near camera 2."
```

### Attribute

```text
"Find the white vehicle."
```

### Location

```text
"Find the person near the entrance."
```

### Temporal

```text
"Find what happened during the last minute."
```

### Combined

```text
"Find the white car near camera 3 during the last minute."
```

### Cross-Camera

```text
"Where did this vehicle appear next?"
```

### Referential

```text
"Where did it appear next?"
```

These queries will be grounded against actual video events.

---

# Development Philosophy

The project prioritizes:

```text
1. Retrieval accuracy
2. Camera localization
3. Timestamp localization
4. Evidence generation
5. Persistent memory
6. Evaluation
7. Stretch features
```

Do not sacrifice the core retrieval system for flashy features.

---

# Stretch Goals

If the MVP is stable, potential extensions include:

- Cross-camera re-identification
- Improved temporal reasoning
- Live-stream simulation
- Standing queries / alerts
- Privacy-aware processing
- More advanced visual attributes
- More sophisticated event reasoning

Stretch features must not destabilize the core system.

---

# Scope Boundaries

The project will NOT initially include:

- Custom detector training
- Custom VLM training
- Full live CCTV infrastructure
- Production-scale distributed systems
- Kubernetes
- Multiple competing vector databases
- Multiple competing ML pipelines
- Complex microservice architecture
- Full production authentication/authorization

The objective is a strong, demonstrable research prototype.

---

# Development Workflow

Each person works independently on their branch.

Typical workflow:

```bash
git switch <your-branch>
git pull

# work

git add .
git commit -m "Meaningful commit message"
git push
```

Do not make architectural changes affecting another person's component without communicating them first.

---

# Synchronization Rules

The main coordination chat is the source of truth for:

- Architecture
- Shared schemas
- API contracts
- Model choices
- Dataset decisions
- Scope changes
- Integration requirements
- Breaking changes
- Hackathon strategy

Individual development chats are for implementation.

If a change affects another component, report it in the main coordination channel.

---

# Status Format

Use the following format for development updates:

```text
STATUS — PERSON X

Completed:
-

Currently working:
-

Blocked:
-

Files changed:
-

Inputs required from:
-

Outputs now available to:
-

Interface/schema changes:
-

Performance:
-

Known issues:
-

Next:
-
```

For breaking/shared changes:

```text
CHANGE NOTICE

Component:

Old:

New:

Reason:

Affected:

Action required:
```

---

# Current Development Phase

The current phase is:

**Independent component development.**

The three branches are being developed independently while respecting the shared architecture and contracts.

Integration will occur only after the component-level implementations are sufficiently stable.

The eventual integration flow will be:

```text
cv-pipeline
      +
retrieval-engine
      +
backend-foundation
      ↓
integration
      ↓
Testing
      ↓
main
```

---

# Project Goal

ARGUS should ultimately demonstrate:

> **An explainable, open-vocabulary, temporally indexed multi-camera video retrieval system that converts natural-language queries into grounded camera, timestamp, and visual evidence while maintaining persistent semantic memory.**

The goal is not simply to build a CCTV dashboard.

The goal is to demonstrate a technically defensible approach to **conversational, grounded, multi-camera video intelligence**.