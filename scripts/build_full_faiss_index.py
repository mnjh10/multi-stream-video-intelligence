"""
ARGUS — Full 11-Camera FAISS Index Builder.
Builds the complete 61,039-observation FAISS index from frozen Person 1 observations and crops.
"""

import gc
import json
import math
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
from collections import Counter

import numpy as np
import torch
from PIL import Image

from retrieval.embeddings import CLIPEmbedder
from retrieval.index_storage import IndexStorage
from retrieval.observation import Observation
from retrieval.observation_loader import iter_observations
from retrieval.vector_index import VectorIndex


def build_full_index(
    observations_path: str = "data/observations/observations.jsonl",
    index_path: str = "data/index/argus.index",
    metadata_path: str = "data/index/argus_metadata.json",
    batch_size: int = 32,
    progress_interval: int = 1000,
):
    print("=" * 60, flush=True)
    print("ARGUS — FULL 11-CAMERA FAISS INDEX BUILD", flush=True)
    print("=" * 60, flush=True)

    obs_file = Path(observations_path)
    if not obs_file.exists():
        raise FileNotFoundError(f"Observations file not found: {obs_file}")

    # Count total observations
    print("Counting total records in observations file...", flush=True)
    total_observations = 0
    with obs_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_observations += 1

    expected_batches = math.ceil(total_observations / batch_size)

    # Initialize Embedder
    embedder = CLIPEmbedder()
    device = embedder.device

    print(f"total observations = {total_observations}", flush=True)
    print(f"batch size = {batch_size}", flush=True)
    print(f"device = {device}", flush=True)
    print(f"expected number of batches = {expected_batches}", flush=True)
    print("=" * 60, flush=True)

    if total_observations != 61039:
        print(f"WARNING: Expected 61,039 observations, found {total_observations}!", flush=True)

    # Initialize FAISS Vector Index (512-dim FlatIP)
    vector_index = VectorIndex(dimension=512)

    start_time = time.time()
    batch_obs = []
    batch_tensors = []
    batch_metadata = []
    processed_count = 0
    batches_processed = 0

    print("Beginning extraction and indexing loop...", flush=True)

    for obs in iter_observations(jsonl_path=obs_file):
        crop_p = Path(obs.crop_path)
        if not crop_p.exists():
            raise FileNotFoundError(f"Crop file missing: {crop_p} for {obs.observation_id}")

        img = Image.open(crop_p).convert("RGB")
        tensor = embedder.preprocess(img)
        batch_tensors.append(tensor)
        batch_metadata.append(obs.to_dict())
        processed_count += 1

        if len(batch_tensors) >= batch_size:
            # Run inference on batch
            input_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                embeddings = embedder.model.encode_image(input_tensor)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)

            vector_index.add(embeddings=embeddings_np, metadata=batch_metadata)
            batches_processed += 1

            batch_tensors.clear()
            batch_metadata.clear()

            if processed_count % progress_interval == 0 or batches_processed % 30 == 0:
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total_observations - processed_count) / speed if speed > 0 else 0
                vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                print(
                    f"[Progress] Processed {processed_count}/{total_observations} "
                    f"({processed_count / total_observations * 100:.1f}%) | "
                    f"Batches: {batches_processed}/{expected_batches} | "
                    f"Speed: {speed:.1f} obs/s | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {remaining:.1f}s | "
                    f"VRAM: {vram_mb:.1f} MB",
                    flush=True,
                )

    # Process final leftover batch
    if batch_tensors:
        input_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embeddings = embedder.model.encode_image(input_tensor)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        vector_index.add(embeddings=embeddings_np, metadata=batch_metadata)
        batches_processed += 1
        batch_tensors.clear()
        batch_metadata.clear()

    total_time = time.time() - start_time
    print("=" * 60, flush=True)
    print(f"Indexing completed in {total_time:.2f} seconds ({total_time / 60:.2f} mins)!", flush=True)
    print(f"Average throughput: {processed_count / total_time:.1f} obs/s", flush=True)
    print(f"Total vectors in FAISS index: {vector_index.index.ntotal}", flush=True)
    print(f"Total metadata records: {len(vector_index.metadata)}", flush=True)
    print("=" * 60, flush=True)

    # Persist Index
    print(f"Saving FAISS index to {index_path}...", flush=True)
    print(f"Saving metadata to {metadata_path}...", flush=True)
    storage = IndexStorage()
    storage.save(
        vector_index=vector_index,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    index_size_bytes = Path(index_path).stat().st_size
    meta_size_bytes = Path(metadata_path).stat().st_size
    print(f"Index size on disk: {index_size_bytes / (1024**2):.2f} MB", flush=True)
    print(f"Metadata size on disk: {meta_size_bytes / (1024**2):.2f} MB", flush=True)

    # Verification
    print("=" * 60, flush=True)
    print("RUNNING IMMEDIATE INTEGRITY AND RELOAD AUDIT...", flush=True)

    reloaded_index = VectorIndex(dimension=512)
    storage.load(
        vector_index=reloaded_index,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    print(f"Reloaded ntotal: {reloaded_index.index.ntotal}", flush=True)
    print(f"Reloaded metadata count: {len(reloaded_index.metadata)}", flush=True)
    print(f"Reloaded dimension: {reloaded_index.index.d}", flush=True)

    # Unique IDs
    ids = [m["observation_id"] for m in reloaded_index.metadata]
    unique_ids = set(ids)
    print(f"Unique observation IDs: {len(unique_ids)} / {len(ids)}", flush=True)

    # Camera breakdown
    cam_counts = Counter(m["camera_id"] for m in reloaded_index.metadata)
    print("Camera breakdown in reloaded index:", flush=True)
    for cam in sorted(cam_counts.keys()):
        print(f"  {cam}: {cam_counts[cam]}", flush=True)

    # Position mapping checks
    test_positions = [0, 1, 100, 1000, 10000, 30000, 50000, 61038]
    print("\nDeterministic position mapping checks:", flush=True)
    for pos in test_positions:
        if pos < len(reloaded_index.metadata):
            meta = reloaded_index.metadata[pos]
            print(f"  Pos {pos:5d} -> obs_id: {meta['observation_id']}, camera_id: {meta['camera_id']}, time: {meta['timestamp']}", flush=True)

    # Vector norm check
    print("\nVector reconstruction and norm check for sample positions:", flush=True)
    for pos in [0, 1000, 30000, 61038]:
        if pos < reloaded_index.index.ntotal:
            vec = reloaded_index.index.reconstruct(pos)
            norm = float(np.linalg.norm(vec))
            print(f"  Pos {pos:5d} -> reconstructed norm: {norm:.6f}", flush=True)

    print("=" * 60, flush=True)
    print("INDEX BUILD AND PERSISTENCE AUDIT COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    build_full_index()
