"""
Verification script for full FAISS index persistence and retrieval.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import numpy as np
from collections import Counter
from backend.services.retrieval import RealRetrievalProvider
from retrieval.vector_index import VectorIndex
from retrieval.index_storage import IndexStorage


def run_checks():
    print("=" * 60)
    print("ARGUS — PERSISTENCE AND POSITION MAPPING VERIFICATION")
    print("=" * 60)

    index_path = "data/index/argus.index"
    metadata_path = "data/index/argus_metadata.json"

    # Reload 1
    idx1 = VectorIndex(dimension=512)
    storage = IndexStorage()
    storage.load(idx1, index_path, metadata_path)

    # Reload 2 (simulate fresh load)
    idx2 = VectorIndex(dimension=512)
    storage.load(idx2, index_path, metadata_path)

    assert idx1.index.ntotal == 61039, f"Expected 61039, got {idx1.index.ntotal}"
    assert len(idx1.metadata) == 61039, f"Expected 61039, got {len(idx1.metadata)}"
    assert idx1.index.d == 512

    # Determinism across positions
    test_positions = [0, 1, 100, 1000, 10000, 30000, 50000, 61038]
    print("Verifying position mappings across reloads:")
    for pos in test_positions:
        m1 = idx1.metadata[pos]
        m2 = idx2.metadata[pos]
        assert m1 == m2, f"Metadata mismatch at pos {pos}"
        v1 = idx1.index.reconstruct(pos)
        v2 = idx2.index.reconstruct(pos)
        assert np.allclose(v1, v2, atol=1e-6), f"Vector mismatch at pos {pos}"
        print(f"  Pos {pos:5d} -> obs_id: {m1['observation_id']}, cam: {m1['camera_id']}, frame: {m1['frame_index']}, time: {m1['timestamp']}")

    print("[PASS] Position mappings and vectors perfectly match across reloads.")

    # Camera breakdown
    cam_counts = Counter(m["camera_id"] for m in idx1.metadata)
    print("\nPer-Camera Observation Counts:")
    expected_counts = {
        "cam_01": 3956,
        "cam_02": 6063,
        "cam_03": 5324,
        "cam_04": 4308,
        "cam_05": 5267,
        "cam_06": 4357,
        "cam_07": 6104,
        "cam_08": 5949,
        "cam_09": 1810,
        "cam_10": 5948,
        "cam_11": 11953,
    }
    for cam in sorted(expected_counts.keys()):
        actual = cam_counts.get(cam, 0)
        exp = expected_counts[cam]
        assert actual == exp, f"Camera count mismatch for {cam}: got {actual}, expected {exp}"
        print(f"  {cam}: {actual:5d} (Expected: {exp:5d}) [MATCH]")

    print(f"Total across 11 cameras: {sum(cam_counts.values())} (Expected: 61039) [MATCH]")

    # Check Real Retrieval Provider queries
    print("\n" + "=" * 60)
    print("ARGUS — REAL MULTI-CAMERA RETRIEVAL TESTS")
    print("=" * 60)

    provider = RealRetrievalProvider()

    test_queries = [
        ("car", None),
        ("Find a car in camera 1", "cam_01"),
        ("Find a car in camera 2", "cam_02"),
        ("Find a car in camera 6", "cam_06"),
        ("Find a car in camera 11", "cam_11"),
        ("white car", None),
    ]

    for query, expected_cam in test_queries:
        results = provider.search(query, top_k=5)
        cams = [r["camera_id"] for r in results]
        scores = [f"{r['score']:.4f}" for r in results]
        event_ids = [r["event_id"] for r in results]
        obj_types = [r["object_type"] for r in results]
        timestamps = [f"{r['timestamp']:.2f}s" for r in results]

        print(f"Query: \"{query}\"")
        print(f"  Returned cameras: {cams}")
        print(f"  Scores:           {scores}")
        print(f"  Event IDs:        {event_ids}")
        print(f"  Object Types:     {obj_types}")
        print(f"  Timestamps:       {timestamps}")

        if expected_cam:
            assert len(results) > 0, f"Expected results for {expected_cam}, got 0"
            assert all(c == expected_cam for c in cams), f"Expected all {expected_cam}, got {cams}"
            print(f"  [PASS] All {len(cams)} results strictly match {expected_cam}")
        else:
            unique_cams = list(dict.fromkeys(cams))
            print(f"  [INFO] Unique cameras in top-{len(results)}: {unique_cams}")
        print("-" * 50)

    # Retrieval determinism check across 2 searches
    r1 = provider.search("white car", top_k=5)
    r2 = provider.search("white car", top_k=5)
    assert [x["event_id"] for x in r1] == [x["event_id"] for x in r2]
    for x1, x2 in zip(r1, r2):
        assert abs(x1["score"] - x2["score"]) < 1e-5
    print("[PASS] Retrieval determinism verified.")


if __name__ == "__main__":
    run_checks()
