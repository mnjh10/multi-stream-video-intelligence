import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.services.retrieval import RealRetrievalProvider

p = RealRetrievalProvider()

test_queries = [
    'find a car',
    'find a red car',
    'find a blue car',
    'find a black car',
    'find a white car',
    'find a truck',
    'find a person',
    'find a bicycle',
    'find a dog',
    'red car',
    'speeding car',
    'car turning right',
    'car at night',
    'Find a car in camera 1',
    'Find a car in camera 5',
    'Find a car in camera 11',
]

for q in test_queries:
    try:
        res = p.search(q, top_k=5)
        top_score = f"{res[0]['score']:.4f}" if res else "NONE"
        cams = [r['camera_id'] for r in res]
        print(f"{q:30s} -> {len(res)} results | Top: {top_score} | Cams: {cams}")
    except Exception as e:
        print(f"{q:30s} -> ERROR: {e}")
