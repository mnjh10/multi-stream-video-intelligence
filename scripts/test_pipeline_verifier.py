import json
from retrieval.attribute_verifier import AttributeVerifier
from backend.services.retrieval import get_retrieval_provider

p = get_retrieval_provider()
print("Provider loaded.")

# Test query
query = "Find a red car in camera 05"
print(f"Testing query: '{query}'")

res = p.search(query, top_k=5)
print(f"Results count: {len(res)}")
for r in res:
    print(f"Event: {r['event_id']}, Cam: {r['camera_id']}, Obj: {r['object_id']}, TS: {r['best_timestamp']}s, Score: {r['score']:.3f}")
    print(f"  BBox: {r.get('bbox')}, Frame: {r.get('frame_index')}, Status: {r.get('verification_status')}")
    print(f"  Attr details: {r.get('attribute_details')}")
