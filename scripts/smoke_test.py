import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# 1. POST /query
res = client.post('/query', json={'query': 'Find a car in camera 6', 'top_k': 5})
assert res.status_code == 200, f'Query status: {res.status_code}'
data = res.json()
assert data['status'] == 'ok'
assert len(data['results']) > 0
results = data['results']
assert all(r['camera_id'] == 'cam_06' for r in results)

first = results[0]
event_id = first['event_id']
camera_id = first['camera_id']
timestamp = first['timestamp']
video_path = first['evidence']['source_video']
scores = [f"{r['score']:.4f}" for r in results]

print('Query PASS')
print(f'Camera: {camera_id}')
print(f'Timestamp: {timestamp}')
print(f'Event ID: {event_id}')
print(f'Source Video: {video_path}')
print(f'Top scores: {scores}')

# 2. POST /evidence
ev_res = client.post('/evidence', json={
    'source_video': video_path,
    'timestamp': timestamp,
    'camera_id': camera_id,
    'event_id': event_id
})
assert ev_res.status_code == 200, f'Evidence status: {ev_res.status_code}'
ev_data = ev_res.json()
assert ev_data['status'] == 'ok'
frame_path = ev_data['frame_path']
assert Path(frame_path).exists()
size = Path(frame_path).stat().st_size
assert size > 1000

print('Evidence PASS')
print(f'Evidence frame: {frame_path} ({size} bytes)')
