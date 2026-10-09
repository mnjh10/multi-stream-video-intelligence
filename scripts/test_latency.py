import urllib.request
import json
import time

queries = [
    'find a car',
    'find a red car',
    'find a truck',
    'find a blue car',
    'white car',
]

for i, q in enumerate(queries, 1):
    t0 = time.time()
    req = urllib.request.Request(
        'http://127.0.0.1:8000/query',
        data=json.dumps({'query': q, 'top_k': 5}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
            dur = time.time() - t0
            print(f'Query {i} ({q}): {dur:.2f}s | status={data["status"]} | results={len(data["results"])}')
    except Exception as e:
        dur = time.time() - t0
        print(f'Query {i} ({q}): {dur:.2f}s | FAILED: {e}')
