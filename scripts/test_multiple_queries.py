import urllib.request
import json

queries = [
    'find a car',
    'find a red car',
    'find a truck',
    'white vehicle',
    'car in camera 2',
]

print("Testing direct backend on port 8000:")
for i, q in enumerate(queries, 1):
    req = urllib.request.Request(
        'http://127.0.0.1:8000/query',
        data=json.dumps({'query': q, 'top_k': 5}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            print(f'Query {i} ("{q}"): status={data["status"]}, results={len(data["results"])}')
    except Exception as e:
        print(f'Query {i} ("{q}") FAILED: {e}')

print("\nTesting Vite proxy on port 5173:")
for i, q in enumerate(queries, 1):
    req = urllib.request.Request(
        'http://localhost:5173/query',
        data=json.dumps({'query': q, 'top_k': 5}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            print(f'Query {i} ("{q}"): status={data["status"]}, results={len(data["results"])}')
    except Exception as e:
        print(f'Query {i} ("{q}") FAILED: {e}')
