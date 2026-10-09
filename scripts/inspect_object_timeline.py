import json

with open('data/index/argus_metadata.json', 'r') as f:
    meta = json.load(f)

cam6_obs = [m for m in meta if m['camera_id'] == 'cam_06']
obj_counts = {}
for m in cam6_obs:
    obj_counts[m['object_id']] = obj_counts.get(m['object_id'], 0) + 1

top_objs = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print('Top objects in cam_06:', top_objs)

sample_obj = top_objs[0][0]
obj_timeline = [m for m in cam6_obs if m['object_id'] == sample_obj]
obj_timeline.sort(key=lambda x: x['timestamp'])
print(f"Timeline for {sample_obj} ({len(obj_timeline)} observations):")
for m in obj_timeline[:5]:
    print(f"  time={m['timestamp']:.2f}s, frame={m['frame_index']}, bbox={m['bbox']}")
print("  ...")
for m in obj_timeline[-2:]:
    print(f"  time={m['timestamp']:.2f}s, frame={m['frame_index']}, bbox={m['bbox']}")
