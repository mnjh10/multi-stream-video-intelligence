import cv2
import json
from pathlib import Path

# Load metadata for cam_06 at 207.7s
with open('data/index/argus_metadata.json', 'r') as f:
    meta = json.load(f)

obs = next(m for m in meta if m['camera_id'] == 'cam_06' and m['object_id'] == 'obj_cam06_000004' and abs(m['timestamp'] - 207.7) < 0.1)
print('Found target observation:', obs['observation_id'], obs['bbox'], obs['frame_index'])

# Extract frame from video
cap = cv2.VideoCapture('data/videos/cam_06/vdo.avi')
cap.set(cv2.CAP_PROP_POS_MSEC, 207.7 * 1000.0)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
x1, y1, x2, y2 = [int(round(v)) for v in obs['bbox']]
# Clip to frame
x1, y1 = max(0, x1), max(0, y1)
x2, y2 = min(w, x2), min(h, y2)

print(f"Frame size: {w}x{h}, BBox: ({x1}, {y1}) to ({x2}, {y2})")

# Draw bounding box
color = (0, 229, 255) # BGR cyan
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

label = f"{obs['object_type']} | {obs['object_id']} | {obs['timestamp']:.1f}s"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 2
(text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

# Draw label background
label_y1 = max(0, y1 - text_h - 10)
label_y2 = y1
cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 10, label_y2), (18, 24, 32), -1)
cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 10, label_y2), color, 1)
cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

out_test = 'data/evidence/test_annotated_cam06.jpg'
cv2.imwrite(out_test, frame)
print('Saved test annotated frame:', out_test, Path(out_test).stat().st_size, 'bytes')
