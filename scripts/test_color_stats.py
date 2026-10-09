import cv2
import json
import random
import numpy as np
from pathlib import Path

with open('data/index/argus_metadata.json') as f:
    meta = json.load(f)

# Inspect cam_05 cars returned by OpenCLIP for "Find a red car in camera 05"
c5_cars = [m for m in meta if m['camera_id'] == 'cam_05' and m['object_type'] == 'car']
top_candidates = ['obj_cam05_000119', 'obj_cam05_000105', 'obj_cam05_000044', 'obj_cam05_000048', 'obj_cam05_000017', 'obj_cam05_000115', 'obj_cam05_000114']

print("--- Inspecting cam_05 Top Candidate Crops ---")
for obj_id in top_candidates:
    obs = [m for m in c5_cars if m['object_id'] == obj_id]
    if not obs: continue
    mid = obs[len(obs)//2]
    img = cv2.imread(mid['crop_path'])
    if img is None: continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red hue: [0, 10] or [170, 180], S >= 60, V >= 50
    m1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 60, 50]), np.array([180, 255, 255]))
    red_frac = np.count_nonzero(m1 | m2) / (img.shape[0] * img.shape[1])
    
    # White: S <= 40, V >= 160
    white_m = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 40, 255]))
    white_frac = np.count_nonzero(white_m) / (img.shape[0] * img.shape[1])
    
    # Black/Dark: V <= 60
    black_m = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    black_frac = np.count_nonzero(black_m) / (img.shape[0] * img.shape[1])
    
    print(f"{obj_id} @ {mid['timestamp']:.1f}s ({mid['crop_path']}): Red={red_frac*100:.1f}%, White={white_frac*100:.1f}%, Black={black_frac*100:.1f}%")
