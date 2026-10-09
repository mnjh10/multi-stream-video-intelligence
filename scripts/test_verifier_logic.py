import cv2
import json
import numpy as np

with open('data/index/argus_metadata.json') as f:
    meta = json.load(f)

# Colors and their HSV ranges
# OpenCV HSV: H in [0, 180], S in [0, 255], V in [0, 255]
COLOR_RANGES = {
    "red": [
        (np.array([0, 70, 50]), np.array([10, 255, 255])),
        (np.array([170, 70, 50]), np.array([180, 255, 255]))
    ],
    "blue": [
        (np.array([100, 70, 50]), np.array([135, 255, 255]))
    ],
    "green": [
        (np.array([35, 60, 50]), np.array([85, 255, 255]))
    ],
    "yellow": [
        (np.array([15, 80, 70]), np.array([35, 255, 255]))
    ],
    "white": [
        # Low saturation, high brightness
        (np.array([0, 0, 160]), np.array([180, 45, 255]))
    ],
    "black": [
        # Very low brightness
        (np.array([0, 0, 0]), np.array([180, 255, 55]))
    ],
    "gray": [
        # Low saturation, medium brightness
        (np.array([0, 0, 55]), np.array([180, 40, 160]))
    ],
    "grey": [
        (np.array([0, 0, 55]), np.array([180, 40, 160]))
    ],
}

def verify_color(crop_path: str, color_name: str) -> tuple[bool, float, str]:
    if not crop_path:
        return False, 0.0, "missing_crop"
    img = cv2.imread(crop_path)
    if img is None:
        return False, 0.0, "unreadable_image"

    h, w = img.shape[:2]
    # Center crop (inner 80%) to reduce background road asphalt influence
    y_margin = int(h * 0.1)
    x_margin = int(w * 0.1)
    roi = img[y_margin:h-y_margin, x_margin:w-x_margin]
    if roi.size == 0:
        roi = img

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = roi.shape[0] * roi.shape[1]

    ranges = COLOR_RANGES.get(color_name.lower())
    if not ranges:
        return False, 0.0, "unsupported_attribute"

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    fraction = np.count_nonzero(mask) / total_pixels

    # Conservative threshold:
    # Colors with narrow chromatic bands (red, blue, green, yellow): >= 0.18
    # Achromatic (white, black, gray): >= 0.25
    thresh = 0.25 if color_name.lower() in ("white", "black", "gray", "grey") else 0.18
    
    if fraction >= thresh:
        return True, fraction, "attribute_verified"
    elif fraction >= (thresh * 0.55):
        return False, fraction, "unverified_candidate"
    else:
        return False, fraction, "rejected"

# Test on cam_05 cars
c5_cars = [m for m in meta if m['camera_id'] == 'cam_05' and m['object_type'] == 'car']
checked_objs = {}
for m in c5_cars:
    oid = m['object_id']
    if oid not in checked_objs:
        is_red, frac, status = verify_color(m['crop_path'], 'red')
        checked_objs[oid] = (is_red, frac, status, m['timestamp'], m['crop_path'])

verified_red = {k: v for k, v in checked_objs.items() if v[0]}
print(f"Total cars in cam_05: {len(checked_objs)}")
print(f"Verified RED cars in cam_05: {len(verified_red)}")
for k, v in verified_red.items():
    print(f"  {k}: frac={v[1]*100:.1f}%, status={v[2]}, ts={v[3]}s, crop={v[4]}")

# Also check white
verified_white = {k: v for k, v in checked_objs.items() if verify_color(v[4], 'white')[0]}
print(f"\nVerified WHITE cars in cam_05: {len(verified_white)}")
for k, v in list(verified_white.items())[:5]:
    is_w, frac, st = verify_color(v[4], 'white')
    print(f"  {k}: frac={frac*100:.1f}%, ts={v[3]}s")
