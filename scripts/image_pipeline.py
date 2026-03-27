import cv2
import os
from detect_and_crop import BusDetector
from ocr_pipeline import run_ocr_on_image

MODEL_PATH = "../models/best.pt"
IMAGE_PATH = "../data/train/images/bus_video5_294.jpg"
TEMP_DIR = "../temp_crops"
CONF_THRESH = 0.4


os.makedirs(TEMP_DIR, exist_ok=True)

print("[INFO] Loading detector...")
detector = BusDetector(MODEL_PATH, CONF_THRESH)

print("[INFO] Loading image...")
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Failed to load image"

print("[INFO] Running detection...")
results = detector.detect(img)

print("[INFO] Cropping detections...")
crops = detector.crop_boxes(img, results, TEMP_DIR)

final_output = []


for i, bus in enumerate(crops["bus_front"]):
    bus_x1, bus_y1, bus_x2, bus_y2 = bus["bbox"]

    bus_routes = []
    bus_dests = []

    
    for r in crops["route_number"]:
        x1, y1, x2, y2 = r["bbox"]
        if bus_x1 <= x1 <= bus_x2:
            bus_routes.append(r)

    for d in crops["destination"]:
        x1, y1, x2, y2 = d["bbox"]
        if bus_x1 <= x1 <= bus_x2:
            bus_dests.append(d)

    route_text = ""
    dest_text = ""

    if bus_routes:
        route_text = run_ocr_on_image(bus_routes[0]["path"])["text"]

    if bus_dests:
        dest_text = run_ocr_on_image(bus_dests[0]["path"])["text"]

    final_output.append({
        "bus_id": i,
        "route": route_text,
        "destination": dest_text
    })


print("\n================ FINAL BUS OUTPUT ================\n")
for bus in final_output:
    print(f"BUS {bus['bus_id']}")
    print(f"ROUTE      : {bus['route']}")
    print(f"DESTINATION: {bus['destination']}")
    print("--------------------------------------------------")
