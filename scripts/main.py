import os
import gc
import time
import cv2
import psutil
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
from rapidfuzz import process, fuzz
from ocr_pipeline import run_ocr_on_image
from tts_player import announce
import matplotlib.pyplot as plt
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ================= CONFIG =================
VIDEO_SOURCE = "../data/data-test/Test_night.mp4"

BUS_MODEL_PATH = "../models/best.pt"
PERSON_MODEL_PATH = "yolov8n-pose.pt"

ROUTES_FILE = "../data/routes.txt"

CONF_THRESH = 0.25
IMG_SIZE = 640
FRAME_SKIP = 1

TEMPORAL_WINDOW = 7
STABLE_THRESHOLD = 2

LEVENSHTEIN_THRESHOLD = 70

# minimum frames a bus_front must be seen before counting as a real detection
MIN_FRAMES_TO_COUNT = 10

# display every Nth frame (2 = half speed rendering, 3 = one third, etc.)
DISPLAY_SKIP = 2

# smaller display size = faster rendering
DISPLAY_SIZE = (800, 450)

# ================= BUS STOP CONFIG =================

BUS_STOP_ID = "stop_1"

STOP_BUS_LIST = {
    "stop_1": ["8","205"],
    "stop_2": ["425","25","86","98"],
    "stop_3": ["366","238","9"],
    "stop_4" : ["25","86","256","276","104","D8","262"]
}

# ================= DISTANCE ESTIMATION =================

REAL_BUS_WIDTH = 2.6
FOCAL_LENGTH_PIXELS = 800

bus_detection_widths = {}

# ================= MEMORY METRICS =================

peak_cpu_mem = 0
peak_gpu_mem = 0

# capped at 500 samples (~every 50 frames for 25000 frames) to prevent unbounded growth
MEM_LOG_MAXLEN = 500
mem_log_frames = deque(maxlen=MEM_LOG_MAXLEN)
mem_log_cpu    = deque(maxlen=MEM_LOG_MAXLEN)
mem_log_gpu    = deque(maxlen=MEM_LOG_MAXLEN)

# ================= PASSENGER MEMORY =================

PERSON_MEMORY_SECONDS = 3
last_person_seen_time = 0

# ================= GPU INFO =================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

# ================= LOAD ROUTES =================

with open(ROUTES_FILE,"r") as f:
    VALID_DESTINATIONS = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded {len(VALID_DESTINATIONS)} valid destinations")

# ================= LOAD MODELS =================

print("[INFO] Loading person model...")
person_model = YOLO(PERSON_MODEL_PATH)

print("[INFO] Loading bus model...")
bus_model = YOLO(BUS_MODEL_PATH)

if torch.cuda.is_available():
    person_model.to("cuda")
    bus_model.to("cuda")

bus_class_names = bus_model.names

# ================= BUS MEMORY =================

bus_memory = defaultdict(lambda: {
    "route_hist": deque(maxlen=TEMPORAL_WINDOW),
    "dest_hist": deque(maxlen=TEMPORAL_WINDOW),
    "announced": False,
    "last_seen": 0,
    "frames_seen": 0,
    "ocr_routes": deque(maxlen=50),
    "ocr_dests_raw": deque(maxlen=50),
    "ocr_dests_corrected": deque(maxlen=50)
})

bus_final_info = {}

# tracks every unique bus front panel seen regardless of announcement
all_detected_bus_ids = set()

# route -> timestamp of last announcement, allows re-announcement after cooldown
# minimum 30s prevents side panel of same bus re-triggering while still in frame
ANNOUNCE_COOLDOWN_SECONDS = 15
announced_routes = {}  # route -> (last announced time, clean destination)

# ================= HELPERS =================

def majority_vote(seq):

    if not seq:
        return None, 0

    counts = {}

    for s in seq:
        if not s:
            continue

        counts[s] = counts.get(s, 0) + 1

    best = max(counts, key=counts.get)

    return best, counts[best]

def correct_destination(raw_text):

    if not raw_text:
        return None, raw_text

    original_raw = raw_text
    raw_text = raw_text.lower().replace(".", "").replace("'", "").strip()

    normalized_routes = [r.lower() for r in VALID_DESTINATIONS]

    # try both scorers and pick whichever gives the highest confidence match
    match_partial = process.extractOne(
        raw_text,
        normalized_routes,
        scorer=fuzz.partial_ratio
    )

    match_sort = process.extractOne(
        raw_text,
        normalized_routes,
        scorer=fuzz.token_sort_ratio
    )

    # pick the better scoring match across both scorers
    best_match = None
    best_score = 0
    best_index = None

    for m in [match_partial, match_sort]:
        if m and m[1] > best_score:
            best_score = m[1]
            best_match = m[0]
            best_index = m[2]

    if best_index is None:
        return original_raw, original_raw

    if best_score >= LEVENSHTEIN_THRESHOLD:
        return VALID_DESTINATIONS[best_index], original_raw

    return original_raw, original_raw

def build_announcement(route, dest):

    return f"Attention please. Bus number {route} to {dest} has arrived."

# ================= PIPELINE =================

print("[INFO] Starting pipeline...")

results = bus_model.track(
    source=VIDEO_SOURCE,
    stream=True,
    persist=True,
    conf=CONF_THRESH,
    imgsz=IMG_SIZE,
    verbose=False
)

frame_count = 0

for r in results:

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    frame = r.orig_img.copy()

    # ================= PERSON DETECTION =================

    person_results = person_model(frame, conf=0.5, verbose=False)

    waiting_persons = []

    if person_results:

        for pbox in person_results[0].boxes:

            px1, py1, px2, py2 = pbox.xyxy[0].cpu().numpy().astype(int)

            waiting_persons.append((px1, py1, px2, py2))

            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

    if len(waiting_persons) > 0:
        last_person_seen_time = time.time()

    person_recently_seen = (time.time() - last_person_seen_time) < PERSON_MEMORY_SECONDS

    # ================= BUS DETECTION =================

    bus_fronts = []
    route_boxes = []
    dest_boxes = []

    if r.boxes is not None:

        for box in r.boxes:

            cls_id = int(box.cls[0])
            cls_name = bus_class_names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if cls_name == "bus_front":
                bus_fronts.append(box)

            elif cls_name == "route_number":
                route_boxes.append(box)

            elif cls_name == "destination":
                dest_boxes.append(box)

    # ================= PROCESS EACH BUS =================

    for bus_box in bus_fronts:

        if bus_box.id is None:
            continue

        bus_id = int(bus_box.id[0])

        mem = bus_memory[bus_id]
        mem["frames_seen"] += 1

        # if this is a new ID but we already know its route from a previous ID,
        # inherit only route_hist (NOT dest_hist) from same-route buses seen recently
        # dest_hist is NOT merged — each bus reads its own destination fresh to prevent
        # destination bleeding between different buses (e.g. bus 5 Romford → bus 238 Barking)
        if mem["frames_seen"] == 1 and len(mem["route_hist"]) == 0:
            for prev_id, prev_mem in bus_memory.items():
                if prev_id == bus_id:
                    continue
                prev_route, prev_count = majority_vote(prev_mem["route_hist"])
                # get current bus's route from OCR if available
                current_route, _ = majority_vote(mem["route_hist"])
                if (prev_route
                        and prev_count >= STABLE_THRESHOLD
                        and not prev_mem["announced"]
                        and time.time() - prev_mem["last_seen"] < 15
                        and (current_route is None or current_route == prev_route)):
                    mem["route_hist"].extend(prev_mem["route_hist"])
                    mem["frames_seen"] += prev_mem["frames_seen"]
                    print(f"[MERGE] Bus {bus_id} inherited route history from Bus {prev_id} (route {prev_route})")
                    break

        has_ocr_evidence = len(mem["ocr_routes"]) >= 3 or len(mem["ocr_dests_raw"]) >= 3

        if mem["frames_seen"] >= MIN_FRAMES_TO_COUNT and has_ocr_evidence:
            all_detected_bus_ids.add(bus_id)


        mem["last_seen"] = time.time()

        bx1, by1, bx2, by2 = bus_box.xyxy[0].cpu().numpy().astype(int)

        cv2.putText(frame, f"Bus ID:{bus_id}", (bx1, by1-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        matched_route = None
        matched_dest = None

        for rb in route_boxes:

            rx1, _, rx2, _ = rb.xyxy[0].cpu().numpy().astype(int)

            if not (rx2 < bx1 or rx1 > bx2):
                matched_route = rb
                break

        for db in dest_boxes:

            dx1, _, dx2, _ = db.xyxy[0].cpu().numpy().astype(int)

            if not (dx2 < bx1 or dx1 > bx2):
                matched_dest = db
                break

        # ===== OCR ROUTE =====

        if matched_route:

            rx1, ry1, rx2, ry2 = matched_route.xyxy[0].cpu().numpy().astype(int)

            crop = frame[max(0, ry1):ry2, max(0, rx1):rx2]

            if crop.size > 0:

                text = run_ocr_on_image(crop)["text"]

                del crop

                if text:

                    mem["route_hist"].append(text)
                    mem["ocr_routes"].append(text)

        # ===== OCR DESTINATION =====

        if matched_dest:

            dx1, dy1, dx2, dy2 = matched_dest.xyxy[0].cpu().numpy().astype(int)

            crop = frame[max(0, dy1):dy2, max(0, dx1):dx2]

            if crop.size > 0:

                text = run_ocr_on_image(crop)["text"]

                del crop

                corrected, raw = correct_destination(text)

                if corrected:

                    mem["dest_hist"].append(corrected)
                    mem["ocr_dests_raw"].append(raw)          # before Levenshtein
                    mem["ocr_dests_corrected"].append(corrected)  # after Levenshtein

        # ===== TEMPORAL STABILIZATION =====

        route_final, route_count = majority_vote(mem["route_hist"])
        dest_final, dest_count = majority_vote(mem["dest_hist"])

        # inherit clean destination from first sighting of this route (fixes Barang→Barking)
        # but only within cooldown window — after cooldown, fresh destination is used
        if route_final and route_final in announced_routes:
            last_time, clean_dest = announced_routes[route_final]
            if (time.time() - last_time) < ANNOUNCE_COOLDOWN_SECONDS:
                # same bus still in cooldown — suppress and inherit clean dest
                dest_final = clean_dest
                if not mem["announced"]:
                    mem["announced"] = True
            # else: cooldown expired, new bus — use its own fresh OCR destination

        if (not mem["announced"]
            and route_final
            and dest_final
            and person_recently_seen
            and route_final in STOP_BUS_LIST.get(BUS_STOP_ID, [])):

            if route_count >= STABLE_THRESHOLD and dest_count >= STABLE_THRESHOLD:

                bus_width = bx2 - bx1
                bus_detection_widths[bus_id] = bus_width

                announcement = build_announcement(route_final, dest_final)

                print(f"[ANNOUNCEMENT] {announcement}")

                announce(announcement)

                mem["announced"] = True

                bus_final_info[bus_id] = (route_final, dest_final)
                announced_routes[route_final] = (time.time(), dest_final)  # store time + clean dest

    # ================= MEMORY MONITOR =================

    process_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    peak_cpu_mem = max(peak_cpu_mem, process_mem)

    gpu_mem = 0

    if torch.cuda.is_available():

        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)

        peak_gpu_mem = max(peak_gpu_mem, gpu_mem)

    if frame_count % 50 == 0:

        print(f"[MEMORY] Frame {frame_count} | CPU RAM: {process_mem:.2f} MB | GPU VRAM: {gpu_mem:.2f} MB")

        mem_log_frames.append(frame_count)
        mem_log_cpu.append(process_mem)
        mem_log_gpu.append(gpu_mem)

    # force garbage collection and CUDA cache flush every 200 frames
    # prevents pymalloc and PyTorch allocator from hoarding freed memory (critical for Jetson)
    if frame_count % 200 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================= DISPLAY =================

    if frame_count % DISPLAY_SKIP == 0:

        display_frame = cv2.resize(frame, DISPLAY_SIZE)

        cv2.imshow("Smart Bus Vision System", display_frame)

        del display_frame

    del frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= FINAL REPORT =================

print("\n========== MEMORY REPORT ==========")

avg_cpu_mem = sum(mem_log_cpu) / len(mem_log_cpu) if mem_log_cpu else 0
avg_gpu_mem = sum(mem_log_gpu) / len(mem_log_gpu) if mem_log_gpu else 0

print(f"Average CPU RAM Usage: {avg_cpu_mem:.2f} MB")
print(f"Average GPU VRAM Usage: {avg_gpu_mem:.2f} MB")

# ================= PERFORMANCE METRICS =================

print("\n========== BUS PERFORMANCE REPORT ==========")

total_detected  = len(all_detected_bus_ids)
total_announced = len(bus_final_info)
total_skipped   = total_detected - total_announced

print(f"\n  Total Bus Front Panels Detected : {total_detected}")
print(f"  Announced (matched stop route)  : {total_announced}")
print(f"  Not Announced (wrong stop/route): {total_skipped}")

announced_ids    = set(bus_final_info.keys())
not_announced_ids = all_detected_bus_ids - announced_ids

if announced_ids:
    print(f"  Announced Bus IDs               : {sorted(announced_ids)}")
if not_announced_ids:
    print(f"  Not Announced Bus IDs           : {sorted(not_announced_ids)}")

print()

for bus_id, bus_width in bus_detection_widths.items():

    route, dest = bus_final_info.get(bus_id, ("Unknown", "Unknown"))

    mem = bus_memory[bus_id]

    # --- Route OCR accuracy (no Levenshtein correction applied to routes) ---
    route_acc = 0.0

    if mem["ocr_routes"]:
        route_acc = sum(
            fuzz.ratio(txt, route)
            for txt in mem["ocr_routes"]
        ) / len(mem["ocr_routes"])

    # --- Destination OCR accuracy BEFORE Levenshtein correction ---
    dest_acc_before = 0.0

    if mem["ocr_dests_raw"]:
        dest_acc_before = sum(
            fuzz.ratio(txt, dest)
            for txt in mem["ocr_dests_raw"]
        ) / len(mem["ocr_dests_raw"])

    # --- Destination OCR accuracy AFTER Levenshtein correction ---
    dest_acc_after = 0.0

    if mem["ocr_dests_corrected"]:
        dest_acc_after = sum(
            fuzz.ratio(txt, dest)
            for txt in mem["ocr_dests_corrected"]
        ) / len(mem["ocr_dests_corrected"])

    distance = (REAL_BUS_WIDTH * FOCAL_LENGTH_PIXELS) / bus_width

    print(f"\nBus {bus_id} | Route: {route} | Destination: {dest}")
    print(f"  Route OCR Accuracy          : {route_acc:.2f}%")
    print(f"  Destination Accuracy Before Levenshtein : {dest_acc_before:.2f}%  (raw OCR vs final)")
    print(f"  Destination Accuracy AFTER  : {dest_acc_after:.2f}%  (post Levenshtein vs final)")
    print(f"  Levenshtein Improvement     : +{max(0.0, dest_acc_after - dest_acc_before):.2f}%")
    print(f"  Bus Width                   : {bus_width} px")
    print(f"  Estimated Distance          : {distance:.2f} meters")

cv2.destroyAllWindows()

# ================= MEMORY GRAPH =================

avg_cpu_mem = sum(mem_log_cpu) / len(mem_log_cpu) if mem_log_cpu else 0
avg_gpu_mem = sum(mem_log_gpu) / len(mem_log_gpu) if mem_log_gpu else 0

fig, (ax_cpu, ax_gpu) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax_cpu.plot(mem_log_frames, mem_log_cpu, color="#D85A30", linewidth=2, label="CPU RAM")
ax_cpu.axhline(y=avg_cpu_mem, color="#D85A30", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Avg: {avg_cpu_mem:.1f} MB")
ax_cpu.set_ylabel("CPU RAM (MB)")
ax_cpu.set_title("CPU RAM Usage Over Frames")
ax_cpu.legend(loc="upper left")
ax_cpu.grid(True, linestyle="--", alpha=0.3)

if torch.cuda.is_available() and any(v > 0 for v in mem_log_gpu):
    ax_gpu.plot(mem_log_frames, mem_log_gpu, color="#378ADD", linewidth=2, label="GPU VRAM")
    ax_gpu.axhline(y=avg_gpu_mem, color="#378ADD", linestyle="--", linewidth=1, alpha=0.5,
                   label=f"Avg: {avg_gpu_mem:.1f} MB")
    ax_gpu.set_ylabel("GPU VRAM (MB)")
    ax_gpu.set_title("GPU VRAM Usage Over Frames")
    ax_gpu.legend(loc="upper left")
    ax_gpu.grid(True, linestyle="--", alpha=0.3)
else:
    ax_gpu.text(0.5, 0.5, "GPU VRAM data not available",
                ha="center", va="center", transform=ax_gpu.transAxes,
                color="gray", fontsize=12)
    ax_gpu.set_title("GPU VRAM Usage Over Frames")

ax_gpu.set_xlabel("Frame")

plt.tight_layout()
plt.savefig("memory_usage.png", dpi=150)
print("\n[INFO] Memory graph saved to memory_usage.png")
plt.show()