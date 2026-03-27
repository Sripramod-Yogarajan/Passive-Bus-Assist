import cv2
import os
from ultralytics import YOLO

class BusDetector:
    def __init__(self, model_path, conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf = conf_thresh
        self.class_names = self.model.names

    def detect(self, img):
        results = self.model(img, conf=self.conf)
        return results

    def crop_boxes(self, img, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        crops = {
            "bus_front": [],
            "route_number": [],
            "destination": []
        }

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]

                if cls_name not in crops:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_path = os.path.join(
                    output_dir, f"{cls_name}_{len(crops[cls_name])}.jpg"
                )
                cv2.imwrite(crop_path, crop)

                crops[cls_name].append({
                    "path": crop_path,
                    "bbox": (x1, y1, x2, y2)
                })

        return crops
