import os
import time
import urllib.request
import torch
from ultralytics import YOLO
from PIL import Image
import cv2

from src.models.efficientnet_classifier import EfficientNetClassifier
from src.datasets.transforms import get_val_transforms

CLASS_NAMES = [
    'american_crow', 'anna_hummingbird', 'black_phoebe', 'bushtit',
    'california_quail', 'california_scrub_jay', 'cat', 'dog',
    'eurasian_collared_dove', 'house_finch', 'mourning_dove',
    'northern_mockingbird', 'song_sparrow', 'squirrel', 'white_crowned_sparrow'
]

CONFIDENCE_THRESHOLD = 0.65
MIN_CROP_AREA_RATIO = 0.005
WEIGHTS_PATH = "MDV6-yolov10-e.pt"
CHECKPOINT_PATH = "ml/wildlife-monitor/outputs/checkpoints/efficientnet_wildlife_best.pth"
CROPS_DIR = "ml/wildlife-monitor/outputs/crops"
CROP_RETENTION_DAYS = 30


class WildlifePipeline:
    """
    Loads the detector and classifier once, then can be called repeatedly
    on individual frames/images — this is what a live camera loop or
    FastAPI endpoint will actually use.
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(WEIGHTS_PATH):
            urllib.request.urlretrieve(
                "https://zenodo.org/records/15398270/files/MDV6-yolov10-e.pt?download=1",
                WEIGHTS_PATH
            )
        self.detector = YOLO(WEIGHTS_PATH)

        self.classifier = EfficientNetClassifier(num_classes=len(CLASS_NAMES)).to(self.device)
        self.classifier.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
        self.classifier.eval()

        self.transform = get_val_transforms()

        os.makedirs(CROPS_DIR, exist_ok=True)
        self.cleanup_old_crops()

    def cleanup_old_crops(self, retention_days=CROP_RETENTION_DAYS):
        """
        Deletes crop files older than retention_days.
        Called once on pipeline startup — good enough for now.
        A real scheduled cleanup (cron/background task) comes later, in deployment.
        """
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for fname in os.listdir(CROPS_DIR):
            fpath = os.path.join(CROPS_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                deleted += 1
        if deleted:
            print(f"Cleanup: removed {deleted} crop(s) older than {retention_days} days")

    def process_image(self, image_path):
        """
        Runs detection + classification on one image.
        Returns a list of dicts, one per detection (classified, unknown, or rejected).
        Saves an image file to disk for every successfully classified detection.
        """
        results = self.detector(image_path, conf=0.1, imgsz=960, verbose=False)
        orig_img = results[0].orig_img
        frame_h, frame_w = orig_img.shape[:2]
        frame_area = frame_h * frame_w

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0])
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / frame_area

            if area_ratio < MIN_CROP_AREA_RATIO:
                detections.append({
                    "status": "rejected_small_crop",
                    "detector_confidence": det_conf,
                    "area_ratio": area_ratio,
                })
                continue

            crop_bgr = orig_img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

            species = CLASS_NAMES[pred_idx.item()]
            display_conf = min(conf.item(), 0.99)

            if display_conf < CONFIDENCE_THRESHOLD:
                species = "unknown"

            timestamp = int(time.time() * 1000)
            crop_filename = f"{CROPS_DIR}/crop_{timestamp}_{species}.jpg"
            cv2.imwrite(crop_filename, crop_bgr)

            detections.append({
                "status": "classified",
                "species": species,
                "classifier_confidence": display_conf,
                "detector_confidence": det_conf,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "crop_path": crop_filename,
            })

        return detections