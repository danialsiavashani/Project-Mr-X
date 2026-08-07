import os
import time
import uuid
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

# Confirmed empirically via detector.names — do not assume, verify per weights file
MEGADETECTOR_ANIMAL_CLASS = 0

CONFIDENCE_THRESHOLD = 0.65
MIN_CROP_AREA_RATIO = 0.005
CROP_PADDING_RATIO = 0.12  # expand box by 12% on each side before classifying
WEIGHTS_PATH = "MDV6-yolov10-e.pt"
CHECKPOINT_PATH = "outputs/checkpoints/efficientnet_wildlife_best.pth"
CROPS_DIR = "outputs/crops"
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
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for fname in os.listdir(CROPS_DIR):
            fpath = os.path.join(CROPS_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                deleted += 1
        if deleted:
            print(f"Cleanup: removed {deleted} crop(s) older than {retention_days} days")

    @staticmethod
    def _clamp_box(x1, y1, x2, y2, frame_w, frame_h):
        """Ensures box coordinates are valid and inside the frame."""
        x1 = max(0, min(frame_w, x1))
        y1 = max(0, min(frame_h, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))
        return x1, y1, x2, y2

    @staticmethod
    def _pad_box(x1, y1, x2, y2, frame_w, frame_h, pad_ratio=CROP_PADDING_RATIO):
        """
        Expands the box by pad_ratio on each side before cropping for classification.
        Purpose: MegaDetector's tight box can clip a tail, head, or wing — giving the
        classifier a crop unlike the well-composed photos it trained on. A modest
        context margin keeps more of the animal (and a bit of surrounding context)
        in frame, closer to what the classifier actually learned from.
        """
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * pad_ratio)
        pad_y = int(box_h * pad_ratio)
        return WildlifePipeline._clamp_box(
            x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, frame_w, frame_h
        )

    def process_image(self, image_path):
        """
        Runs detection + classification on one image.
        Returns a list of dicts, one per detection:
          - "classified": species result (species is "unknown" if below CONFIDENCE_THRESHOLD)
          - "rejected_small_crop": box too small relative to frame (uses the UNPADDED box size)
          - "skipped_non_animal": MegaDetector flagged this as person/vehicle, never classified
        """
        results = self.detector(image_path, conf=0.1, imgsz=960, verbose=False)
        orig_img = results[0].orig_img
        frame_h, frame_w = orig_img.shape[:2]
        frame_area = frame_h * frame_w

        detections = []

        for i, box in enumerate(results[0].boxes):
            det_class = int(box.cls[0])
            det_conf = float(box.conf[0])

            if det_class != MEGADETECTOR_ANIMAL_CLASS:
                label = self.detector.names.get(det_class, f"class_{det_class}")
                detections.append({
                    "status": "skipped_non_animal",
                    "megadetector_class": label,
                    "detector_confidence": det_conf,
                })
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = self._clamp_box(x1, y1, x2, y2, frame_w, frame_h)

            if x2 <= x1 or y2 <= y1:
                continue  # invalid/degenerate box, skip silently

            # Area check uses the ORIGINAL box — padding is for crop quality,
            # not for redefining what counts as "too small to bother with"
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / frame_area

            if area_ratio < MIN_CROP_AREA_RATIO:
                detections.append({
                    "status": "rejected_small_crop",
                    "detector_confidence": det_conf,
                    "area_ratio": area_ratio,
                })
                continue

            px1, py1, px2, py2 = self._pad_box(x1, y1, x2, y2, frame_w, frame_h)

            crop_bgr = orig_img[py1:py2, px1:px2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probs, k=3, dim=1)

            top_predictions = [
                {"species": CLASS_NAMES[idx], "confidence": round(prob, 4)}
                for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist())
            ]

            raw_conf = top_predictions[0]["confidence"]
            species = top_predictions[0]["species"]
            display_conf = min(raw_conf, 0.99)  # never display/log absolute certainty — see note below

            if raw_conf < CONFIDENCE_THRESHOLD:
                species = "unknown"

            timestamp = int(time.time() * 1000)
            unique_id = uuid.uuid4().hex[:8]
            crop_filename = f"{CROPS_DIR}/crop_{timestamp}_{unique_id}_{species}.jpg"
            cv2.imwrite(crop_filename, crop_bgr)  # save the ORIGINAL (unpadded) crop for review/logging

            detections.append({
                "status": "classified",
                "species": species,
                "classifier_confidence": display_conf,
                "raw_confidence": raw_conf,
                "top_predictions": top_predictions,
                "detector_confidence": det_conf,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "crop_path": crop_filename,
            })

        return detections