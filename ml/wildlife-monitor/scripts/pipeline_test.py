import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ultralytics import YOLO
from PIL import Image
import cv2

from src.models.efficientnet_classifier import EfficientNetClassifier
from src.datasets.transforms import get_val_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load MegaDetector ----
detector = YOLO("MDV6-yolov9-c.pt")  # adjust path if it's not in repo root

# ---- Load your trained classifier ----
CLASS_NAMES = [
    'american_crow', 'anna_hummingbird', 'black_phoebe', 'bushtit',
    'california_quail', 'california_scrub_jay', 'cat', 'dog',
    'eurasian_collared_dove', 'house_finch', 'mourning_dove',
    'northern_mockingbird', 'song_sparrow', 'squirrel', 'white_crowned_sparrow'
]

classifier = EfficientNetClassifier(num_classes=15).to(device)
classifier.load_state_dict(torch.load(
    "ml/wildlife-monitor/outputs/checkpoints/efficientnet_wildlife_best.pth",
    map_location=device
))
classifier.eval()

val_transform = get_val_transforms()

# ---- Run detection ----
IMAGE_PATH = r"C:\Users\dania\Pictures\catdog.jpg"
results = detector(IMAGE_PATH)

orig_img = results[0].orig_img  # BGR numpy array

for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    det_conf = float(box.conf[0])

    # Crop, convert BGR -> RGB, to PIL for the classifier's transforms
    crop_bgr = orig_img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    # Classify the crop
    input_tensor = val_transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    species = CLASS_NAMES[pred_idx.item()]
    display_conf = min(conf.item(), 0.99)  # never claim absolute certainty
    print(f"Detection {i}: MegaDetector conf={det_conf:.2f} "
          f"-> Classified as '{species}' (confidence={display_conf:.2f})")