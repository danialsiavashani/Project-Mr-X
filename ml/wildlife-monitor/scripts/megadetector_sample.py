from ultralytics import YOLO
import os

weights_path = "MDV6-yolov9-c.pt"
model = YOLO(weights_path)

IMAGE_PATH = r"C:\Users\dania\Pictures\yolo-test.png"  # squirrel/dog image
OUTPUT_DIR = "ml/wildlife-monitor/outputs/crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = model(IMAGE_PATH)

for result in results:
    boxes = result.boxes
    orig_img = result.orig_img  # the raw image as a numpy array, BGR format

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # Slice the image array — this IS the crop
        crop = orig_img[y1:y2, x1:x2]

        # Save it
        crop_filename = f"{OUTPUT_DIR}/crop_{i}_conf{conf:.2f}.jpg"
        import cv2
        cv2.imwrite(crop_filename, crop)
        print(f"Saved {crop_filename} — class {cls_id}, confidence {conf:.2f}")