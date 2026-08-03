import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import cv2

from src.inference.detect_and_classify import WildlifePipeline

DETECTION_INTERVAL_SECONDS = 2  # how often to run full detection+classification
DEDUP_WINDOW_SECONDS = 30       # matches original project spec — same species within this window = same visit

pipeline = WildlifePipeline()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check that it's connected and not in use by another app.")

print("Webcam live test running. Press 'q' to quit.")

last_detection_time = 0
last_boxes = []

# Deduplication tracking: species -> last time we actually logged/printed it as "new"
last_logged = {}


def is_duplicate_visit(species, now):
    """
    Returns True if this species was already logged within DEDUP_WINDOW_SECONDS.
    Updates the tracker regardless, so the window keeps extending while the animal stays.
    """
    last_seen = last_logged.get(species)
    is_dup = last_seen is not None and (now - last_seen) < DEDUP_WINDOW_SECONDS
    last_logged[species] = now  # always refresh, so a lingering animal doesn't re-trigger right at the boundary
    return is_dup


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    now = time.time()
    if now - last_detection_time >= DETECTION_INTERVAL_SECONDS:
        last_detection_time = now
        detections = pipeline.process_image(frame)

        last_boxes = []
        for det in detections:
            if det["status"] != "classified":
                continue

            species = det["species"]
            x1, y1, x2, y2 = det["box"]
            label = f"{species} ({det['classifier_confidence']:.2f})"
            last_boxes.append((x1, y1, x2, y2, label))  # always draw, even if it's a duplicate visit

            if species == "unknown":
                continue  # don't dedup-track unknowns, nothing meaningful to deduplicate

            if is_duplicate_visit(species, now):
                print(f"(still present) {label}")
            else:
                print(f"NEW detection: {label}")

    for (x1, y1, x2, y2, label) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Wildlife Pipeline - Live Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")