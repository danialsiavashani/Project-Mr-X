import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from src.inference.detect_and_classify import WildlifePipeline

VIDEO_PATH = r"C:\Users\dania\Pictures\sq.mp4"  # update to your downloaded file
DETECTION_INTERVAL_SECONDS = 2
DEDUP_WINDOW_SECONDS = 30
DISPLAY_MAX_WIDTH = 900  # resize only for the preview window, not for detection

pipeline = WildlifePipeline()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
detection_interval_frames = max(1, int(fps * DETECTION_INTERVAL_SECONDS))

print(f"Video FPS: {fps:.1f}, running detection every {detection_interval_frames} frames "
      f"(~{DETECTION_INTERVAL_SECONDS}s of video time)")

last_boxes = []
last_logged = {}
frame_count = 0


def is_duplicate_visit(species, video_time_seconds):
    last_seen = last_logged.get(species)
    is_dup = last_seen is not None and (video_time_seconds - last_seen) < DEDUP_WINDOW_SECONDS
    last_logged[species] = video_time_seconds
    return is_dup


while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_count += 1
    video_time_seconds = frame_count / fps

    if frame_count % detection_interval_frames == 0:
        # Detection always runs on the ORIGINAL, full-resolution frame —
        # resizing here would shrink small/distant animals further, working
        # directly against the small-object recall improvements from days ago.
        detections = pipeline.process_image(frame)

        last_boxes = []
        for det in detections:
            if det["status"] != "classified":
                continue

            species = det["species"]
            x1, y1, x2, y2 = det["box"]
            label = f"{species} ({det['classifier_confidence']:.2f})"
            last_boxes.append((x1, y1, x2, y2, label))

            if species == "unknown":
                continue

            timestamp_str = f"{video_time_seconds:.1f}s"
            if is_duplicate_visit(species, video_time_seconds):
                print(f"[{timestamp_str}] (still present) {label}")
            else:
                print(f"[{timestamp_str}] NEW detection: {label}")

    # Draw boxes at ORIGINAL resolution first, so coordinates line up correctly
    for (x1, y1, x2, y2, label) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize ONLY the display copy, after boxes are already drawn
    display_frame = frame
    h, w = frame.shape[:2]
    if w > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / w
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("Wildlife Pipeline - Video Test", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")