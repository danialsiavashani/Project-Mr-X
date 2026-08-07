import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import cv2
import requests

from src.camera.shared_camera import SharedCamera
from src.inference.detect_and_classify import WildlifePipeline

API_BASE_URL = "http://127.0.0.1:8000"
CAMERA_ID = "backyard_cam_1"
DETECTION_INTERVAL_SECONDS = 2
STREAM_PUSH_INTERVAL_SECONDS = 0.1
STATUS_CHECK_INTERVAL_SECONDS = 2  # how often to ask "is anyone watching?"

SPECIES_NAME_MAP = {
    "american_crow": "American Crow", "anna_hummingbird": "Anna's Hummingbird",
    "black_phoebe": "Black Phoebe", "bushtit": "Bushtit",
    "california_quail": "California Quail", "california_scrub_jay": "California Scrub-Jay",
    "cat": "Cat", "dog": "Dog", "eurasian_collared_dove": "Eurasian Collared-Dove",
    "house_finch": "House Finch", "mourning_dove": "Mourning Dove",
    "northern_mockingbird": "Northern Mockingbird", "song_sparrow": "Song Sparrow",
    "squirrel": "Squirrel", "white_crowned_sparrow": "White-crowned Sparrow",
}


def get_species_id_lookup():
    response = requests.get(f"{API_BASE_URL}/species", timeout=5)
    response.raise_for_status()
    name_to_id = {s["name"]: s["id"] for s in response.json()}
    return {c: name_to_id[d] for c, d in SPECIES_NAME_MAP.items() if d in name_to_id}


def log_detection(detection, species_id_lookup):
    if detection["status"] != "classified" or detection["species"] == "unknown":
        return
    species_id = species_id_lookup.get(detection["species"])
    if species_id is None:
        return
    payload = {
        "species_id": species_id,
        "confidence": detection["classifier_confidence"],
        "crop_path": detection["crop_path"],
        "detector_confidence": detection["detector_confidence"],
        "model_version": "MDV6-yolov10-e",
        "source": CAMERA_ID,
    }
    try:
        requests.post(f"{API_BASE_URL}/detections", json=payload, timeout=5)
    except requests.RequestException as e:
        print(f"Log failed: {e}")


def push_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return
    try:
        requests.post(
            f"{API_BASE_URL}/stream/push/{CAMERA_ID}",
            data=buffer.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=2,
        )
    except requests.RequestException:
        pass


def check_should_stream():
    try:
        resp = requests.get(f"{API_BASE_URL}/stream/should-stream/{CAMERA_ID}", timeout=2)
        return resp.json().get("active", False)
    except requests.RequestException:
        return False  # fail safe: don't push if we can't confirm anyone's watching


if __name__ == "__main__":
    species_id_lookup = get_species_id_lookup()
    pipeline = WildlifePipeline()

    cam = SharedCamera(source=0)
    cam.start()

    last_detection_time = 0
    last_stream_push = 0
    last_status_check = 0
    stream_active = False

    print("Running. Ctrl+C to stop.")
    try:
        while True:
            frame = cam.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            now = time.time()

            if now - last_status_check >= STATUS_CHECK_INTERVAL_SECONDS:
                stream_active = check_should_stream()
                last_status_check = now

            if stream_active and now - last_stream_push >= STREAM_PUSH_INTERVAL_SECONDS:
                push_frame(frame)
                last_stream_push = now

            if now - last_detection_time >= DETECTION_INTERVAL_SECONDS:
                last_detection_time = now
                for det in pipeline.process_image(frame):
                    log_detection(det, species_id_lookup)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()