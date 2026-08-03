import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

from src.inference.detect_and_classify import WildlifePipeline

API_BASE_URL = "http://127.0.0.1:8000"

# Your classifier's CLASS_NAMES use folder-style names (e.g. "american_crow"),
# but the species table uses display names (e.g. "American Crow").
# Build the mapping once, using the API itself as the source of truth.
SPECIES_NAME_MAP = {
    "american_crow": "American Crow",
    "anna_hummingbird": "Anna's Hummingbird",
    "black_phoebe": "Black Phoebe",
    "bushtit": "Bushtit",
    "california_quail": "California Quail",
    "california_scrub_jay": "California Scrub-Jay",
    "cat": "Cat",
    "dog": "Dog",
    "eurasian_collared_dove": "Eurasian Collared-Dove",
    "house_finch": "House Finch",
    "mourning_dove": "Mourning Dove",
    "northern_mockingbird": "Northern Mockingbird",
    "song_sparrow": "Song Sparrow",
    "squirrel": "Squirrel",
    "white_crowned_sparrow": "White-crowned Sparrow",
}


def get_species_id_lookup():
    """Fetch species from the API and build classifier-name -> species_id map."""
    response = requests.get(f"{API_BASE_URL}/species")
    response.raise_for_status()
    species_list = response.json()

    name_to_id = {s["name"]: s["id"] for s in species_list}

    lookup = {}
    for classifier_name, display_name in SPECIES_NAME_MAP.items():
        if display_name in name_to_id:
            lookup[classifier_name] = name_to_id[display_name]
        else:
            print(f"WARNING: '{display_name}' not found in species table")

    return lookup


def log_detection(detection, species_id_lookup):
    """POST one detection result to the API. Skips 'unknown' and rejected crops."""
    if detection["status"] != "classified":
        print(f"Skipping ({detection['status']})")
        return

    if detection["species"] == "unknown":
        print("Skipping (low-confidence unknown, not logged)")
        return

    species_id = species_id_lookup.get(detection["species"])
    if species_id is None:
        print(f"WARNING: no species_id found for '{detection['species']}', skipping")
        return

    payload = {
        "species_id": species_id,
        "confidence": detection["classifier_confidence"],
        "crop_path": detection["crop_path"],
        "detector_confidence": detection["detector_confidence"],
        "model_version": "MDV6-yolov10-e",
        "source": "backyard_cam_1",
    }

    response = requests.post(f"{API_BASE_URL}/detections", json=payload)
    if response.status_code == 200:
        print(f"Logged: {detection['species']} (confidence={detection['classifier_confidence']:.2f})")
    else:
        print(f"Failed to log detection: {response.status_code} {response.text}")


if __name__ == "__main__":
    species_id_lookup = get_species_id_lookup()

    pipeline = WildlifePipeline()

    IMAGE_PATH = r"C:\Users\dania\Pictures\bj.jpg"
    results = pipeline.process_image(IMAGE_PATH)

    for detection in results:
        log_detection(detection, species_id_lookup)