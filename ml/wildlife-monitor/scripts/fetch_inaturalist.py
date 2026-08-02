import requests
import os
import time

SPECIES = {
    "squirrel": "Sciurus niger",
}

OUTPUT_ROOT = "ml/wildlife-monitor/data/raw/wildlife"
IMAGES_PER_SPECIES = 100
START_COUNT = 300


def get_place_id(place_name):
    resp = requests.get(
        "https://api.inaturalist.org/v1/places/autocomplete",
        params={"q": place_name}
    )
    results = resp.json()["results"]
    # Filter for an exact name match to avoid picking up similarly-named places
    # (e.g. "Baja California Sur" also matches a "California" search)
    for r in results:
        if r["name"] == place_name:
            return r["id"]
    # Fallback if no exact match found
    print(f"WARNING: no exact match for '{place_name}', using first result: {results[0]['display_name']}")
    return results[0]["id"]


def download_species(common_name, folder_name, place_id, limit, start_count):
    out_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    params = {
        "taxon_name": common_name,
        "place_id": place_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": 200,
        "order_by": "created_at",
        "order": "desc",
    }
    resp = requests.get("https://api.inaturalist.org/v1/observations", params=params)
    print(f"DEBUG: status={resp.status_code}, total_results={resp.json().get('total_results')}")
    results = resp.json().get("results", [])

    count = start_count
    saved = 0
    for obs in results:
        if saved >= limit:
            break
        for photo in obs.get("photos", []):
            if saved >= limit:
                break
            url = photo["url"].replace("square", "medium")
            img_data = requests.get(url).content
            filepath = os.path.join(out_dir, f"{folder_name}_{count:03d}.jpg")
            with open(filepath, "wb") as f:
                f.write(img_data)
            count += 1
            saved += 1
            time.sleep(0.3)

    print(f"{common_name}: saved {saved} new images (total on disk now includes {count} numbered files)")


if __name__ == "__main__":
    california_id = get_place_id("California")
    print(f"Using place_id={california_id} for California")
    for folder, name in SPECIES.items():
        download_species(name, folder, california_id, IMAGES_PER_SPECIES, START_COUNT)