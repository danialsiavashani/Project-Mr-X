import requests
import os
import time

# folder_name -> iNaturalist common name
SPECIES = {
    "house_finch": "House Finch",
    "california_scrub_jay": "California Scrub-Jay",
    "black_phoebe": "Black Phoebe",
    "eurasian_collared_dove": "Eurasian Collared-Dove",
    "mourning_dove": "Mourning Dove",
    "bushtit": "Bushtit",
    "california_quail": "California Quail",
}

OUTPUT_ROOT = "data/raw/birds"
IMAGES_PER_SPECIES = 50


def get_place_id(place_name):
    """Look up a place ID by name instead of guessing/hardcoding it."""
    resp = requests.get(
        "https://api.inaturalist.org/v1/places/autocomplete",
        params={"q": place_name}
    )
    return resp.json()["results"][0]["id"]


def download_species(common_name, folder_name, place_id, limit):
    out_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    params = {
        "taxon_name": common_name,
        "place_id": place_id,
        "quality_grade": "research",  # community-verified IDs only
        "photos": "true",
        "per_page": 200,
        "order_by": "created_at",
        "order": "desc",
    }
    resp = requests.get("https://api.inaturalist.org/v1/observations", params=params)
    results = resp.json().get("results", [])

    count = 0
    for obs in results:
        if count >= limit:
            break
        for photo in obs.get("photos", []):
            if count >= limit:
                break
            url = photo["url"].replace("square", "medium")  # higher-res version
            img_data = requests.get(url).content
            filepath = os.path.join(out_dir, f"{folder_name}_{count:03d}.jpg")
            with open(filepath, "wb") as f:
                f.write(img_data)
            count += 1
            time.sleep(0.3)  # be polite to the API, don't hammer it

    print(f"{common_name}: saved {count} images to {out_dir}")


if __name__ == "__main__":
    california_id = get_place_id("California")
    for folder, name in SPECIES.items():
        download_species(name, folder, california_id, IMAGES_PER_SPECIES)