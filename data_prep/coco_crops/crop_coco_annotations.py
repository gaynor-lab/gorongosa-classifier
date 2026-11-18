import argparse
import csv
import json
import re
from pathlib import Path
from PIL import Image

SOURCE = "traptagger"

# Categories to drop entirely (per Shawn's exclusion list)
EXCLUDED_CATEGORIES = {
    "Primate",
    "Antelope",
    "Vehicles/Humans/Livestock",  # VHL
    "Small & Medium Mammals",
    "Mongoose",
    "Unknown",
}

# Classes to rename (first is from TrapTagger, second is desired name)
REMAPPED_CLASSES = {
    "bird": "bird_other",
    "blue duiker": "duiker_blue",
    "cape buffalo": "buffalo",
    "thicktailed bushbaby": "bushbaby",
    "common duiker": "duiker",
    "duiker_common": "duiker",
    "red duiker": "duiker_red",
    "honey badger": "honey_badger",
    "banded mongoose": "mongoose_banded",
    "egyptian or large grey mongoose": "mongoose_large_grey",
    "marsh mongoose": "mongoose_marsh",
    "slender mongoose": "mongoose_slender",
    "white-tailed mongoose": "mongoose_white_tailed",
    "bushy-tailed mongoose": "mongoose_bushy_tailed",
    "rat" : "rodent",
    "sable": "sable_antelope",
    "samango monkey": "samango_monkey",
    "vervet monkey": "vervet_monkey",
    "wild dog": "wild_dog",
}


# TrapTagger's "Nothing" tag = reviewed MegaDetector box with no animal present.
# Treated as the background/negative class rather than dropped.
# To exclude it instead, just add "Nothing" to EXCLUDED_CATEGORIES above and
# delete the BACKGROUND_CATEGORIES block below.
BACKGROUND_CATEGORIES = {"Nothing"}

# Site folders look like A08, B05, C06, D07 (one letter + 2-3 digits).
SITE_FOLDER_PATTERN = re.compile(r"^[A-Za-z]\d{2,3}$")


def get_match_key(path_string):
    """
    Match images using the site-code folder onward, e.g.
    A08/A08_R1/filename.JPG, so the JSON and local folder tree
    don't need identical parent folders.
    """
    parts = Path(path_string).parts
    for i, part in enumerate(parts):
        if SITE_FOLDER_PATTERN.match(part):
            return str(Path(*parts[i:]))
    return str(Path(*parts[-3:]))


def safe_folder_name(name):
    """Make species names safe for folder names."""
    return name.replace(" ", "_").replace("/", "_")


def main(json_path, image_root, output_root):
    json_path = Path(json_path)
    image_root = Path(image_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)

    image_map = {img["id"]: img for img in coco["images"]}
    category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    local_image_map = {}
    for image_path in image_root.rglob("*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            key = get_match_key(str(image_path))
            local_image_map[key] = image_path

    saved_count = 0
    excluded_count = 0
    missing_image_count = 0
    csv_rows = []

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in image_map:
            missing_image_count += 1
            continue

        image_info = image_map[image_id]
        category_id = annotation["category_id"]
        species = category_map.get(category_id, "unknown")

        if species in EXCLUDED_CATEGORIES:
            excluded_count += 1
            continue

        if species in BACKGROUND_CATEGORIES:
            species = "background"

        #re-map species names for consistency
        species = REMAPPED_CLASSES.get(species.lower(), species)
        
        json_image_key = get_match_key(image_info["file_name"])
        if json_image_key not in local_image_map:
            missing_image_count += 1
            continue

        image_path = local_image_map[json_image_key]
        species_folder = output_root / safe_folder_name(species)
        species_folder.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            x, y, w, h = annotation["bbox"]
            left = int(x * width)
            top = int(y * height)
            right = int((x + w) * width)
            bottom = int((y + h) * height)
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            crop = img.crop((left, top, right, bottom))
            # crop_name = f"{Path(image_info['file_name']).stem}_ann_{annotation['id']}.jpg"
            #change filename to match Sara's foramt (base_site_species) but need to handle potential duplicates
            crop_name = f"{Path(image_info['file_name']).stem}-ann{annotation['id']}_{image_info['location'].lower()}_{species}.jpg"
            crop_path = species_folder / crop_name
            crop.save(crop_path)

        csv_rows.append({
            "crop_path": str(crop_path),
            "species": species,
            "site": image_info["location"],
            "source": SOURCE,
        })
        saved_count += 1

    csv_path = output_root / "manifest_traptagger.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "species", "site", "source"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved crops: {saved_count}")
    print(f"Skipped (excluded category): {excluded_count}")
    print(f"Skipped (missing/unmatched image): {missing_image_count}")
    print(f"Manifest written to: {csv_path}")
    print(f"Output folder: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.json, args.image_root, args.output)
