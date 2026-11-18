````markdown
# COCO Crop Generation

Generate cropped animal images from TrapTagger COCO annotations.

## Goal

This script takes a COCO JSON file and local camera trap images, then crops animals using the bounding box annotations. The cropped images are saved into folders by species.

## Workflow

```text
COCO JSON
    ↓
Read annotation
    ↓
Find image
    ↓
Read bounding box
    ↓
Crop animal
    ↓
Save by species
````

## COCO File Structure

The COCO JSON contains three main sections:

* `images`: image metadata and file paths
* `annotations`: bounding boxes and species IDs
* `categories`: species ID to species name mapping

The sections are linked like this:

```text
images
   ↑ image_id
annotations
   ↓ category_id
categories
```

## Script

```text
src/coco_crops/crop_coco_annotations.py
```

## What the script does

1. Reads the COCO JSON file.
2. Matches annotations to images.
3. Matches category IDs to species names.
4. Converts normalized bounding boxes to pixel coordinates.
5. Crops animals from images.
6. Saves crops into species folders.

Example output:

```text
output_crops/
├── Samango_Monkey/
├── Waterbuck/
└── warthog/
```

## Libraries Used

* `argparse` – command-line arguments
* `json` – read COCO JSON
* `pathlib` – handle file paths
* `Pillow (PIL)` – open, crop, and save images

Install Pillow:

```bash
pip install pillow
```

## Bounding Boxes

Bounding boxes are stored as:

```python
[x, y, width, height]
```

Example:

```python
[0.6875, 0.5395, 0.1765, 0.2812]
```

The values are normalized between 0 and 1, so they must be converted to pixel coordinates before cropping.

```python
left = int(x * image_width)
top = int(y * image_height)
right = int((x + width) * image_width)
bottom = int((y + height) * image_height)
```

Crop:

```python
crop = img.crop((left, top, right, bottom))
```

## How to Run

From the repository root:

```bash
python src/coco_crops/crop_coco_annotations.py \
  --json "data/example_annotations.json" \
  --image-root "data/images" \
  --output "output_crops"
```

## Arguments

| Argument       | Description                               |
| -------------- | ----------------------------------------- |
| `--json`       | Path to the COCO JSON file                |
| `--image-root` | Folder containing camera trap images      |
| `--output`     | Folder where cropped images will be saved |

## Example Result

Using the sample dataset:

```text
Saved crops: 21
Skipped annotations with missing images: 8963
```

This is expected because the COCO file contains many annotations, while the example image folder contains only a small subset of the full dataset.

Species found:

* Samango Monkey
* Waterbuck
* Warthog

## Notes

* The script matches images using the camera site/run/filename portion of the path (e.g. `D03/D03_R1/06200021.JPG`).
* This helps when the JSON path and local folder path have different parent folders.
* Output images are organized by species name.

## Next Steps

* Test on larger image subsets.
* Check image matching

```
```
