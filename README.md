# Gorongosa National Park 

## How New Images Are Processed

When you run training.py, the script always scans the image folder. If a cached file exists: (`resnet_training/full_df_filtered.csv`), the script will:

- Load the cached filtered dataframe

- Compare filenames against the current folder

- Detect new images not previously processed

- Run MegaDetector only on those new images

- Append passing images to the cached CSV

- Save the updated CSV

- This prevents reprocessing the entire dataset.


<b> First run </b>: If the cached CSV does not exist, MegaDetector runs on all images and a CSV is created

<b> Subsequent Runs </b>: Only new images are processed so it is much faster

📁 Expected Directory Structure

```text
kaitlyn_catalyst/
├── speciesnet/
│   ├── training.py
│   ├── inference.py
│   ├── splitting.py
│   ├── detector.py
│   ├── dataloader.py
│   └── utilities.py
│
├── images/
│   └── all_species_images/
│       ├── IMG_0001_{site}_{class}.jpg
│       ├── IMG_0002_{site}_{class}.jpg
│       └── ...
│
└── resnet_training/
    ├── full_df_filtered.csv
    ├── last_epoch_predictions_*.json
    └── last_model_state_resnet18_*.pkl
```

🧠 MegaDetector Threshold

Configured inside training.py:
`"megadetector_conf": 0.2`


## Typical values:

```text
Threshold	Behavior
0.1–0.2	Permissive (keeps more animals, more false positives)
0.3–0.4	Balanced
0.5–0.6	Strict (fewer false positives, may miss small animals)
```
Change this value if:
- too many empty images are kept → increase threshold
- animals are being missed → decrease threshold

🚀 Running Training

Activate environment:

`conda activate speciesnet`


Run training:

`python training.py`


If new images were added, only those will be processed by MegaDetector.

🧪 Running Inference

<b> Single image </b> :

`python inference.py --image path/to/image.jpg`


<b> Folder </b> :

`python inference.py --folder path/to/images`


<b> Folder + save CSV </b> :

`python inference.py --folder path/to/images --output preds.csv`


The script automatically loads the latest saved checkpoint.

💾 <b> Model Checkpoints </b> :

Saved in:

`resnet_training/last_model_state_resnet18_YYYYMMDD_HHMMSS.pkl`


<b> Checkpoint contains </b>:

- model_state

- optimizer_state

- class_names

- training config

🔁 <b> Resume Training (Optional) </b>

- Load the checkpoint

- Restore model + optimizer state

- Continue training loop

(Current script saves checkpoints but does not auto-resume — can be added.)

📊 <b> Outputs </b>

After training completes:

```text 
resnet_training/
├── full_df_filtered.csv
├── last_epoch_predictions_train.json
├── last_epoch_predictions_valid.json
├── last_epoch_predictions_holdout.json
└── last_model_state_resnet18_*.pkl
```

🧩 <b> Conceptual Flow </b>
```text
Raw Images
    ↓
build_df_from_folder()
    ↓
MegaDetector (incremental filtering)
    ↓
Train/Val/Holdout split
    ↓
ResNet18 training
    ↓
Checkpoint saved
    ↓
Inference
```

## Important Notes

If an image fails MegaDetector, it will not appear in full_df_filtered.csv. Adding new images does NOT overwrite previous results. To completely rebuild filtering, delete: `resnet_training/full_df_filtered.csv`, then rerun training.
