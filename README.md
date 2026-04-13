# data-preprocessing-for-ai-model

Prepares a 4-class image dataset (`cloud`, `fire`, `normal`, `smoke`) for
training a wildfire detection model. Raw data comes from two sources, gets
extracted into `dataset_raw/`, then preprocessed and split into
`dataset_clean/`.

## Project layout

```
fire_project/
├── extract_data.py       # Pulls raw images out of the downloads
├── preprocess.py         # Resizes / denoises / contrast / splits
├── requirements.txt      # opencv-python, numpy
├── dataset_raw/          # Source images, one subdir per class
│   ├── cloud/            # filled by extract_data.py (1,080 JPEGs)
│   ├── fire/             # empty — add your own source images
│   ├── normal/           # empty — add your own source images
│   └── smoke/            # filled by extract_data.py (737 JPEGs)
└── dataset_clean/        # Output: train/val/test × class
    ├── train/  (70%)
    ├── val/    (20%)
    └── test/   (10%)
```

## Data sources

Both files are expected in `~/Downloads/` before running `extract_data.py`:

| File | Contents | Destination |
|---|---|---|
| `cloud.tar.gz` (extracted to `~/Downloads/cloud/`) | 1,080 JPEGs from a Mobotix camera | `dataset_raw/cloud/` |
| `Wildfire Smoke.v1-raw.tfrecord.zip` (extracted to `~/Downloads/Wildfire Smoke.v1-raw.tfrecord/`) | 3× `Smoke.tfrecord` (train/valid/test) from Roboflow | `dataset_raw/smoke/` |

`extract_data.py` reads the TFRecord container format directly with the
Python stdlib — it scans each record for JPEG `FFD8FF…FFD9` markers, so
**no TensorFlow or protobuf is required**.

## Setup

Ubuntu 24.04 ships Python 3 without `pip` and blocks `pip install` globally
(PEP 668). The simplest path is to use the apt packages:

```bash
sudo apt install -y python3-opencv python3-numpy
```

Or, if you prefer a venv:

```bash
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1) Populate dataset_raw/cloud and dataset_raw/smoke from ~/Downloads
python3 extract_data.py

# 2) Resize, denoise, contrast-boost, split into train/val/test
python3 preprocess.py
```

## Preprocessing pipeline (per image)

Implemented in `preprocess.py`:

1. **Load** with `cv2.imread` (files that fail to decode are skipped).
2. **Resize** to **224×224**.
3. **BGR → RGB** color reorder.
4. **Gaussian blur** (3×3, σ=0) — suppresses sensor noise.
5. **Contrast boost** — `pixel' = 1.2·pixel + 10`, clipped to 0–255.
6. **Normalize** to `[0, 1]` (`/ 255`).
7. **Save back** as JPEG (×255, `uint8`, RGB → BGR, `cv2.imwrite`).

### Dataset-level steps

- Each subdir of `dataset_raw/` is treated as a class label.
- Only `.jpg`, `.jpeg`, `.png` files are processed.
- Per class, files are shuffled and split **70% train / 20% val / 10% test**.
- Output tree: `dataset_clean/{train,val,test}/<class>/`.

### Caveats

- The shuffle is **not seeded** — re-running produces a different split.
  Add `random.seed(42)` at the top of `preprocess.py` for reproducibility.
- Images are re-encoded as JPEG on save, which adds a small compression loss
  on top of the resize.
- The `/ 255` normalization is undone before saving (on-disk files are
  standard `uint8` JPEGs). Re-normalize in your training pipeline.

## Packaging the cleaned dataset

```bash
tar -czf dataset_clean.tar.gz dataset_clean/
# or
zip -r dataset_clean.zip dataset_clean/
```

Move the archive via USB, cloud storage (Drive/Dropbox), `scp`, or GitHub.

To unpack on another machine:

```bash
tar -xzf dataset_clean.tar.gz
```

## Current dataset counts

| Class | train | val | test | total |
|---|---|---|---|---|
| cloud | 762 | 208 | 110 | 1,080 |
| smoke | 515 | 148 | 74  | 737 |
| fire | 0 | 0 | 0 | 0 |
| normal | 0 | 0 | 0 | 0 |

`fire/` and `normal/` will stay empty until raw source images are added to
`dataset_raw/fire/` and `dataset_raw/normal/`.