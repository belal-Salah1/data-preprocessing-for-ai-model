"""
Prepare raw datasets for preprocess.py.

Inputs (already downloaded and extracted in ~/Downloads):
  - ~/Downloads/cloud/                               (1,080 JPEGs)
  - ~/Downloads/Wildfire Smoke.v1-raw.tfrecord/
        {train,valid,test}/Smoke.tfrecord

Outputs:
  - dataset_raw/cloud/    copied JPEGs
  - dataset_raw/smoke/    JPEGs extracted from the three TFRecord files

Uses only the Python standard library.
"""

import os
import shutil
import struct
from pathlib import Path

HOME = Path.home()
DOWNLOADS = HOME / "Downloads"
CLOUD_SRC = DOWNLOADS / "cloud"
SMOKE_SRC = DOWNLOADS / "Wildfire Smoke.v1-raw.tfrecord"

PROJECT = Path(__file__).resolve().parent
CLOUD_DST = PROJECT / "dataset_raw" / "cloud"
SMOKE_DST = PROJECT / "dataset_raw" / "smoke"


# ----------------------------
# 1. Copy cloud JPEGs
# ----------------------------
def copy_cloud():
    CLOUD_DST.mkdir(parents=True, exist_ok=True)
    files = [f for f in CLOUD_SRC.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    for f in files:
        shutil.copy2(f, CLOUD_DST / f.name)
    print(f"[cloud] copied {len(files)} files -> {CLOUD_DST}")


# ----------------------------
# 2. TFRecord parser (no tensorflow required)
#
# TFRecord layout (per record):
#   uint64 length                (little-endian)
#   uint32 masked_crc32(length)  (ignored here)
#   bytes  payload               (length bytes, a serialized tf.train.Example)
#   uint32 masked_crc32(payload) (ignored here)
#
# Each payload contains one encoded image. Roboflow exports use JPEG, so we
# locate the JPEG bytes inside the payload by scanning for the SOI/EOI markers
# (FF D8 FF ... FF D9). This sidesteps needing protobuf to parse the Example.
# ----------------------------
def iter_tfrecords(path):
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            fh.read(4)                       # length crc
            payload = fh.read(length)
            fh.read(4)                       # payload crc
            if len(payload) < length:
                return
            yield payload


def extract_jpeg(payload):
    start = payload.find(b"\xff\xd8\xff")
    if start < 0:
        return None
    # Find the LAST FFD9 after start; images may contain embedded thumbnails
    # with their own FFD9, so rfind gives the outer end-of-image marker.
    end = payload.rfind(b"\xff\xd9")
    if end < 0 or end <= start:
        return None
    return payload[start:end + 2]


def extract_smoke():
    SMOKE_DST.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "valid", "test"):
        tfr = SMOKE_SRC / split / "Smoke.tfrecord"
        if not tfr.exists():
            print(f"[smoke] missing: {tfr}")
            continue
        count = 0
        for idx, payload in enumerate(iter_tfrecords(tfr)):
            jpeg = extract_jpeg(payload)
            if jpeg is None:
                continue
            out = SMOKE_DST / f"smoke_{split}_{idx:05d}.jpg"
            with open(out, "wb") as fh:
                fh.write(jpeg)
            count += 1
        print(f"[smoke] {split}: extracted {count} images")
        total += count
    print(f"[smoke] total: {total} files -> {SMOKE_DST}")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    if not CLOUD_SRC.exists():
        raise SystemExit(f"Not found: {CLOUD_SRC}")
    if not SMOKE_SRC.exists():
        raise SystemExit(f"Not found: {SMOKE_SRC}")

    copy_cloud()
    extract_smoke()
    print("Done. Now run:  python3 preprocess.py")
