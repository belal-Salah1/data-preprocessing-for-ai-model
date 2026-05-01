import cv2
import os
import numpy as np
import random

IMG_SIZE = 320

SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}


# ----------------------------
# 1. Preprocess image
# ----------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Denoise (important for smoke/cloud)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Enhance contrast (helps fire detection)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    # Normalize (0 → 1)
    img = img / 255.0

    return img


# ----------------------------
# 2. Save image
# ----------------------------
def save_image(img, output_path):
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)


# ----------------------------
# 3. Split dataset
# ----------------------------
def split_files(files):
    random.shuffle(files)

    n = len(files)
    train_end = int(n * SPLIT["train"])
    val_end = int(n * (SPLIT["train"] + SPLIT["val"]))

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }


# ----------------------------
# 4. Process full dataset
# ----------------------------
def process_dataset(input_dir, output_dir):

    classes = os.listdir(input_dir)

    for cls in classes:
        class_path = os.path.join(input_dir, cls)

        files = [f for f in os.listdir(class_path)
                 if f.endswith((".jpg", ".png", ".jpeg"))]

        split = split_files(files)

        for split_name in split:

            out_class_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(out_class_dir, exist_ok=True)

            for file in split[split_name]:

                in_path = os.path.join(class_path, file)
                out_path = os.path.join(out_class_dir, file)

                img = preprocess_image(in_path)

                if img is not None:
                    save_image(img, out_path)

    print("🔥 DONE: Dataset is ready!")


# ----------------------------
# RUN
# ----------------------------

if __name__=="__main__":
    process_dataset("dataset_raw", "dataset_clean")