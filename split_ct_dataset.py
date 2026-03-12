import os
import shutil
import random

# Paths
SOURCE_DIR = "CT_Images"
DEST_DIR = "CT_Dataset"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

def split_class(class_name):
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(src_class_dir)
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLIT_RATIO["train"] * total)
    val_end = train_end + int(SPLIT_RATIO["val"] * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        dest_dir = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            src_path = os.path.join(src_class_dir, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.copy(src_path, dest_path)

    print(f"{class_name}: {len(splits['train'])} train, "
          f"{len(splits['val'])} val, {len(splits['test'])} test")

# Run for both classes
split_class("stone")
split_class("normal")

print("Dataset split completed successfully")
