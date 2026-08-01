import os
import random
import shutil

SOURCE_ROOT = "ml/wildlife-monitor/data/raw/wildlife"
TRAIN_ROOT = "ml/wildlife-monitor/data/processed/train"
VALID_ROOT = "ml/wildlife-monitor/data/processed/valid"

VAL_SPLIT = 0.2
random.seed(42)  # reproducible split

classes = os.listdir(SOURCE_ROOT)

for cls in classes:
    src_dir = os.path.join(SOURCE_ROOT, cls)
    files = os.listdir(src_dir)
    random.shuffle(files)

    split_idx = int(len(files) * (1 - VAL_SPLIT))
    train_files = files[:split_idx]
    valid_files = files[split_idx:]

    train_dst = os.path.join(TRAIN_ROOT, cls)
    valid_dst = os.path.join(VALID_ROOT, cls)
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(valid_dst, exist_ok=True)

    for f in train_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(train_dst, f))
    for f in valid_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(valid_dst, f))

    print(f"{cls}: {len(train_files)} train, {len(valid_files)} valid")