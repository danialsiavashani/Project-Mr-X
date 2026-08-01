import os
import random

for species in ["cat", "dog"]:
    folder = f"ml/wildlife-monitor/data/raw/catdog/{species}"
    files = os.listdir(folder)
    keep = set(random.sample(files, 300))  # keep 300 random images

    for f in files:
        if f not in keep:
            os.remove(os.path.join(folder, f))

print("Done downsampling.")