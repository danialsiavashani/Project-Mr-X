import os
import random

folder = "ml/wildlife-monitor/data/raw/squirrels/squirrel"
files = os.listdir(folder)
keep = set(random.sample(files, min(280, len(files))))

for f in files:
    if f not in keep:
        os.remove(os.path.join(folder, f))

print(f"Kept {len(keep)} images")
