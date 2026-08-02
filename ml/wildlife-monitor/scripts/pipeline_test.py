import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.detect_and_classify import WildlifePipeline

pipeline = WildlifePipeline()

IMAGE_PATH = r"C:\Users\dania\Pictures\catbird.jpg"
results = pipeline.process_image(IMAGE_PATH)

for r in results:
    print(r)