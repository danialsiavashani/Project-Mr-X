import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from src.camera.shared_camera import SharedCamera

cam = SharedCamera(source=0)
cam.start()

# Wait for the first real frame instead of assuming a fixed delay is enough
timeout = 5
start = time.time()
while cam.get_latest_frame() is None:
    if time.time() - start > timeout:
        raise RuntimeError(f"No frame received within {timeout}s — check camera connection.")
    time.sleep(0.1)

print(f"First frame ready after {time.time() - start:.2f}s")

for i in range(3):
    frame = cam.get_latest_frame()
    print(f"Frame {i}: shape={frame.shape if frame is not None else None}")
    time.sleep(0.5)

cam.stop()
print("Camera released cleanly.")