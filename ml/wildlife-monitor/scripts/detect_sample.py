from ultralytics import YOLO

# Loads pretrained YOLOv8n (nano — smallest/fastest variant)
# Downloads weights automatically on first run (~6MB)
model = YOLO("yolov8n.pt")

# Run inference on a single image
# Replace with a real path - any photo with a person, animal, or common object works for now
results = model(r"C:\Users\dania\Pictures\yolo-test.png")

# results is a list (one entry per image you passed in)
for result in results:
    print(result.boxes)       # raw bounding box data
    result.show()             # opens a window with boxes drawn on the image