from ultralytics import YOLO
import cv2

# Load YOLOv8 model trained on people (COCO person class)
model = YOLO("yolov8x.pt")  # you can use yolov8n.pt for faster, lighter inference

# Set tracking configuration
source = 0  # use 0 for webcam, or replace with "video.mp4"
tracker_config = "botsort.yaml"  # or "strongsort.yaml" for ReID-optimized tracker

# Run the tracker — detects only 'person' class
results = model.track(
    source=source,
    tracker=tracker_config,
    classes=[0],      # restrict to person class
    persist=True,     # keep IDs between frames
    stream=True,      # yields frames so we can handle OpenCV display
)

# Loop over frames
for frame_result in results:
    frame = frame_result.orig_img.copy()
    boxes = frame_result.boxes

    # Each box has .id, .xyxy, .conf, .cls
    if boxes.id is not None:
        for box, person_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"ID {int(person_id)}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    cv2.imshow("Person Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
