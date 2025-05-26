import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model
model = YOLO("best3.pt")
names = model.names

# Open the video file or webcam
cap = cv2.VideoCapture('5.mp4')  # Replace with 0 for webcam

# Get frame width, height, and FPS for saving video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 500))

    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # Show frame
    cv2.imshow("FRAME", frame)
    
    # Save the frame
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

