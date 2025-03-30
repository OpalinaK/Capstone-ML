import cv2
import pytesseract
import time
from ultralytics import YOLO

# Load YOLO model and Tesseract OCR
# model = YOLO("runs/detect/train3/weights/best.pt")
model = YOLO("best.pt")
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Open video file or webcam
video_path = "airport.mp4"  # Change this to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Define relevant classes and their corresponding messages
relevant_classes = {
    'bathrooms': "Restroom detected. Follow the direction.",
    'left arrow': "Go Left",
    'right arrow': "Go Right",
    'up arrow': "Go Straight",
    'down arrow': "Proceed Downstairs",
    'thin left arrow': "Go Left",
    'thin right arrow': "Go Right",
    'thin up arrow': "Go Straight",
}

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# For saving the output video
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Store detected objects and text
previous_detections = set()
previous_texts = set()

frame_count = 0
ocr_interval = 5  # Run OCR every 5 frames to reduce lag

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    new_detections = set()
    new_texts = set()
    directions = []

    # Run OCR every 'ocr_interval' frames
    if frame_count % ocr_interval == 0:
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for OCR
        text_results = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        for i in range(len(text_results['text'])):
            text = text_results['text'][i].strip()
            confidence = float(text_results['conf'][i])

            if confidence > 50 and ("Gate" in text or "Restroom" in text):
                new_texts.add(text)
        ocr_time = time.time() - start_time
        print(f"OCR processing time: {ocr_time:.2f} seconds")

    # Apply YOLO for object detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_label = model.names[class_id]

            new_detections.add(class_label)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Process new detections and texts
    if new_detections != previous_detections or new_texts != previous_texts:
        for class_label in new_detections:
            if class_label in relevant_classes:
                directions.append(relevant_classes[class_label])

        for text in new_texts:
            if "Gate" in text:
                directions.append(f"Proceed to {text}")
            elif "Restroom" in text:
                directions.append("Restroom detected. Follow the direction.")

        # Update previous states
        previous_detections = new_detections
        previous_texts = new_texts

    # Overlay results on the frame
    y_offset = 30
    for direction in directions:
        cv2.putText(frame, direction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    # Display frame
    cv2.imshow("Detected Signs", frame)
    out.write(frame)

    # Adjust wait time for smooth processing
    wait_time = max(1, int(ocr_time * 1000)) if frame_count % ocr_interval == 0 else 1
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

    frame_count += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
