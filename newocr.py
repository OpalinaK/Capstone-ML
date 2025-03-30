import cv2
import easyocr
from ultralytics import YOLO

model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

image_path = "image2.jpg"
image = cv2.imread(image_path)
directions = []
full_text_results = reader.readtext(image)

for (_, text, confidence) in full_text_results:
    if "Gate" in text:
        directions.append(f"Proceed to {text}")
    elif "Restroom" in text:
        directions.append("Restroom detected. Follow the direction.")

results = model(image)

relevant_classes = {
    'bathrooms': "Restroom detected. Follow the direction.",
    'airplane symbol': "Follow signs to the boarding area.",
    'left arrow': "Go Left",
    'right arrow': "Go Right",
    'up arrow': "Go Straight",
    'down arrow': "Proceed Downstairs",
    'thin left arrow': "Go Left",
    'thin right arrow': "Go Right",
    'thin up arrow': "Go Straight",
}


for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_label = model.names[class_id]
        if class_label in relevant_classes:
            directions.append(relevant_classes[class_label])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

if directions:
    for direction in directions:
        print(direction)
else:
    print("No relevant info detected.")

cv2.imshow("Detected Signs", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
