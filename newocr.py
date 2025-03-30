import cv2
import easyocr
from ultralytics import YOLO


model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

image_path = "image4.jpg"
image = cv2.imread(image_path)
directions = []
full_text_results = reader.readtext(image)

# Grouping
grouped_texts = []
final_group = []
horizontal_threshold = 250  # Adjust for horizontal proximity
vertical_threshold = 30  # Allow slight vertical alignment

for i, (bbox1, text1, conf1) in enumerate(full_text_results):
    (x1, y1), (x2, y2) = bbox1[0], bbox1[2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    merged = False
    for group in grouped_texts:
        for _, (gx1, gy1, gx2, gy2), _ in group:
            horizontal_distance = abs(x1 - gx2)  # Focus on right edge of previous box
            vertical_distance = abs((y1 + y2) / 2 - (gy1 + gy2) / 2)

            if horizontal_distance < horizontal_threshold and vertical_distance < vertical_threshold:
                group.append((text1, (x1, y1, x2, y2), conf1))
                merged = True
                break
        if merged:
            break
    if not merged:
        grouped_texts.append([(text1, (x1, y1, x2, y2), conf1)])

for group in grouped_texts:
    x1s, y1s, x2s, y2s = [], [], [], []
    for _, (x1, y1, x2, y2), _ in group:
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    gx1, gy1, gx2, gy2 = int(min(x1s)), int(min(y1s)), int(max(x2s)), int(max(y2s))
    cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)  # Red box for grouped text

    grouped_text = " ".join([word[0] for word in group])
    final_group.append(grouped_text)

for g in final_group:
    print(g)
    if 'Gates' not in g:
        print("remove")
        final_group.remove(g)

if len(final_group) == 0:
    print("No Gates Info")
    exit()



# Destination = 'A'
# for g in final_group:
#     if 'A' in g:






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
