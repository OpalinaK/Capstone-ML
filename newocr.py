import cv2
import easyocr
from ultralytics import YOLO

model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

image_path = "image3.jpg"
image = cv2.imread(image_path)

full_text_results = reader.readtext(image)

print("Text detected in the blue:")
for (bbox, text, confidence) in full_text_results:
    (x1, y1), (x2, y2) = bbox[0], bbox[2]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue Boxes
    cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    print(f"'{text}' (Confidence: {confidence:.2f})")

# Running YOLO
results = model(image)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Boxes

cv2.imshow("Detected Texts", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
