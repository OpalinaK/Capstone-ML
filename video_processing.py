import cv2
import easyocr
import time
import queue
import sounddevice as sd
import vosk
import json
import pyttsx3
import threading
import re
import numpy as np
from ultralytics import YOLO
from word2number import w2n  

MODEL_PATH = "/Users/danielkim/Desktop/18500/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
tts_engine = pyttsx3.init()

tts_engine.setProperty("rate", 150)
tts_engine.setProperty("volume", 1.0)
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[min(6, len(voices)-1)].id)

q = queue.Queue()
final_destination = None
destination_lock = threading.Lock()

yolo_model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

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

def words_to_numbers(text):
    words = text.lower().split()
    converted_words = []
    
    for word in words:
        try:
            num = w2n.word_to_num(word)
            converted_words.append(str(num))
        except ValueError:
            converted_words.append(word)

    return " ".join(converted_words)

def extract_important(s):
    match = re.search(r'([A-Z]\d+)', s)
    return match.group(0) if match else None

def mimic_speech():
    global final_destination
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=2000, device=None,
                            dtype="int16", channels=1, callback=lambda indata, frames, time, status: q.put(bytes(indata))):
        print("TALK NOW")
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                
                if text:
                    print(f"Raw Recognized: {text}")
                    converted_text = words_to_numbers(text)
                    print(f"Converted: {converted_text}")

                    match = re.search(r"gate\s+([A-Z]\d+)", converted_text, re.IGNORECASE)
                    if match:
                        with destination_lock:
                            final_destination = match.group(1)
                            print(f"Final Destination Set: {final_destination}")
                            tts_engine.say(f"Destination set to {final_destination}")
                            tts_engine.runAndWait()
                        return

def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)

def process_video(video_path):
    global final_destination
    final_destination = "Gate A19"
    with destination_lock:
        while final_destination is None:
            print("Waiting for destination...")
            time.sleep(1)

    final_destination = extract_important(final_destination)
    if final_destination is None:
        print("Failed to extract valid gate information.")
        return

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        directions = []
        full_text_results = reader.readtext(frame)
        grouped_texts = []
        final_group = []
        destination_box = None

        horizontal_threshold = 250
        vertical_threshold = 30

        # Grouping Text Results
        for i, (bbox1, text1, conf1) in enumerate(full_text_results):
            (x1, y1), (x2, y2) = bbox1[0], bbox1[2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            merged = False
            for group in grouped_texts:
                for _, (gx1, gy1, gx2, gy2), _ in group:
                    horizontal_distance = abs(x1 - gx2)
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
            x1s, y1s, x2s, y2s = zip(*[(x1, y1, x2, y2) for _, (x1, y1, x2, y2), _ in group])
            gx1, gy1, gx2, gy2 = int(min(x1s)), int(min(y1s)), int(max(x2s)), int(max(y2s))
            grouped_text = " ".join([word[0] for word in group])
            final_group.append((grouped_text, (gx1, gy1, gx2, gy2)))

        with destination_lock:
            for text, box in final_group:
                if final_destination in text:
                    destination_box = box
                    print(f"Final Destination Box: {box}")
                    break

        results = yolo_model(frame)
        closest_direction = None
        min_distance = float('inf')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_label = yolo_model.names[class_id]

                if class_label in relevant_classes and destination_box:
                    distance = calculate_distance(destination_box, (x1, y1, x2, y2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_direction = relevant_classes[class_label]

        message = f"Closest Direction to {final_destination}: {closest_direction}" if closest_direction else "No relevant direction detected."
        print(message)
        tts_engine.say(message)
        tts_engine.runAndWait()

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = threading.Thread(target=mimic_speech)
    t2 = threading.Thread(target=process_video, args=("airport.mp4",))

    t1.start()
    t1.join()
    t2.start()
    t2.join()
