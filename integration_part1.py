import cv2
import time
import threading
from ultralytics import YOLO
import easyocr
import sounddevice as sd
import vosk
import json
import pyttsx3
import queue
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


q = queue.Queue()
final_destination = None
destination_lock = threading.Lock()


def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))


def words_to_numbers(text):
    words = text.lower().split()
    converted_words = []
    
    for i, word in enumerate(words):
        try:
            num = w2n.word_to_num(word)
            converted_words.append(str(num))
        except ValueError:
            converted_words.append(word)

    return " ".join(converted_words)

def get_destination():
    
    global final_destination
    MODEL_PATH = "/Users/danielkim/Desktop/18500/vosk-model-en-us-0.22"

    SAMPLE_RATE = 16000
    MODEL = vosk.Model(MODEL_PATH)
    recognizer = vosk.KaldiRecognizer(MODEL, SAMPLE_RATE)
    tts_engine = pyttsx3.init()

    tts_engine.setProperty("rate", 150)
    tts_engine.setProperty("volume", 1.0)
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[min(6, len(voices)-1)].id)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=2000, device=None,
                            dtype="int16", channels=1, callback=callback):
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

                    match = matching(converted_text)
                    if match != "":
                        with destination_lock:
                            final_destination = match
                            print(f"Final Destination Set: {final_destination}")
                            tts_engine.say(f"Destination set to {final_destination}, starting to render video")
                            tts_engine.runAndWait()
                            start_rendering()

def matching(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    if "gate" not in tokens:
        return ""
    index = tokens.index("gate")
    letter = ""
    for i in range(index + 1, len(tokens)):
        if re.fullmatch(r'[a-zA-Z]', tokens[i]):
            letter = tokens[i].upper()
            index = i
            break
        elif tokens[i] == "be":
            letter = "B"
            index = i
            break
        elif tokens[i] == "ii":
            letter = "E"
            index = i
            break

    digits = []
    for j in tokens[index + 1:]:
        try:
            number = w2n.word_to_num(j)
            digits.append(number)
        except:
            continue
    s = 0
    for i in digits:
        s += i
    s = str(s)

    return f"{letter}{''.join(s)}" if letter and digits else ""


def process_ocr(frame, frame_id, reader, ocr_interval):
    global ocr_results, ocr_confidence
    if frame_id % ocr_interval == 0:
        results = reader.readtext(frame)
        ocr_results = [(bbox, text, conf) for bbox, text, conf in results if conf >= 0.5]
        ocr_confidence = [conf for _, _, conf in ocr_results]



def start_rendering():
    model = YOLO("best.pt")
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

    irrelevant_labels = [
        'baggage claim', 'danger-electricity', 'emergency exit', 'restaurants', 
        'handicapped symbol', 'no trespassing'
    ]
    video_path = "IMG_7051.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))



    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    previous_detections = set()
    previous_texts = set()

    frame_count = 0
    ocr_interval = 5

    ocr_results = []
    ocr_confidence = []


    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        new_detections = set()
        new_texts = set()
        directions = []

        threading.Thread(target=process_ocr, args=(frame, frame_count, reader, ocr_interval)).start()

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_label = model.names[class_id]
                confidence = box.conf[0].item()

                if confidence > 0.3 and class_label not in irrelevant_labels:
                    new_detections.add(class_label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if new_detections != previous_detections or new_texts != previous_texts:
            for class_label in new_detections:
                if class_label in relevant_classes:
                    directions.append(relevant_classes[class_label])

            for text, conf in zip(ocr_results, ocr_confidence):
                bbox, text_value, _ = text
                if final_destination in text_value:
                    directions.append(f"Proceed to {text_value}")
                    print(f"Proceed to {text_value}")
                    tts_engine.say(f"Proceed to {text_value}")
                elif "Restroom" in text_value:
                    directions.append("Restroom detected. Follow the direction.")
                    print("Restroom detected. Follow the direction.")
                    tts_engine.say("Restroom detected. Follow the direction.")

                (x1, y1) = (int(bbox[0][0]), int(bbox[0][1]))
                (x2, y2) = (int(bbox[2][0]), int(bbox[2][1]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, text_value, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            previous_detections = new_detections
            previous_texts = new_texts

        y_offset = 30
        for direction in directions:
            cv2.putText(frame, direction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        curr_time = time.time()
        fps_display = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps_display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Detected Signs", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_destination()
