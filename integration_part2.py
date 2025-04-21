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
import queue
import threading
import re
import numpy as np
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

# def get_destination():
    
#     global final_destination
#     final_destination = "B40"
#     print(f"Final Destination Set: {final_destination}")
#     start_rendering()


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
                            tts_engine.say(f"Destination set to {final_destination}, starting video")
                            start_rendering()


def locate_destination_box(frame, reader):
    full_text_results = reader.readtext(frame)
    grouped_texts = []
    final_group = []
    destination_box = None

    horizontal_threshold = 250
    vertical_threshold = 30

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

    for text, box in final_group:
        if decide_gate(final_destination, text):
            print(f"Destination found: {text} at {box}")
            return box

    return None

def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)


def decide_gate(final_destination, text):
    if final_destination in text:
        return True
    
    gate_match = re.match(r'([A-Z])(\d+)', final_destination)
    range_match = re.search(r'([A-Z])(\d+)-(?:[A-Z])?(\d+)', text)

    if gate_match and range_match:
        gate_letter, gate_number = gate_match.group(1), int(gate_match.group(2))
        range_letter, start_num, end_num = range_match.group(1), int(range_match.group(2)), int(range_match.group(3))

        if gate_letter == range_letter and start_num <= gate_number <= end_num:
            return True
    return False

def start_rendering():
    global final_destination
    if final_destination is None:
        print("failed")
        exit()

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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    ocr_interval = 5
    destination_box = None
    destination_found = False

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        directions = []

        if frame_count % ocr_interval == 0 and not destination_found:
            destination_box = locate_destination_box(frame, reader)
            if destination_box is not None:
                destination_found = True
                print("Gate found.")
            else:
                print("Still searching for gate...")

        results = model(frame)

        closest_direction = None
        min_distance = float('inf')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_label = model.names[class_id]
                confidence = box.conf[0].item()

                if confidence > 0.3 and class_label not in irrelevant_labels:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if class_label in relevant_classes and destination_box:
                        distance = calculate_distance(destination_box, (x1, y1, x2, y2))
                        if distance < min_distance:
                            min_distance = distance
                            closest_direction = relevant_classes[class_label]

        if closest_direction:
            directions.append(f"{closest_direction}")

        if not destination_found:
            directions.append("Searching for gate...")

        y_offset = 30
        for direction in directions:
            cv2.putText(frame, direction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        curr_time = time.time()
        fps_display = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps_display}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Detected Signs", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    if not destination_found:
        print("Gate not found during video.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_destination()
