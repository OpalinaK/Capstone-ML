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
from difflib import get_close_matches

q = queue.Queue()
final_destination = None
destination_lock = threading.Lock()
timers = {}

# Timer
def start_timer(name):
    timers[name] = time.time()

def stop_timer(name):
    if name in timers:
        elapsed = time.time() - timers[name]
        print(f"{name} took {elapsed:.4f} seconds")
    else:
        print(f"Timer {name} not started")

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))


# Vosk Implementation
def words_to_numbers(text):
    if "forty" in text:
        text = text.replace("forty", "40")
    elif "too" in text:
        text = text.replace("too", "2")
    elif "to" in text:
        text = text.replace("to", "2")
    elif "before" in text:
        text = text.replace("before", "be 4")
    elif "for" in text:
        text = text.replace("for", "4")      
    words = text.lower().split()
    converted_words = []
    
    for i, word in enumerate(words):
        try:
            num = w2n.word_to_num(word)
            converted_words.append(str(num))
        except ValueError:
            converted_words.append(word)

    return " ".join(converted_words)


def better_matching(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()

    # 1
    gate_candidates = ["gate", "gates", "grate", "grates", "cates", "kate", "date", "eight", "gay", "jake"]
    gate_word = None
    gate_index = -1

    for i, token in enumerate(tokens):
        if get_close_matches(token, gate_candidates, n=1, cutoff=0.7):
            gate_word = token
            gate_index = i
            break

    if gate_word is None:
        return ""
    
    #2
    letter = ""
    for i in range(gate_index + 1, len(tokens)):
        if re.fullmatch(r'[a-zA-Z]', tokens[i]):
            letter = tokens[i].upper()
            gate_index = i
            break
        elif tokens[i] in ("be", "bee"):
            letter = "B"
            gate_index = i
            break
        elif tokens[i] in ("see", "sea"):
            letter = "C"
            gate_index = i
            break
        elif tokens[i] in ("ii", "e", "ee", "he", "eat", "ie"):
            letter = "E"
            gate_index = i
            break

    #3
    digits = []
    for token in tokens[gate_index + 1:]:
        try:
            number = w2n.word_to_num(token)
            digits.append(number)
        except:
            continue

    if not letter or not digits:
        return ""

    gate_number = sum(digits)
    return f"{letter}{gate_number}"


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



def get_destination():
    global final_destination
    MODEL_PATH = "/Users/danielkim/Desktop/18500/vosk-model-small-en-us-0.15"
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
        print("I'm ready to assist you. Please say your destination when ready.")
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
                            tts_engine.say(f"Destination set to {final_destination}.")
                            tts_engine.runAndWait()
                            start_rendering()

def get_destination_testing():
    
    global final_destination
    final_destination = "B48"
    print(f"Final Destination Set: {final_destination}")
    start_rendering()

def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)


def sign_bounding(boxes, frame_width, frame_height, left_expansion=250, right_expansion=500, top_expansion=150, bottom_expansion=300):
    if not boxes:
        return []

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    expanded_x1 = max(0, x1 - left_expansion)
    expanded_y1 = max(0, y1 - top_expansion)
    expanded_x2 = min(frame_width, x2 + right_expansion)
    expanded_y2 = min(frame_height, y2 + bottom_expansion)

    return [(expanded_x1, expanded_y1, expanded_x2, expanded_y2)]


def locate_destination_box(full_text_results):
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
            return box

    return None


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
    persistent_direction = None
    direction_counter = 0
    last_spoken_direction = None
    stability = 20  # Number of consistent frames needed

    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 150)
    tts_engine.setProperty("volume", 1.0)
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[min(6, len(voices)-1)].id)

    model = YOLO("best.pt")
    reader = easyocr.Reader(['en'])

    relevant_classes = {
    'bathrooms': "Restroom detected. Please proceed in the indicated direction.",
    'left arrow': "Please turn left at the next opportunity.",
    'right arrow': "Please turn right at the next opportunity.",
    'up arrow': "Please continue moving straight ahead.",
    'down arrow': "Please proceed downstairs carefully.",
    'thin left arrow': "Please turn left at the next opportunity.",
    'thin right arrow': "Please turn right at the next opportunity.",
    'thin up arrow': "Please continue moving straight ahead.",
    }

    video_path = "IMG_7051.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    yolo_interval = 10
    ocr_interval = 10
    prev_time = time.time()
    destination_box = None
    destination_found = False

    held_min_distance = float('inf')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        directions = []
        detected_boxes = []
        closest_direction = None
        possible_direction = None
        min_distance = float('inf')

        if frame_count % yolo_interval == 0:
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    if confidence > 0.3:
                        detected_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # green box

        merged_regions = sign_bounding(detected_boxes, frame_width, frame_height)
        for x1, y1, x2, y2 in merged_regions:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # purple box
            cropped = frame[y1:y2, x1:x2]

            if frame_count % ocr_interval == 0 and not destination_found:
                full_text_results = reader.readtext(cropped)
                destination_box = locate_destination_box(full_text_results)

                for bbox, text, conf in full_text_results:
                    class_id = int(box.cls[0])
                    class_label = model.names[class_id]
                    (x1, y1), (x2, y2) = bbox[0], bbox[2]
                    if class_label in relevant_classes and destination_box:
                        distance = calculate_distance(destination_box, (x1, y1, x2, y2))
                        if distance < min_distance:
                            min_distance = distance
                            possible_direction = relevant_classes[class_label]

        if min_distance < held_min_distance:
            if held_min_distance != float('inf'):
                tts_engine.say("Correction, changing direction.")
            held_min_distance = min_distance
            closest_direction = possible_direction
            print(f"New direction selected: {closest_direction}")

        # Stability check
        if frame_count % ocr_interval == 0:
            if closest_direction is not None:
                if closest_direction == persistent_direction:
                    direction_counter += 1
                else:
                    persistent_direction = closest_direction
                    direction_counter = 1
        else:
            direction_counter += 1

        if persistent_direction and direction_counter >= stability:
            if persistent_direction != last_spoken_direction:
                tts_engine.say(persistent_direction)
                tts_engine.runAndWait()
                last_spoken_direction = persistent_direction

            directions.append(f"{persistent_direction}")
        elif not destination_found:
            directions.append("Searching for gate...")

        # Draw directions
        y_offset = 30
        for direction in directions:
            cv2.putText(frame, direction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        out.write(frame)
        cv2.imshow("Merged Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imshow('Last Frame', frame)
            cv2.waitKey(0)
            break
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            print("Paused. Press any key to continue...")
            cv2.imshow('Paused Frame', frame)
            cv2.waitKey(0)
            cv2.destroyWindow('Paused Frame')

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_destination()
