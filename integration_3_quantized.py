import cv2
import time
import threading
import easyocr
import sounddevice as sd
import vosk
import json
import pyttsx3
import queue
import re
import numpy as np
from word2number import w2n 
import onnxruntime as ort

q = queue.Queue()
final_destination = None
destination_lock = threading.Lock()
timers = {}

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

def get_destination_test():
    global final_destination
    final_destination = "A20"
    print(f"Final Destination Set: {final_destination}")
    start_rendering()

def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)

def sign_bounding(boxes, frame_width, frame_height, left_expansion=200, right_expansion=500, top_expansion=100, bottom_expansion=300):
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

def preprocess(frame, input_shape=(640, 640)):
    img = cv2.resize(frame, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, frame_shape, input_shape=(640, 640), conf_threshold=0.3):
    boxes = []
    scores = []
    class_ids = []

    output = outputs[0]
    if len(output.shape) == 3:
        output = output[0]

    for det in output:
        conf = det[4]
        if conf > conf_threshold:
            class_id = int(det[5])
            x, y, w, h = det[0], det[1], det[2], det[3]

            x1 = int((x - w / 2) * frame_shape[1] / input_shape[0])
            y1 = int((y - h / 2) * frame_shape[0] / input_shape[1])
            x2 = int((x + w / 2) * frame_shape[1] / input_shape[0])
            y2 = int((y + h / 2) * frame_shape[0] / input_shape[1])

            boxes.append((x1, y1, x2, y2))
            scores.append(conf)
            class_ids.append(class_id)
    return boxes

def start_rendering():
    persistent_direction = None
    onnx_session = ort.InferenceSession("best_quant.onnx")
    reader = easyocr.Reader(['en'])
    video_path = "IMG_7051.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_count = 0
    yolo_interval = 10
    ocr_interval = 10
    destination_box = None
    destination_found = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        directions = []
        detected_boxes = []
        closest_direction = None
        min_distance = float('inf')

        if frame_count % yolo_interval == 0:
            input_tensor = preprocess(frame)
            input_name = onnx_session.get_inputs()[0].name
            outputs = onnx_session.run(None, {input_name: input_tensor})
            detected_boxes = postprocess(outputs, frame.shape)
            for x1, y1, x2, y2 in detected_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        merged_regions = sign_bounding(detected_boxes, frame_width, frame_height)
        for x1, y1, x2, y2 in merged_regions:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cropped = frame[y1:y2, x1:x2]

            if frame_count % ocr_interval == 0 and not destination_found:
                full_text_results = reader.readtext(cropped)
                destination_box = locate_destination_box(full_text_results)
                for bbox, text, conf in full_text_results:
                    (tx1, ty1), (tx2, ty2) = bbox[0], bbox[2]
                    tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
                    cv2.rectangle(frame, (tx1 + x1, ty1 + y1), (tx2 + x1, ty2 + y1), (255, 0, 0), 2)
                    cv2.putText(frame, text, (tx1 + x1, ty1 + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                if destination_box is not None:
                    destination_found = True

        if destination_box:
            for bbox, text, conf in full_text_results:
                (tx1, ty1), (tx2, ty2) = bbox[0], bbox[2]
                tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
                distance = calculate_distance(destination_box, (tx1 + x1, ty1 + y1, tx2 + x1, ty2 + y1))
                if distance < min_distance:
                    min_distance = distance
                    closest_direction = "Destination Nearby"

        if closest_direction:
            persistent_direction = closest_direction
        if persistent_direction:
            directions.append(f"{persistent_direction}")
        elif not destination_found:
            directions.append("Searching for gate...")

        y_offset = 30
        for direction in directions:
            cv2.putText(frame, direction, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        out.write(frame)
        cv2.imshow("Merged Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_destination_test()
