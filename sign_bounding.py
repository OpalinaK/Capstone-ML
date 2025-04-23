import cv2
import time
import threading
import queue
import json
import numpy as np
import sounddevice as sd
import pyttsx3
import vosk
from ultralytics import YOLO
from word2number import w2n
import re

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


def sign_bounding(boxes, frame_width, frame_height, left_expansion=200, right_expansion=400, top_expansion=100, bottom_expansion=300):
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

def start_rendering():
    model = YOLO("best.pt")

    video_path = "IMG_7051.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    yolo_interval = 1
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_boxes = []

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

        curr_time = time.time()
        fps_display = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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
