import os
import cv2
import numpy as np
import torch
import threading
import queue
import time

from camera.camera_stream import CameraStream
from detectors.detection import Detector
from recognizers.recognition import Recognizer
from gallery.gallery_db import Gallery

frames_queue = queue.Queue(maxsize=10)
detections_queue = queue.Queue(maxsize=10)
faces_queue = queue.Queue(maxsize=10)
results_queue = queue.Queue(maxsize=10)

# --------------------- Threaded Pipeline Functions ---------------------
def camera_loop(camera):
    print("[Camera] Started")
    while True:
        frame = camera.read_frame()
        if frame is not None and not frames_queue.full():
            frames_queue.put(frame)
        time.sleep(1 / 10)

def detection_loop(detector):
    print("[Detection] Started")
    while True:
        frame = frames_queue.get()
        bboxes = detector.detect_faces(frame)
        detections_queue.put((frame, bboxes))

def extraction_loop():
    print("[Extraction] Started")
    while True:
        frame, bboxes = detections_queue.get()
        for bbox in bboxes:
            x, y, w, h = bbox
            face_img = frame[y:y+h, x:x+w]
            faces_queue.put((face_img, bbox))

def recognition_loop(recognizer, gallery):
    print("[Recognition] Started")
    while True:
        face_img, bbox = faces_queue.get()
        emb = recognizer.get_embedding(face_img)
        face_id, score = gallery.find_best_match(emb)
        result = {
            'id': face_id,
            'score': score,
            'bbox': bbox,
            'status': 'match' if face_id else 'unknown'
        }
        results_queue.put(result)

def result_loop():
    print("[ResultConsumer] Started")
    while True:
        result = results_queue.get()
        print("[RESULT]", result)

# --------------------- Entry Point ---------------------
if __name__ == '__main__':
    device  = torch.device(0)
    camera = CameraStream()
    detector = Detector()
    recognizer = Recognizer()
    gallery = Gallery()

    threads = [
        threading.Thread(target=camera_loop, args=(camera,), daemon=True),
        threading.Thread(target=detection_loop, args=(detector,), daemon=True),
        threading.Thread(target=extraction_loop, daemon=True),
        threading.Thread(target=recognition_loop, args=(recognizer, gallery), daemon=True),
        threading.Thread(target=result_loop, daemon=True)
    ]

    for thread in threads:
        thread.start()

    print("[System] All components started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[System] Stopping...")
        camera.release()