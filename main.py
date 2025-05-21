import os
import sys
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

# --------------------- Queues ---------------------
frames_queue = queue.Queue(maxsize=100)
detections_queue = queue.Queue(maxsize=100)
faces_queue = queue.Queue(maxsize=100)
results_queue = queue.Queue(maxsize=100)

stop_event = threading.Event()

# --------------------- Threaded Pipeline Functions ---------------------
def camera_loop(camera):
    print("[Camera] Started")
    while not stop_event.is_set():
        frame = camera.read_frame()

        if frame is None:
            print("[Camera] No frame received. Exiting...")
            stop_event.set()
            camera.release()
            break

        try:
            frames_queue.put(frame, )
        except queue.Full:
            print("[Camera] Frame dropped due to full queue")

    print("[Camera] Stopped")

def detection_loop(detector):
    print("[Detection] Started")
    while not (stop_event.is_set() and frames_queue.empty()):
        try:
            frame = frames_queue.get()
        except queue.Empty:
            continue
        frame_id = 1
        bboxes = detector.detect_faces(frame)
        
        detection_json = {
            'frame': frame,
            'bboxes': bboxes,
            'frame_id': frame_id
        }

        detections_queue.put(detection_json)

    print("[Detection] Stopped")

def extraction_loop():
    print("[Extraction] Started")
    while not (stop_event.is_set() and detections_queue.empty()):
        try:
            dettection_q_data = detections_queue.get()
            bboxes = dettection_q_data['bboxes']
            frame = dettection_q_data['frame']
            frame_id = dettection_q_data['frame_id']
        except queue.Empty:
            continue
        
        dettection_q_data['faces'] = []
        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                x, y, w, h, = box[0], box[1], box[2], box[3]
                face_img = frame[y:h, x:w]
                # cv2.imwrite('abc/face_' + str(frame_id) + '.jpg', face_img)
                dettection_q_data['faces'].append(face_img)
        
        faces_queue.put(dettection_q_data)
    
    print("[Extraction] Stopped")

def recognition_loop(recognizer, gallery):
    print("[Recognition] Started")
    while not (stop_event.is_set() and detections_queue.empty()):
        try:
            # face_img, bbox = faces_queue.get()
            faces_q_data = faces_queue.get()
            bboxes = faces_q_data['bboxes']
            frame = faces_q_data['frame']
            frame_id = faces_q_data['frame_id']
            faces = None
            
            if 'faces' in faces_q_data:
                faces = faces_q_data['faces']

        except queue.Empty:
            continue
        
        if bboxes is None or len(bboxes) == 0:
            results_queue.put(faces_q_data)
            continue

        if faces is None or len(faces) == 0:
            results_queue.put(faces_q_data)
        else:
            emb_batch = recognizer.get_embedding_batch(faces)
            face_ids, scores = gallery.find_best_match_batch(emb_batch)
            faces_q_data['results_face_ids'] = face_ids
            faces_q_data['results_scores'] = scores
            results_queue.put(faces_q_data)
    
    print("[Recognition] Stopped")

def result_loop():
    print("[ResultConsumer] Started")
    while not (stop_event.is_set() and results_queue.empty()):
        try:
            # result = results_queue.get()
            result = results_queue.get()

            if 'results_face_ids' in result:
                print("[RESULT results_face_ids]", result['results_face_ids'])
                print("[RESULT results_scores]", result['results_scores'])

        except queue.Empty:
            continue

    print("[ResultConsumer] Stopped")

def video_writer_loop(output_path='output_annotated_video.avi', fps=20):
    print("[VideoWriter] Started")

    # Initialize video writer once frame dimensions are known
    writer_initialized = False
    writer = None

    while not (stop_event.is_set() and results_queue.empty()):
        try:
            result = results_queue.get(timeout=1)

            frame = result.get('frame')
            bboxes = result.get('bboxes', [])
            ids = result.get('results_face_ids', [])
            scores = result.get('results_scores', [])
            frame_id = result.get('frame_id', -1)

            if frame is None:   
                continue

            # Initialize writer with frame size
            if not writer_initialized:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                writer_initialized = True

            # Draw boxes and labels
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                label = f"{ids[i]}" if ids and ids[i] else "unknown"
                color = (0, 255, 0) if ids and ids[i] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (w, h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            writer.write(frame)

        except queue.Empty:
            continue

    if writer:
        writer.release()

    print("[VideoWriter] Stopped")

# --------------------- Entry Point ---------------------
if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    camera = CameraStream(r'facerec_video.mp4')
    detector = Detector(device)
    recognizer = Recognizer(device)
    gallery = Gallery(r'gallery_images', recognizer, device)

    threads = [
        threading.Thread(target=camera_loop, args=(camera,), daemon=True),
        threading.Thread(target=detection_loop, args=(detector,), daemon=True),
        threading.Thread(target=extraction_loop, daemon=True),
        threading.Thread(target=recognition_loop, args=(recognizer, gallery), daemon=True),
        # threading.Thread(target=result_loop, daemon=True),
        threading.Thread(target=video_writer_loop, daemon=True)
    ]

    st = time.time()

    for t in threads:
        t.start()
        time.sleep(0.5)  # small stagger

    try:
        while True:
            if stop_event.is_set():
                print("[System] Stop signal received.")
                break
            time.sleep(1)
            a = 1
    except KeyboardInterrupt:
        print("[System] KeyboardInterrupt received. Stopping...")
        stop_event.set()
    
    # Wait for threads to finish
    for t in threads:
        t.join()
    
    stop_event.set()


    print("[System] All components finished.")
    print('Total time taken:', time.time() - st)
    camera.release()