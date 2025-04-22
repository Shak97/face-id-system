import cv2
import numpy as np
import threading
import queue
import time

# --------------------- Queues for Each Stage ---------------------
frames_queue = queue.Queue(maxsize=10)
detections_queue = queue.Queue(maxsize=10)
faces_queue = queue.Queue(maxsize=10)
results_queue = queue.Queue(maxsize=10)

# --------------------- Camera Stream ---------------------
class CameraStream:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

# --------------------- Detector ---------------------
class Detector:
    def __init__(self):
        # self.model = YOLODetector()
        pass

    def detect_faces(self, frame):
        # return self.model.detect(frame)
        return [(0, 0, 100, 100)]  # Dummy bbox

# --------------------- Recognizer and Gallery ---------------------
class Recognizer:
    def __init__(self):
        # self.model = FaceRecognizer()
        pass

    def get_embedding(self, face_img):
        # return self.model.embed(face_img)
        return np.random.rand(128)

class Gallery:
    def __init__(self):
        self.embeddings = {}

    def find_best_match(self, embedding, threshold=0.6):
        best_score = -1
        best_id = None
        for face_id, ref_emb in self.embeddings.items():
            score = self.cosine_similarity(embedding, ref_emb)
            if score > best_score and score > threshold:
                best_score = score
                best_id = face_id
        return best_id, best_score

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
