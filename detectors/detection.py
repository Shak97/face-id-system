import torch
from .yolo_detector.face_detector import YoloDetector

class Detector:
    def __init__(self, device, target_size = 720, min_face = 90):
        self.device = device
        self.target_size = target_size
        self.min_face = min_face
        self.model = YoloDetector(target_size=self.target_size, device=self.device, min_face=self.min_face)
    
    def detect_faces(self, frame):
        dh, dw, c = frame.shape
        bboxes, points = self.model.predict(frame)

        res_boxes = []

        if bboxes is None or len(bboxes) == 0:
            return None
        
        for box in bboxes[0]:
            x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            area = ((x1+w)-x1) * ((y1 + h)-y1)
            if area >= self.min_face:
                res_boxes.append((x1, y1, w, h))
        
        return res_boxes