from yolo_detector.face_detector import YoloDetector

class Detector:
    def __init__(self, device, target_size = 720, min_face = 90):
        self.device = device
        self.target_size = target_size
        self.min_face = min_face
        self.model = YoloDetector(target_size=self.target_size, device=self.device, min_face=self.min_face)

    def detect_faces(self, frame):
        dh, dw, c = frame.shape
        bboxes, points = self.model.predict(frame)

        sizes = []
        x, y, w, h = None, None, None, None
        if bboxes is None or len(bboxes) == 0:
            return None
        
        for i in bboxes[0]:
            x1, y1, x2, y2 = int(i[0]), int(i[1]), int(i[2]), int(i[3])
            sizes.append((x2-x1) * (y2-y1))
        
        if len(sizes) == 0:
            return None
        
        selected_bbox = bboxes[0][sizes.index(max(sizes))]

        x, y, w, h = int(selected_bbox[0]), int(selected_bbox[1]), int(selected_bbox[2]), int(selected_bbox[3])

        if x + 7 <= dw:
            x = x + 7
        
        if y + 7 <= dh:
            y = y + 7
        
        w = w - 7
        h = h - 7

        return [x, y, w, h]
        
