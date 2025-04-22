import os
import cv2
import numpy as np
from PIL import Image
import torch


class Gallery:
    def __init__(self, gallery_path, recognizer, device):
        self.gallery_path = gallery_path
        self.recognizer = recognizer
        self.device = device
        self.embeddings = {}

        self.__generate_gallery_embeddings()
    

    def __generate_gallery_embeddings(self):
        for filename in os.listdir(self.gallery_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(self.gallery_path, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.recognizer.input_size)
                img = Image.fromarray(img)
                img = self.recognizer.transform(img)[None, :, :, :]
                img = img.to(self.device)

                with torch.no_grad():
                    embedding = self.recognition_model(img).detach().cpu().numpy()
                
                face_id = os.path.splitext(filename)[0]
                self.embeddings[face_id] = embedding

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