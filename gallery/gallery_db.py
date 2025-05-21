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
        self.embeddings_value = []
        self.embeddings_key = []

        self.unknown_counter = 0
        self.unknown_embeddings = {}

        self.__generate_gallery_embeddings()
    

    def __generate_gallery_embeddings(self):
        for filename in os.listdir(self.gallery_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(self.gallery_path, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.recognizer.input_size)

                with torch.no_grad():
                    embedding = self.recognizer.get_embedding(img)
                    # embedding = self.normalize_embedding(embedding)
                    embedding = embedding / np.expand_dims(np.sqrt(np.sum(np.power(embedding, 2), 1)), 1)
                
                face_id = os.path.splitext(filename)[0]
                self.embeddings[face_id] = embedding
                self.embeddings_value.append(embedding)
                self.embeddings_key.append(face_id)
            
        self.embeddings_value = np.array(self.embeddings_value)
        self.embeddings_value = self.embeddings_value.squeeze()

        # self.mu = np.mean(self.embeddings_value, axis=0)
    
    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding / (norm + 1e-10)

    
    def find_best_match_batch(self, embeddings, threshold=0.25):
        best_scores = []
        best_ids = []

        for emb_i in range(len(embeddings)):
            embedding = embeddings[emb_i]
            embedding = embedding.reshape(1, 512)
            embedding = embedding / np.expand_dims(np.sqrt(np.sum(np.power(embedding, 2), 1)), 1)
            scores = np.sum(np.multiply(embedding, self.embeddings_value), 1)
            max_similarity = max(scores)
            
            unknown_keys = list(self.unknown_embeddings.keys())
            unknown_values = np.array(list(self.unknown_embeddings.values()))

            if unknown_keys is not None and len(unknown_keys) > 0:
                unknown_values = unknown_values.reshape(unknown_values.shape[0], 512)
                unknown_scores = np.sum(np.multiply(embedding, unknown_values), 1)
                max_unknown_similarity = max(unknown_scores)
            else:
                max_unknown_similarity = -1
                unknown_scores = None

            name = None
            print(max_similarity)
            if max_similarity >= 0.35:
                max_similarity_index = list(scores).index(max_similarity)
                name = os.path.basename(self.embeddings_key[max_similarity_index])
            elif max_unknown_similarity >= 0.50:
                name = unknown_keys[(list(unknown_scores).index(max_unknown_similarity))]
            else:
                name = 'unknown'
                self.unknown_counter += 1
                name = f'unknown_{self.unknown_counter}'
                self.unknown_embeddings[name] = embedding

            best_ids.append(name)
            best_scores.append(max_similarity)


        return best_ids, best_scores