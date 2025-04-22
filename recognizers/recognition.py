import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from inception_resnet_SE.cbam import CBAM

class Recognizer:
    def __init__(self, device):
        
        self.model = CBAM(50, feature_dim = 512, mode = 'ir_se')
       
        state_dict = torch.load(r'recognizers/weights/Iter_243000_net.ckpt', map_location=device)['net_state_dict']
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)
        self.model.eval()

        self.input_size = (112, 112)

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.device = device

    def get_embedding(self, img):
        img = self.transform(img).to(self.device)[None, :, :, :]
        embedding = None
        with torch.no_grad():
            embedding = self.model(img)
            embedding = embedding.detach().cpu().numpy()
        return embedding