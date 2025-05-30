from __future__ import annotations
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,10))
    def forward(self,x): return self.head(self.features(x))

def preprocess(image:Image.Image)->torch.Tensor:
    gray=ImageOps.grayscale(image); arr=np.asarray(gray)
    if arr.mean()>127: gray=ImageOps.invert(gray)
    bbox=gray.getbbox()
    if not bbox: raise ValueError("Draw a digit before predicting.")
    crop=gray.crop(bbox); crop.thumbnail((20,20)); canvas=Image.new("L",(28,28)); canvas.paste(crop,((28-crop.width)//2,(28-crop.height)//2))
    return torch.tensor(np.asarray(canvas,dtype=np.float32)/255).unsqueeze(0).unsqueeze(0)

def predict(model,image):
    model.eval()
    with torch.inference_mode(): probabilities=torch.softmax(model(preprocess(image)),dim=1)[0].numpy()
    return int(probabilities.argmax()),probabilities

