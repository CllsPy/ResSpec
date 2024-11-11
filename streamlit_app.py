import torch
import streamlit as st

from torchvision import models
from torchvision import transforms

resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

from PIL import Image
img = Image.open('snake.jpg')
img
