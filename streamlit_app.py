import torch
import streamlit as st
from PIL import Image
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


img = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)
img = PIL.open(img)
if img is not None:
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        resnet.eval()
        out = resnet(batch_t)
        
        with open('imagenet_classes.txt') as f:
                labels = [line.strip() for line in f.readlines()]
        
        _, index = torch.max(out, 1)
        
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        labels[index[0]], percentage[index[0]].item()
