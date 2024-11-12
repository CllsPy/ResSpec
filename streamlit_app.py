import torch
import streamlit as st
from PIL import Image
from torchvision import models
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt


resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

# sidebar
with st.sidebar:
        st.markdown("Title")

# cols
col1, col2, col3 = st.columns(3)
left, middle, right = st.columns(3, vertical_alignment="bottom")

with st.container(height=300):
        left.markdown("## 1. Upload an imagem")
        img = left.file_uploader("", type=["jpg", "png", "jpeg"])

if img is not None:
        img = Image.open(img)

        with st.container(height=500):
                left.markdown("## 2. Your Image")
                left.image(img, caption="Uploaded Image.", use_column_width=True)

        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        resnet.eval()
        out = resnet(batch_t)

        with open('imagenet_classes.txt') as f:
                labels = [line.strip() for line in f.readlines()]
                

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        
        indices = indices.squeeze()  # Remove any extra dimensions (e.g., if indices is 2D)
        
        indices = indices[:5]
        
        top_labels = [labels[idx.item()] for idx in indices]  # Use idx.item() to convert tensor to integer
        top_percentages = [percentage[idx].item() for idx in indices]
        
        # Plotting with matplotlib
        fig, ax = plt.subplots()
        ax.barh(top_labels, top_percentages, color='red')
        ax.set_xlabel('Percentage')
        ax.set_title('Top 5 Predictions')
        

        with st.container(height=300):
                right.markdown("## 3. Label for your image")
                plt.grid(True)
                right.pyplot(fig)


