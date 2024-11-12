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

with col1:
        st.markdown("Upload an imagem")
        img = st.file_uploader("File", type=["jpg", "png", "jpeg"])


if img is not None:
        img = Image.open(img)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        resnet.eval()
        out = resnet(batch_t)

with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
        

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        
        _, indices = torch.sort(out, descending=True)
        # [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
        
        
        # Sorting the output and getting the top 5 indices
        _, indices = torch.sort(out, descending=True)
        
        # If indices is 2D, we need to flatten it or select the correct part
        indices = indices.squeeze()  # Remove any extra dimensions (e.g., if indices is 2D)
        
        # Select only the top 5 indices (as 1D tensor)
        indices = indices[:5]
        
        # Getting the top 5 labels and percentages
        top_labels = [labels[idx.item()] for idx in indices]  # Use idx.item() to convert tensor to integer
        top_percentages = [percentage[idx].item() for idx in indices]
        
        # Plotting with matplotlib
        fig, ax = plt.subplots()
        ax.barh(top_labels, top_percentages, color='skyblue')
        ax.set_xlabel('Percentage')
        ax.set_title('Top 5 Predictions')
        
        # Display the plot in Streamlit
        st.pyplot(fig)


