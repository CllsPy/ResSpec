import torch
import streamlit as st
from PIL import Image
from torchvision import models
from torchvision import transforms
import pandas as pd


resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])


img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if img is not None:
        img = Image.open(img)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        resnet.eval()
        out = resnet(batch_t)
        
        with open('imagenet_classes.txt') as f:
                labels = [line.strip() for line in f.readlines()]
        
        #_, index = torch.max(out, 1)
        
        # percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        # answer = {labels[index[0]], percentage[index[0]].item()}
        # dic = {'label': labels[index[0]], '%': percentage[index[0]].item()}
        # data = pd.DataFrame(dic, index=[0])
        # data

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        _, indices = torch.sort(out, descending=True)
        [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


        # Sorting the output and getting the top 5 indices
        _, indices = torch.sort(out, descending=True)
        
        # Make sure indices is a 1D tensor (indices[0] assumes out is 2D)
        indices = indices[:5]  # Select only the top 5 indices
        
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



