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

        # Sorting the output in descending order and getting the indices
        _, indices = torch.sort(out, descending=True)
        
        # Get the top 5 items and their percentages
        top_labels = [labels[idx] for idx in indices[:5]]
        top_percentages = [percentage[idx].item() for idx in indices[:5]]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Creating a horizontal bar chart
        y_pos = np.arange(len(top_labels))  # Positions for each bar
        ax.barh(y_pos, top_percentages, align='center', color='skyblue')
        
        # Add labels and titles
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_labels)
        ax.set_xlabel('Percentage (%)')
        ax.set_title('Top 5 Items with Their Percentages')
        
        # Adding percentage labels on the bars
        for i, v in enumerate(top_percentages):
            ax.text(v + 1, i, f'{v}%', va='center', color='black')
        
        # Show plot
        plt.tight_layout()
        plt.show()
  




