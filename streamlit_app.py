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
        st.markdown("## Objective")
        st.markdown('''This app uses a pre-trained ResNet model to classify images. 
        Upload an image, and the model predicts the top 5 possible labels along with their confidence scores, 
        visualized in a bar chart. ''')
        
        st.markdown("## Summary")

        st.markdown(''' 
                - Upload an imagem of an animal
                - See your upload
                - Get your label
        
        ''')
        st.markdown("---")
        
        st.markdown(f' made by [CLL](https://github.com/CllsPy)')

with st.form('Image Classifier'):
        st.markdown("## 1. Upload an imagem")
        st.info(
                """Please upload an image of an animal 
                (in JPG, PNG, or JPEG format)  for classification.  
                The model will analyze the image and predict the most likely animal species.'""")
        
        img = st.file_uploader("", type=["jpg", "png", "jpeg"])
        st.form_submit_button('Submit Image')

        if not img:
                st.error('Input an image')
                st.stop()

        img = Image.open(img)
        st.markdown("## 2. Your Image")
        st.image(img, caption=".", use_column_width=True)
                
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        resnet.eval()
        out = resnet(batch_t)

        with open('imagenet_classes.txt') as f:
                labels = [line.strip() for line in f.readlines()]
                

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        
        indices = indices.squeeze()  # Remove any extra dimensions (e.g., if indices is 2D)
        
        indices = indices[:3]
        
        top_labels = [labels[idx.item()] for idx in indices]  # Use idx.item() to convert tensor to integer
        top_percentages = ([percentage[idx].item() for idx in indices])

        df = pd.DataFrame({'Labels':top_labels, 'Probability':top_percentages})
        # Plotting with matplotlib
        # fig, ax = plt.subplots()
        
        # ax.barh(top_labels, top_percentages, color='gray',)
        # ax.set_xlabel('Percentage')
        # ax.set_title('Top 3 Predictions')

        # st.markdown("## 3. Label for your image")
        # plt.grid(True)
        # st.pyplot(fig)
        df = df['Probability'].max(axis=0)
        df
                          


