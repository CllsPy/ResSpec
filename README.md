# Animal Image Classifier 

## Overview
This application uses a pre-trained ResNet-101 model to classify animal images. Users can upload an image of an animal, and the model will predict the top 5 possible animal species along with their confidence scores. The results are displayed as a bar chart showing the predicted labels and their respective probabilities.

## Features
- **Image Upload**: Upload an image of an animal in JPG, PNG, or JPEG format.
- **Animal Classification**: The model classifies the uploaded image and provides the top 5 possible labels along with their confidence percentages.
- **Visualization**: A bar chart is displayed showing the top 5 predicted labels and their probabilities.

## How It Works
1. **Image Upload**: Upload an image of an animal using the file uploader.
2. **Model Prediction**: The uploaded image is processed and passed through a ResNet-101 model that has been pre-trained on the ImageNet dataset.
3. **Top 5 Predictions**: The model generates predictions, and the top 5 possible animal labels are shown with confidence scores.
4. **Results Visualization**: A horizontal bar chart is displayed to visualize the top predictions.

## Requirements
- Python 3.x
- Streamlit
- PyTorch
- PIL (Python Imaging Library)
- torchvision
- matplotlib
- pandas

## Installation

To set up this application on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/animal-image-classifier.git
   cd animal-image-classifier
   ```

2. **Install dependencies**:
   Make sure you have Python 3 installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Streamlit application with the following command:
   ```bash
   streamlit run app.py
   ```

4. **Open the app**:
   Once the app is running, open your browser and navigate to the address provided in the terminal (usually `http://localhost:8501`).

## Model Information
This application uses the **ResNet-101** model, a deep convolutional neural network that is pre-trained on the **ImageNet** dataset. It consists of 101 layers and is designed to handle complex image classification tasks. The model is fine-tuned for animal classification through its use of skip connections, which help overcome issues with very deep networks.

## Image Requirements
The image you upload should be of an animal. The model has been trained on general object categories, so while it is optimized for animal images, other objects might not yield accurate predictions.

## Contributing
Feel free to contribute to this project by submitting bug reports, feature requests, or pull requests. All contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Acknowledgments
- **ResNet** (Residual Networks) architecture developed by Microsoft Research.
- **Streamlit** for providing an easy-to-use interface for building web apps.
- **PyTorch** and **torchvision** for providing the pre-trained models and tools for deep learning.

