import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import EmotionCNN

model = EmotionCNN()

model_path = os.path.join(os.getcwd(), r'C:\Users\nancy\OneDrive\Radna površina\projekt\RUSU_ProjektPy\our_model.pt_final.pt')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
# Function to make predictions
def predict_emotion(image):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)

    # Get predicted emotion
    _, predicted_class = torch.max(output, 1)
    emotion = emotions[predicted_class.item()]
    
    return emotion

# Streamlit app
st.title("Emotion Recognition App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make prediction
    prediction = predict_emotion(uploaded_file)

    # Display the predicted emotion
    st.write("Predicted Emotion:", prediction)
