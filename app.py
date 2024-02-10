import torch 
from model import EmotionCNN
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Grayscale, Lambda
import torch.nn.functional as F
import pandas as pd
import cv2
import numpy as np
import altair as alt

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.set_page_config(
    page_title="Emotion Classification",
    page_icon=":sunglasses:",
    layout="wide"
)

model = EmotionCNN()
model.load_state_dict(torch.load('bs64_4s_final.pt'))
model.eval()

preprocess = transforms.Compose([
    Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1)),  
])


def preprocess_image(image):
    # Convert image to black and white
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bw = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        image_bw = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2RGB)
    else:
        image_bw = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Zoom in on the face (you may need to adjust parameters based on your model and image size)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_bw, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_image = image_bw[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (64, 64))
    else:
        return None

    face_image = Image.fromarray(face_image)
    input_tensor = preprocess(face_image)

    return input_tensor



def predict(image):
    input_tensor = preprocess_image(image)
    if input_tensor.shape[0] != 3:
        input_tensor = input_tensor[:3, :, :]
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)
    return probabilities.numpy().squeeze()

st.title('Classification of basic emotions according to facial expression')
st.markdown("""
This web application uses a Convolutional Neural Network (CNN) to classify basic emotions based on facial expressions. Upload an image, and the model will predict the emotions present in the image.
""")
st.write("------------")
uploaded_files = st.sidebar.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Provided colors
colors = ['#E64345', '#E48F1B', '#F7D027', '#6BA547', '#60CEED', '#619ED6', '#B77EA3']

if uploaded_files:
    
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)

        col1, col2, col3 = st.columns([1.5, 0.9, 3]) 
        
        with col1:
            st.write("");
            st.write("");
            st.write("");
            st.image(image, width=300)

        with col2:
            probabilities = predict(image)
            predicted_emotion = emotions[probabilities.argmax()]
            col2.subheader(f'Prediction: {predicted_emotion}')
            for emotion, prob in zip(emotions, probabilities):
                st.write(f'{emotion}: {round(prob * 100, 2)}%')

        with col3:
            df_altair = pd.DataFrame({'Emotion': emotions, 'Probability': probabilities, 'Color': colors})
            chart = alt.Chart(df_altair).mark_bar().encode(
                x='Emotion:N',
                y='Probability:Q',
                color=alt.Color('Emotion:N', scale=alt.Scale(domain=emotions, range=colors)),
                tooltip=['Emotion:N', 'Probability:Q']
            ).properties(
                width=700,
                height=500,
            )
            st.altair_chart(chart)
        st.write("------------")