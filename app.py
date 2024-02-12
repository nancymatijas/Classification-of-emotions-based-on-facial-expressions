import torch
from model import EmotionCNN
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda
import torch.nn.functional as F
import pandas as pd
import cv2
import numpy as np
import altair as alt
from mtcnn.mtcnn import MTCNN

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.set_page_config(
    page_title="Emotion Classification",
    page_icon=":sunglasses:",
    layout="wide"
)

model = EmotionCNN()
model.load_state_dict(torch.load('bs64_4s_final.pt'))
model.eval()

mtcnn_detector = MTCNN()

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1)), 
])

def preprocess_image(image):
    image_np = np.array(image)
    faces = mtcnn_detector.detect_faces(image_np)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face_image = image_np[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (64, 64))
    else:
        return None

    face_image = Image.fromarray(face_image)
    input_tensor = preprocess(face_image)

    return input_tensor

def predict(image):
    input_tensor = preprocess_image(image)
    if input_tensor is None:
        st.write("");
        st.write("");
        st.write("");
        st.write("");
        st.write("");
        st.warning("No face detected in the uploaded image.")
        return None
    
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
            if probabilities is not None:
                predicted_emotion = emotions[probabilities.argmax()]
                col2.subheader(f'Prediction: {predicted_emotion}')
                for emotion, prob in zip(emotions, probabilities):
                    st.write(f'{emotion}: {round(prob * 100, 2)}%')

        with col3:
            if probabilities is not None:
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
