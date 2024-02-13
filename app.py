import torch
from model import EmotionCNN
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda
import torch.nn.functional as F
import pandas as pd
import numpy as np
import altair as alt
import cv2
from mtcnn.mtcnn import MTCNN
import requests
from io import BytesIO
import base64

# API URL setup
API_BASE_URL = 'https://emotion-prediction-guzs.onrender.com'  # Adjust as needed

st.set_page_config(
    page_title="Emotion Classification",
    page_icon=":sunglasses:",
    layout="wide"
)

# Login Section
def user_login(username, password):
    response = requests.post(f"{API_BASE_URL}/api/users/login", json={"username": username, "password": password})
    if response.status_code == 200:
        # Assuming the response includes a token
        st.session_state['token'] = response.json()['token']
        st.session_state['is_logged_in'] = True
        st.session_state['username'] = username
        st.success("Successfully logged in.")
    else:
        st.error("Login failed. Check your credentials.")

def user_registration(username, password):
    response = requests.post(f"{API_BASE_URL}/api/users/register", json={"username": username, "password": password})
    if response.status_code == 200:
        st.success("Successfully registered. You can now login.")
    else:
        st.error("Registration failed. Please try a different username.")

def post_prediction(image, prediction, percentage):
    if st.session_state.get('is_logged_in', False):

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Prepare the payload according to your API schema
        payload = {
            "image": image_data,  # Assuming 'image' expects a base64 string
            "prediction": prediction,
            "predictionPercentage": percentage
        }
        
        # Send the POST request with the token for authorization
        headers = {
            "Authorization": f"Bearer {st.session_state['token']}",
            "Content-Type": "application/json"
        }
        
        # API endpoint for saving predictions
        response = requests.post(f"{API_BASE_URL}/api/predictions", json=payload, headers=headers)
        
        if response.status_code == 200:
            st.success("Prediction saved successfully.")
        else:
            st.error("Failed to save prediction.")

if st.session_state.get('is_logged_in', False):
    st.sidebar.write(f"You are logged in as {st.session_state['username']}")

# Add a toggle between Login and Registration
if not st.session_state.get('is_logged_in', False):
    action = st.sidebar.radio("Action", ["Login", "Register"])

    if action == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user_login(username, password)
    elif action == "Register":
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Register"):
            user_registration(new_username, new_password)

# Button to fetch predictions
if st.session_state.get('is_logged_in', False):
    if st.button('Show My Predictions'):
        # Make a GET request to fetch predictions
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        response = requests.get(f"{API_BASE_URL}/api/predictions", headers=headers)
        if response.status_code == 200:
            predictions = response.json()
            # Display predictions in a new screen or section
            # Create columns for the images
            # Create a container for each row of predictions
            for index, prediction in enumerate(predictions):
                # Start a new row every 3 predictions
                if index % 3 == 0:
                    cols = st.columns(3)  # Create a new row of columns
                col = cols[index % 3]  # Select the appropriate column
                
                # Decode the base64 image
                base64_image = prediction['image']
                image_data = base64.b64decode(base64_image)
                image = Image.open(BytesIO(image_data))

                # Display the image with the prediction and percentage
                col.image(image, caption=f"{prediction['prediction']} ({prediction['predictionPercentage']}%)", width=200)
        else:
            st.error("Failed to fetch predictions.")

def logout():
    for key in ['is_logged_in', 'token']:
        if key in st.session_state:
            del st.session_state[key]
    # Use st.experimental_rerun() to refresh the page and the state after logging out
    st.experimental_rerun()

# Add a logout button in the sidebar if the user is logged in
if st.session_state.get('is_logged_in', False):
    if st.sidebar.button('Logout'):
        logout()


emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


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
                prediction_percentage = round(probabilities.max() * 100, 2)  # Get the max prediction percentage
                col2.subheader(f'Prediction: {predicted_emotion}')
                for emotion, prob in zip(emotions, probabilities):
                    st.write(f'{emotion}: {round(prob * 100, 2)}%')

                post_prediction(image, predicted_emotion, prediction_percentage)

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
