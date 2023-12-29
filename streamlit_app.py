import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Učitajte model i postavke
model = torch.load(r"C:\Users\nancy\OneDrive\Radna površina\projekt\RUSU_ProjektPy\our_model.pt_final.pt", map_location=torch.device("cpu"))
model.eval()


# Definirajte transformacije za slike (moraju biti iste kao kod treniranja)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def predict_emotion(image):
    # Preprocessajte sliku
    img = transform(image).unsqueeze(0)

    # Dobijte predikciju iz modela
    with torch.no_grad():
        output = model(img)

    # Izvucite indeks najviše vjerojatne klase
    _, predicted = torch.max(output, 1)

    # Vratite predviđenu emociju
    return predicted.item()

def main():
    st.title("Facial Emotion Recognition")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Prikaz slike
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Predviđanje emocije
        emotion_index = predict_emotion(image)
        emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
        predicted_emotion = emotions[emotion_index]

        st.write(f"Predicted Emotion: {predicted_emotion}")

if __name__ == "__main__":
    main()
