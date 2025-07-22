import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load trained model
model = load_model('models/Face_Emotion_Classification.keras')

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Title
st.title("😊 Human Emotion Recognition from Face")
st.write("Choose one of the input methods below to detect the emotion.")

# ---------------------------
# Helper: Preprocess image
# ---------------------------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

# ---------------------------
# Predict and Display
# ---------------------------
def predict_emotion(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    predicted_label = emotions[np.argmax(prediction)]
    return predicted_label

# ---------------------------
# Webcam Transformer Class
# ---------------------------
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            label = predict_emotion(img)
            # Draw label on image
            cv2.putText(img, f'{label.upper()}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
        return img

# ---------------------------
# Streamlit Tabs for Options
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🖼️ Enter Image Path", "📷 Use Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", width=300)
        label = predict_emotion(image)
        st.success(f"**Predicted Emotion:** {label.capitalize()}")

with tab2:
    image_path = st.text_input("Enter local image path:")
    if image_path:
        try:
            image = cv2.imread(image_path)
            st.image(image, caption="Image from Path", width=300)
            label = predict_emotion(image)
            st.success(f"**Predicted Emotion:** {label.capitalize()}")
        except:
            st.error("⚠️ Could not load the image from the given path.")

with tab3:
    st.write("Allow camera access to analyze your real-time emotion.")
    webrtc_streamer(key="emotion", video_transformer_factory=EmotionDetector)
