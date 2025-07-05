import streamlit as st
import tensorflow as tf
from keras.models import load_model # type: ignore
import numpy as np
import os
import cv2

model = load_model(r'E:Projects\Python\Facial_Emotion_Recognition\models\Face_Emotion_Classification.keras')
emotions = [['angry'],
 ['disgust'],
 ['fear'],
 ['happy'],
 ['neutral'],
 ['sad'],
 ['surprise']]

st.header('Human Emotion Recognition')
image_path = st.text_input('Enter Image Path')

image = cv2.imread(image_path) [:,:,0]
image = cv2.resize(image, (48,48))
image = np.invert(np.array([image]))

output = np.argmax(model.predict(image))
outcome = emotions [output]
stn = 'Emotion in the Image is '+ str(outcome[0])
st.markdown(stn)

image_name = os.path.basename(image_path)
st.image('Google_Images/'+image_name, width = 300)