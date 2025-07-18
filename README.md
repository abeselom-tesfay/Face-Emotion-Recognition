# Facial Expression Recognition using CNN

This project implements a deep learning model to classify human facial expressions into distinct emotions such as Angry, Happy, Sad, and others using TensorFlow and Keras. It includes image preprocessing, model training, and a Streamlit-based web application for real-time prediction.

## 🔍 Project Features

* Facial emotion classification using deep learning
* Trained on a large dataset of labeled emotion images
* Live emotion prediction via uploaded image
* Deployed using a user-friendly Streamlit interface
* Modular and easy-to-extend codebase

## 🧠 Recognized Facial Expressions

The following are predictions made by the trained CNN model on sample images:

| Expression | Input Image | Predicted Emotion |
|------------|-------------|-------------------|
| Happy      | ![](recognized_expressions/happy.png) | 😄 Happy |
| Sad        | ![](recognized_expressions/sad.png)   | 😢 Sad   |
| Angry      | ![](recognized_expressions/angry.png) | 😠 Angry |

> The model correctly classifies these facial expressions from real-world images.


## ✨ Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/abeselom-tesfay/Face-Emotion-Recognition.git
   cd Face-Emotion-Recognition
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Upload an image and see the predicted emotion.

## 📦 Dataset

The dataset used for training is not included in this repository due to size limitations.  
You can download it from [this Google Drive link](https://drive.google.com/file/d/14FJae0kO2hUztFr6BwjZCl2fqIMxNzJT/view) and place it in the `Facial_Images/` folder.


## ✅ Requirements

* Python 3.7+
* TensorFlow
* NumPy
* OpenCV (cv2)
* Streamlit
* Matplotlib

