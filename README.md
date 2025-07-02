# Face Emotion Recognition

This project implements a deep learning model to classify human facial expressions into distinct emotions such as Angry, Happy, Sad, and others using TensorFlow and Keras. It includes image preprocessing, model training, and a Streamlit-based web application for real-time prediction.

## 🔍 Project Features

* Facial emotion classification using deep learning
* Trained on a large dataset of labeled emotion images
* Live emotion prediction via uploaded image
* Deployed using a user-friendly Streamlit interface
* Modular and easy-to-extend codebase

## 📁 Directory Structure

* `Human_Emotion_Classification.ipynb` – Jupyter notebook containing the core model training and testing
* `app.py` – Streamlit web app to perform real-time image-based emotion classification
* `.gitignore` – Configuration to exclude datasets, virtual environments, and checkpoints
* `requirements.txt` – Project dependencies
* `README.md` – Project documentation
* `Facial_Images/` – (ignored) Training dataset directory
* `models/` – (ignored) Trained model files

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

## 👨‍💻 Author

**Abeselom Tesfay Gebremariam**
Aspiring MSc student in AI, Machine Learning, and Data Science
