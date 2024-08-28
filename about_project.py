import streamlit as st

def show():
    st.title('About the Project')
    st.write("""
    ### Project Overview
    The Smart Ad Screen is an innovative approach to digital advertising that leverages state-of-the-art computer vision and machine learning technologies. 
    By analyzing the demographic and fashion characteristics of individuals in real-time, this project aims to deliver highly personalized and engaging advertisements.

    ### Technical Details
    - **Face Detection:** Utilizes MediaPipe's face detection model to identify and track faces in the video feed.
    - **Age and Gender Prediction:** Employs pre-trained deep learning models to estimate the age and gender of detected individuals.
    - **Fashion Style Recognition:** A CNN-based model (VGG16) classifies clothing styles into categories such as casual, formal, modern, and sportswear.
    - **Ad Selection Algorithm:** Integrates demographic and fashion data to select and display the most relevant advertisement from a curated set.

    ### Implementation Steps
    1. **Video Capture:** Captures real-time video from the webcam.
    2. **Face Detection:** Identifies faces in the video feed using MediaPipe.
    3. **Feature Extraction:** Extracts age, gender, and fashion features from the detected faces.
    4. **Ad Prediction:** Uses a machine learning model to select and display an ad based on the extracted features.

    """)
