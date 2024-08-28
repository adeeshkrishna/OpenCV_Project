# SmartAd

SmartAd is an intelligent digital advertisement board that leverages computer vision and machine learning to display personalized ads in real-time. The project uses advanced technologies like OpenCV, MediaPipe, TensorFlow, and Scikit-learn to analyze the demographics and fashion style of individuals standing in front of the camera and select targeted advertisements based on their profiles.

# Table of Contents
- Project Overview
- Features
- Technical Details
- Future Enhancements

# Project Overview

The SmartAd project aims to revolutionize the advertising industry by providing real-time, personalized ad experiences. It utilizes a combination of deep learning models and computer vision techniques to detect faces, predict age and gender, classify fashion styles, and select the most appropriate advertisement to display.

# Features

- Real-Time Face Detection: Uses MediaPipe for accurate facial detection.
- Age and Gender Prediction: Employs pretrained Caffe models to predict the age and gender of individuals.
- Fashion Style Recognition: Utilizes a Convolutional Neural Network (CNN) with VGG16 architecture to classify clothing styles.
- Ad Display Algorithm: Implements a Random Forest model to select targeted ads based on demographic and fashion data.
- Streamlit Web Application: Provides an interactive user interface for real-time ad display.

# Technical Details

Packages Used

- OpenCV: For video capture and image processing.
- MediaPipe: For real-time face detection.
- TensorFlow and Keras: For loading and using deep learning models (age, gender, and fashion detection).
- Scikit-learn: For implementing the Random Forest model for ad selection.
- Streamlit: For creating the interactive web application.

Model Details

- Age and Gender Prediction: Uses pretrained Caffe models for predicting the age and gender of detected faces.
- Fashion Style Recognition: A VGG16-based CNN model is used to classify clothing styles into categories like casual, formal, modern, and sportswear.
- Ad Display Algorithm: A Random Forest model trained on demographic and fashion data is used to select the most suitable ad from a predefined set.

#Future Enhancements

- Emotion Detection: Incorporate emotion detection to further refine ad targeting.
- Enhanced Dataset: Expand the dataset to improve model accuracy across diverse demographics.
- Feedback Mechanism: Implement a real-time feedback system to measure ad effectiveness and improve targeting algorithms.
