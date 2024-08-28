import streamlit as st

def show():
    st.title("Welcome to the Smart Ad Screen Project :bookmark:")
    st.subheader("Introduction")
    st.write("""
    This application uses cutting-edge computer vision and machine learning techniques to create a smart advertising experience. 
    It dynamically displays targeted ads based on the age, gender, and dress style of individuals standing in front of the screen.
    """)

    st.subheader("How It Works")
    st.write("""
    - **Face Detection and Analysis:** The system detects and analyzes faces using advanced models from MediaPipe.
    - **Demographic Prediction:** It predicts age and gender from the facial data using pre-trained deep learning models.
    - **Fashion Style Classification:** The application identifies clothing styles using a Convolutional Neural Network (CNN).
    - **Ad Selection:** Based on the detected profile, it selects and displays a targeted advertisement in real-time.
    """)

    st.subheader("Project Goals")
    st.write("""
    - To revolutionize advertising by providing a personalized and interactive experience.
    - To showcase the practical applications of AI and computer vision in modern marketing strategies.
    - To create a real-time, adaptable advertising solution that enhances viewer engagement and satisfaction.
    """)
    
    st.subheader("Technology Stack")
    st.write("""
    - **MediaPipe:** For face detection and analysis.
    - **OpenCV:** For image processing and real-time video capture.
    - **TensorFlow and Keras:** For deep learning-based age, gender, and fashion prediction.
    - **scikit-learn:** For machine learning-based ad selection.
    """)
