import streamlit as st
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from joblib import load
from skimage.transform import resize
import numpy as np
import os

def show():
    st.title('SmartAd: Display Targeted Ads')
    st.write("Start your webcam to see the targeted ads in real-time based on your profile!")

    # Start and Stop buttons for webcam
    start_button = st.button('Start')
    stop_button = st.button('Stop')

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Load the models
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    fashion_model = load_model("fashion_model_1.h5")
    ad_model = load("random_forest_model.pkl")
    scaler = load("rf_scaler.pkl")

    # Define labels
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['male', 'female']
    dress_list = ['casual', 'formal', 'modern', 'sportswear']
    ad_list = ['casual wear', 'electronics', 'formal wear', 'luxury', 'sportswear', 'travel and leisure']

    # Load ads
    ads = [cv2.imread(os.path.join('Ads', ad)) for ad in os.listdir('Ads')]

    # Initialize session state attributes if not present
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    # Start or stop the webcam
    if start_button:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Failed to open webcam. Please check your webcam connection.")
            else:
                st.session_state.run = True
                st.write("Webcam started")
    
    if stop_button:
        st.session_state.run = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.write("Webcam stopped")

    # Process video feed
    if st.session_state.run and st.session_state.cap is not None and st.session_state.cap.isOpened():
        FRAME_WINDOW = st.empty()  # Placeholder for video frames
        AD_WINDOW = st.empty()  # Placeholder for advertisement images
        
        while st.session_state.run:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning("Failed to capture video frame. Please check your webcam.")
                break

            # Convert frame for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            largest_face = None
            largest_area = 0

            # Detect faces and process the largest one
            if results.detections:
                for face in results.detections:
                    bboxC = face.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw) + x
                    h = int(bboxC.height * ih) + y
                    x, y = max(0, x), max(0, y)
                    w, h = min(iw, w), min(ih, h)

                    area = (w - x) * (h - y)
                    if area > largest_area:
                        largest_area = area
                        largest_face = (x, y, w, h)

                if largest_face:
                    x, y, w, h = largest_face
                    face_crop = frame[y:h, x:w]

                    if face_crop.size == 0 or face_crop.shape[0] < 1 or face_crop.shape[1] < 1:
                        continue

                    age_blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
                    gender_blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
                    age_net.setInput(age_blob)
                    gender_net.setInput(gender_blob)
                    age_preds = age_net.forward()
                    gender_preds = gender_net.forward()

                    age = age_list[age_preds[0].argmax()]
                    gender = gender_list[gender_preds[0].argmax()]

                    if age in ['(0-2)', '(4-6)', '(8-12)']:
                        continue
                    elif age == '(15-20)':
                        age_cat = 0
                    elif age == '(25-32)':
                        age_cat = 2
                    elif age == '(38-43)':
                        age_cat = 4
                    elif age == '(48-53)':
                        age_cat = 5

                    gender_cat = 1 if gender == 'male' else 0

                    d_image = resize(rgb_frame, (224, 224))
                    d_image = d_image.reshape(1, 224, 224, 3)
                    prediction = fashion_model.predict(d_image)
                    ind = prediction.argmax(axis=1)
                    dress = dress_list[ind[0]]

                    dress_cat = {'casual': 0, 'formal': 1, 'modern': 2, 'sportswear': 3}[dress]

                    ad_pred = ad_model.predict(scaler.transform([[gender_cat, dress_cat, age_cat]]))
                    ad_val = ad_pred[0]
                    ad_cat = ad_list[ad_val]

                    ad_index = ad_list.index(ad_cat)

                    # Display video feed and ad 
                   
                    FRAME_WINDOW.image(frame, channels='BGR')
                    
                    AD_WINDOW.image(ads[ad_index], channels='BGR')
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources if stopped
    if not st.session_state.run and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        cv2.destroyAllWindows()
