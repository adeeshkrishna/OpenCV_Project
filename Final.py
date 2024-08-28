import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from joblib import load
from skimage.transform import resize
from joblib import load
import warnings
import os
warnings.filterwarnings('ignore')

# Initialize the webcam and MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Load the age and gender models
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Load the fashion analysis model
fashion_model = load_model("fashion_model_1.h5")

# Load the ad prediction model
ad_model = load("random_forest_model.pkl")

# Load the scaler
scaler = load("rf_scaler.pkl")

# Define the age, gender, dress and ad labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['male', 'female']
dress_list=['casual', 'formal','modern','sportswear']
ad_list=['casual wear','electronics','formal wear','luxury','sportswear','travel and leisure']

ads=os.listdir('Ads')
ads

advertisement=[]
for ad in ads:
    advert=cv2.imread('Ads'+"/"+ad)
    advertisement.append(advert)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MediaPipe
    results = face_detection.process(rgb_frame)

    # Initialize variables for selecting the best face
    largest_face = None
    largest_area = 0

    # Check if faces are detected
    if results.detections:
        for face in results.detections:
            bboxC = face.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw) + x
            h = int(bboxC.height * ih) + y

            # Ensure coordinates are within image boundaries
            x, y = max(0, x), max(0, y)
            w, h = min(iw, w), min(ih, h)

            # Calculate the area of the bounding box
            area = (w - x) * (h - y)

            # Update the largest face
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)

        # Process only the largest face
        if largest_face:
            x, y, w, h = largest_face

            # Crop the face from the original frame
            face_crop = frame[y:h, x:w]

            # Check if face_crop is valid and not empty
            if face_crop.size == 0:
                continue  # Skip if face_crop is empty

            # Ensure face_crop has correct dimensions
            if face_crop.shape[0] < 1 or face_crop.shape[1] < 1:
                continue  # Skip if face_crop has invalid dimensions

            # Run age and gender prediction using OpenCV
            age_blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            gender_blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            age_net.setInput(age_blob)
            gender_net.setInput(gender_blob)
            age_preds = age_net.forward()
            gender_preds = gender_net.forward()

            # Get the predicted age and gender
            age = age_list[age_preds[0].argmax()]

            if age in ['(0-2)', '(4-6)', '(8-12)']:
                continue
            elif age =='(15-20)':
                age_cat=0
            elif age =='(25-32)':
                age_cat=2
            elif age =='(38-43)':
                age_cat=4
            elif age =='(48-53)':
                age_cat=5
            
            gender = gender_list[gender_preds[0].argmax()]

            if gender =='male':
                gender_cat=1
            elif gender == "female": 
                gender_cat=0

            
            #dress category prediction

            d_image=resize(rgb_frame,(224,224))
            d_image=d_image.reshape(1,224,224,3)
            prediction=fashion_model.predict(d_image)
            ind=prediction.argmax(axis=1)
            dress=dress_list[ind[0]]

            if dress =='casual':
                dress_cat=0
            elif dress == "formal": 
                dress_cat=1
            elif dress == "modern": 
                dress_cat=2

            #Ad prediction

            ad_pred=ad_model.predict(scaler.transform([[gender_cat,dress_cat,age_cat]]))
            ad_val=ad_pred[0]
            ad_cat=ad_list[ad_val]


            if ad_cat =='casual wear':
                aad=0
            elif ad_cat =='electronics':
                aad=1
            elif ad_cat =='formal wear':
                aad=2
            elif ad_cat =='luxury':
                aad=3
            elif ad_cat =='sportswear':
                aad=4
            elif ad_cat =='travel and leisure':
                aad=5




            # Display the results
            cv2.putText(frame, f"Age: {age}, Gender: {gender}, Dress: {dress}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("SmartAd", advertisement[aad])
    # cv2.imshow("SmartAd", frame)
    print(f"Age: {age}, Gender: {gender}, Dress: {dress}, Ad: {ad_list[aad]}")

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()