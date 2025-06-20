import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained models
face_classifier = cv2.CascadeClassifier(r'C:\Users\pavan\Downloads\Emotion_Detection_CNN-main-20241215T070225Z-001\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\pavan\Downloads\Emotion_Detection_CNN-main-20241215T070225Z-001\Emotion_Detection_CNN-main\model.h5')
gender_classifier = load_model(r'C:\Users\pavan\Downloads\Emotion_Detection_CNN-main-20241215T070225Z-001\Emotion_Detection_CNN-main\Gender_model.h5')

# Emotion and gender labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection with adjusted parameters
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Emotion prediction preprocessing
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray_resized]) != 0:
            roi_gray_resized = roi_gray_resized.astype('float') / 255.0
            roi_gray_resized = img_to_array(roi_gray_resized)
            roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)

            # Emotion prediction
            emotion_prediction = emotion_classifier.predict(roi_gray_resized)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]

            # Gender prediction preprocessing
            roi_color_resized = cv2.resize(roi_color, (256, 256))  # Resize to 256x256 for the gender model
            roi_color_resized = roi_color_resized.astype('float') / 255.0
            roi_color_resized = np.expand_dims(roi_color_resized, axis=0)

            # Gender prediction
            gender_prediction = gender_classifier.predict(roi_color_resized)[0]
            gender_label = gender_labels[gender_prediction.argmax()]

            # Display emotion and gender labels
            label_position = (x, y - 10)
            cv2.putText(frame, f'{emotion_label}, {gender_label}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed frame with predictions
    cv2.imshow('Emotion and Gender Detector', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
