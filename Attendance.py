import time

import cv2
import numpy as np
import os
import pickle
import face_recognition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from attendance_logger import AttendanceLogger

class FaceRecognizer:
    def __init__(self, names_path='data/names.pkl', faces_path='data/face_data.pkl',
                 bg_image_path='bg.png'):
        # Initialize video capture
        self.video = cv2.VideoCapture(0)

        # Initialize attendance logger
        self.attendance_logger = AttendanceLogger()

        # Load training data
        try:
            with open(names_path, 'rb') as f:
                self.LABELS = pickle.load(f)
            with open(faces_path, 'rb') as f:
                self.FACES = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return

        # Ensure consistent length between features and labels
        min_len = min(len(self.FACES), len(self.LABELS))
        self.FACES = self.FACES[:min_len]
        self.LABELS = self.LABELS[:min_len]

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.FACES, self.LABELS, test_size=0.2, random_state=42)

        # Initialize SVM classifier
        self.svm = SVC(kernel='linear', probability=True, C=0.01)  # Adjust 'C' as necessary
        self.svm.fit(X_train, y_train)


        # Validate the model
        val_predictions = self.svm.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        scores = cross_val_score(self.svm, self.FACES, self.LABELS, cv=5)
        print(f"Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%")

        # Load background image for display
        self.imgbackground = cv2.imread(bg_image_path)

    def recognize_faces(self):
        while True:
            ret, frame = self.video.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            for (top, right, bottom, left) in face_locations:
                crop_img = rgb_frame[top:bottom, left:right]
                resized_img = cv2.resize(crop_img, (50, 50))
                resized_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY).flatten().reshape(1, -1)

                # Predict and debug output
                output = self.svm.predict(resized_img_gray)
                confidence = self.svm.predict_proba(resized_img_gray)
                print(f"Predicted Label: {output[0]}, Confidence: {np.max(confidence) * 100:.2f}%")

                # Draw rectangles and text on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (50, 50, 255), 2)
                cv2.rectangle(frame, (left, top - 40), (right, top), (50, 50, 255), -1)
                cv2.putText(frame, str(output[0]), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                self.imgbackground[162:162 + 480, 55:55 + 640] = frame

            cv2.imshow("frame", self.imgbackground)
            k = cv2.waitKey(1)
            if k == ord('o'):
                time.sleep(5)
                self.attendance_logger.log_attendance(output[0])
            elif k == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.recognize_faces()
