import cv2
import os
import time
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from attendance_logger import AttendanceLogger


class FaceRecognizer:
    def __init__(self, haarcascade_path='haarcascade_frontalface_default.xml',
                 names_path='data/names.pkl', faces_path='data/face_data.pkl',
                 bg_image_path='bg.png'):
        # Initialize video capture and face detection
        self.video = cv2.VideoCapture(0)
        self.facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + haarcascade_path)

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

        # Convert training images to grayscale, apply histogram equalization, and ensure consistent preprocessing
        self.FACES = self.preprocess_faces(self.FACES)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.FACES, self.LABELS, test_size=0.2, random_state=42)

        # Initialize KNN classifier
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(X_train, y_train)

        # Validate the model
        val_predictions = self.knn.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Load background image for display
        self.imgbackground = cv2.imread(bg_image_path)

    def preprocess_faces(self, faces):
        processed_faces = []
        for face in faces:
            # Check if the image has more than one channel (color image)
            if len(face.shape) == 3 and face.shape[2] == 3:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face

            # Apply histogram equalization
            face_eq = cv2.equalizeHist(face_gray)
            face_resized = cv2.resize(face_eq, (50, 50)).flatten()
            processed_faces.append(face_resized)
        return np.array(processed_faces)

    def recognize_faces(self):
        while True:
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), (50, 50))

                # Apply histogram equalization
                resized_img_eq = cv2.equalizeHist(resized_img).flatten().reshape(1, -1)

                # Predict and debug output
                output = self.knn.predict(resized_img_eq)
                confidence = self.knn.predict_proba(resized_img_eq)
                print(f"Predicted Label: {output[0]}, Confidence: {np.max(confidence) * 100:.2f}%")

                # Draw rectangles and text on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

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
