import cv2
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import messagebox

class FaceCollector:
    def __init__(self, name, haarcascade_path="haarcascade_frontalface_default.xml", data_dir='data', max_faces=70):
        self.name = name
        self.data_dir = data_dir
        self.max_faces = max_faces
        self.face_data = []
        self.haarcascade_path = haarcascade_path
        # Initialize video capture and face detection
        self.video = cv2.VideoCapture(0)
        self.facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + haarcascade_path)

    def augment_data(self, image):
        """Augment data by rotating and shifting."""
        augmented_images = []
        for angle in [0, 15, -15]:
            M = cv2.getRotationMatrix2D((25, 25), angle, 1.0)
            rotated_img = cv2.warpAffine(image, M, (50, 50))
            augmented_images.append(rotated_img)
        return augmented_images

    def collect_faces(self):
        i = 0
        while True:
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (50, 50))
                augmented_imgs = self.augment_data(resized_img)

                if len(self.face_data) < self.max_faces and i % 10 == 0:
                    self.face_data.extend(augmented_imgs)
                    cv2.putText(frame, str(len(self.face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255),
                                1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)

            # Check if the window is closed
            if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                break

            # Break if the maximum number of faces is reached
            if len(self.face_data) >= self.max_faces:
                break

            i += 1

        self.video.release()
        cv2.destroyAllWindows()

    def save_data(self):
        if len(self.face_data) == 0:
            print("No face data collected, nothing to save.")
            return

        face_data_array = np.array(self.face_data)
        face_data_array = face_data_array.reshape((face_data_array.shape[0], -1))

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        names_path = os.path.join(self.data_dir, 'names.pkl')
        if not os.path.exists(names_path):
            names = [self.name] * len(self.face_data)
            with open(names_path, 'wb') as f:
                pickle.dump(names, f)
        else:
            with open(names_path, 'rb') as f:
                names = pickle.load(f)
            names.extend([self.name] * len(self.face_data))
            with open(names_path, 'wb') as f:
                pickle.dump(names, f)

        faces_path = os.path.join(self.data_dir, 'face_data.pkl')
        if not os.path.exists(faces_path):
            with open(faces_path, 'wb') as f:
                pickle.dump(face_data_array, f)
        else:
            with open(faces_path, 'rb') as f:
                existing_faces = pickle.load(f)
            updated_faces = np.vstack([existing_faces, face_data_array])
            with open(faces_path, 'wb') as f:
                pickle.dump(updated_faces, f)


if __name__ == "__main__":
    name = input("Enter your name: ")
    collector = FaceCollector(name)
    collector.collect_faces()
    collector.save_data()
