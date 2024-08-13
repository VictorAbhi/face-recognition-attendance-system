import cv2
import numpy as np
import os
import pickle
import face_recognition

class FaceCollector:
    def __init__(self, name, data_dir='data', max_faces=100):
        self.name = name
        self.data_dir = data_dir
        self.max_faces = max_faces
        self.face_data = []
        # Initialize video capture
        self.video = cv2.VideoCapture(0)

    def augment_data(self, image):
        """Augment data using various techniques: rotation, flipping, noise, lighting adjustments, etc."""
        augmented_images = []

        # Original image
        augmented_images.append(image)

        # Rotation
        for angle in [15, -15, 30, -30]:
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
            rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            augmented_images.append(rotated_img)

        # Flipping (0: vertical, 1: horizontal, -1: both axes)
        for flip_code in [0, 1, -1]:
            flipped_img = cv2.flip(image, flip_code)
            augmented_images.append(flipped_img)

        # Adding Gaussian noise
        def add_gaussian_noise(img, mean=0, std=10):
            noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            return noisy_img

        augmented_images.append(add_gaussian_noise(image))

        # Lighting variations: adjust brightness and contrast
        for alpha in [0.8, 1.2]:  # Alpha controls brightness
            for beta in [-10, 10]:  # Beta controls contrast
                adjusted_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                augmented_images.append(adjusted_img)

        return augmented_images

    def collect_faces(self):
        i = 0
        while True:
            ret, frame = self.video.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            for (top, right, bottom, left) in face_locations:
                crop_img = rgb_frame[top:bottom, left:right]
                resized_img = cv2.resize(crop_img, (50, 50))  # Resize to the desired dimension

                # Convert the image to grayscale (optional, if needed for augmentation)
                resized_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

                augmented_imgs = self.augment_data(resized_img_gray)

                if len(self.face_data) < self.max_faces and i % 10 == 0:
                    self.face_data.extend([img.flatten() for img in augmented_imgs])
                    cv2.putText(frame, str(len(self.face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                    cv2.rectangle(frame, (left, top), (right, bottom), (50, 50, 255), 1)

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
