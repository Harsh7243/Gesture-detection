import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Gesture categories
gestures = ['increase_volume', 'decrease_volume', 'mute', 'unmute']

# Image size
IMG_SIZE = 400

def load_data():
    """
    Load images from dataset directory and create data and label arrays.
    """
    data = []
    labels = []
    for idx, gesture in enumerate(gestures):
        gesture_folder = f'dataset/{gesture}'
        for img_file in os.listdir(gesture_folder):
            img_path = os.path.join(gesture_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img_resized)
                labels.append(idx)
    
    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

def prepare_data(data, labels):
    """
    Prepare data by normalizing and splitting into train and validation sets.
    """
    # Normalize the data
    data = data / 255.0
    data = np.expand_dims(data, axis=-1)  # Add channel dimension

    # Convert labels to categorical (one-hot encoded)
    labels = to_categorical(labels, num_classes=len(gestures))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val
