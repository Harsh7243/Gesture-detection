# import cv2
# import numpy as np
# import pyautogui
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from utils import gestures

# # Load the trained model
# model = load_model('gesture_model.keras')

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Define the image size
# IMG_SIZE = 400

# # Define a function to preprocess the frame for model prediction
# def preprocess_hand(landmarks, frame_shape):
#     """
#     Preprocesses the landmarks of a detected hand to be used for model prediction.
#     """
#     # Create a blank image of the frame size
#     blank_img = np.zeros(frame_shape[:2], dtype=np.uint8)
    
#     # Extract landmark coordinates and scale them to fit the blank image
#     for lm in landmarks:
#         x, y = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
#         cv2.circle(blank_img, (x, y), 10, 255, -1)  # Draw a circle for each landmark
    
#     # Resize the image to match the input size of the model (400x400)
#     resized_img = cv2.resize(blank_img, (IMG_SIZE, IMG_SIZE))
    
#     # Normalize and reshape the image
#     normalized = resized_img / 255.0  # Normalize pixel values to [0, 1]
#     reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))  # Reshape for model input
    
#     return reshaped

# # Define the function to map gestures to actions
# def perform_action(gesture_index):
#     if gesture_index == 0:
#         print("Increasing Volume")
#         pyautogui.press("volumeup")
#     elif gesture_index == 1:
#         print("Decreasing Volume")
#         pyautogui.press("volumedown")
#     elif gesture_index == 2:
#         print("Muting")
#         pyautogui.press("volumemute")
#     elif gesture_index == 3:
#         print("Unmuting")
#         pyautogui.press("volumemute")

# # Main loop for real-time gesture recognition
# with mp_hands.Hands(
#     max_num_hands=2,  # Enable detection for both hands
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5) as hands:

#     while True:
#         # Read a frame from the webcam
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to RGB format (as required by MediaPipe)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Process the frame with MediaPipe
#         results = hands.process(rgb_frame)

#         # If hand(s) are detected
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks on the frame
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Preprocess hand landmarks for model prediction
#                 preprocessed = preprocess_hand(hand_landmarks.landmark, frame.shape)

#                 # Predict the gesture
#                 prediction = model.predict(preprocessed)
#                 gesture_index = np.argmax(prediction)

#                 # Perform action based on prediction
#                 perform_action(gesture_index)

#         # Display the frame with hand landmarks
#         cv2.imshow('Real-Time Gesture Recognition', frame)

#         # Break the loop on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import gestures

# Load the trained model
model = load_model('gesture_model.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the image size
IMG_SIZE = 400

# Define a function to preprocess the hand region for model prediction
def preprocess_hand_region(hand_landmarks, frame):
    """
    Crops the hand region from the frame based on hand landmarks,
    resizes it to the model's input size, normalizes, and reshapes it.
    """
    # Extract bounding box coordinates of the hand
    h, w, _ = frame.shape
    x_min = int(min([lm.x for lm in hand_landmarks]) * w)
    y_min = int(min([lm.y for lm in hand_landmarks]) * h)
    x_max = int(max([lm.x for lm in hand_landmarks]) * w)
    y_max = int(max([lm.y for lm in hand_landmarks]) * h)
    
    # Crop the hand region from the frame
    hand_region = frame[y_min:y_max, x_min:x_max]

    if hand_region.size == 0:
        return None  # Avoid processing if the region is empty

    # Convert to grayscale and resize to 400x400
    gray_frame = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))

    # Normalize and reshape the image for model input
    normalized = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))  # Reshape for model input

    return reshaped

# Define the function to map gestures to actions
def perform_action(gesture_index):
    if gesture_index == 0:
        print("Increasing Volume")
        pyautogui.press("volumeup")
    elif gesture_index == 1:
        print("Decreasing Volume")
        pyautogui.press("volumedown")
    elif gesture_index == 2:
        print("Muting")
        pyautogui.press("volumemute")
    elif gesture_index == 3:
        print("Unmuting")
        pyautogui.press("volumemute")

# Main loop for real-time gesture recognition
with mp_hands.Hands(
    max_num_hands=2,  # Enable detection for both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB format (as required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)

        # If hand(s) are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Preprocess hand landmarks for model prediction
                preprocessed = preprocess_hand_region(hand_landmarks.landmark, frame)

                if preprocessed is not None:
                    # Predict the gesture
                    prediction = model.predict(preprocessed)
                    gesture_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Perform action based on prediction with a confidence threshold
                    if confidence > 0.7:  # Threshold can be adjusted based on model performance
                        perform_action(gesture_index)

        # Display the frame with hand landmarks
        cv2.imshow('Real-Time Gesture Recognition', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
