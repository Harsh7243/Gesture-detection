# import cv2
# import os
# import time
# import mediapipe as mp
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)  # Allow up to 2 hands
# mp_drawing = mp.solutions.drawing_utils

# # Initialize video capture
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# # Gesture categories
# gestures = ['increase_volume', 'decrease_volume', 'mute', 'unmute']

# # Create directories for each gesture
# for gesture in gestures:
#     os.makedirs(f'dataset/{gesture}', exist_ok=True)

# # Initialize ImageDataGenerator for data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# current_gesture = gestures[0]  # Start with the first gesture
# count = 0  # Initialize image count

# print(f"Collecting images for: {current_gesture}")
# print("Press 'n' to switch to the next gesture, or 'q' to quit.")

# while True:
#     ret, frame = cap.read()
    
#     if ret:
#         # Convert the frame to RGB as required by MediaPipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame with MediaPipe
#         result = hands.process(rgb_frame)
        
#         if result.multi_hand_landmarks:
#             for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
#                 # Draw hand landmarks on the frame for visualization (optional)
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 # Get bounding box around the hand
#                 h, w, _ = frame.shape
#                 x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
#                 y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
#                 x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
#                 y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
                
#                 # Crop the hand region
#                 hand_region = frame[y_min:y_max, x_min:x_max]
                
#                 if hand_region.size > 0:
#                     # Convert the cropped hand region to grayscale
#                     gray_frame = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
#                     resized_frame = cv2.resize(gray_frame, (400, 400))
                    
#                     # Augment the image
#                     augmented_images = datagen.flow(np.expand_dims(resized_frame, axis=0), batch_size=1)
                    
#                     # Save the augmented frames to the corresponding gesture folder
#                     for i in range(3):  # Save a few augmented images
#                         aug_image = next(augmented_images)[0].astype('uint8')
#                         cv2.imwrite(f'dataset/{current_gesture}/hand_{hand_idx}_image_{count}_{i}.jpg', aug_image)
                    
#                     count += 1
                    
#                     # Display the hand region being captured
#                     cv2.imshow(f'Hand {hand_idx} Region', hand_region)
        
#         # Display the frame with the drawn landmarks
#         cv2.imshow('Collecting Gesture Data', frame)

#         # Add a small delay to allow proper positioning of gestures
#         time.sleep(0.5)
        
#     else:
#         print("Error: Unable to read frame from camera.")
#         break

#     # Switch gesture on pressing 'n'
#     if cv2.waitKey(1) & 0xFF == ord('n'):
#         count = 0
#         current_gesture = gestures[(gestures.index(current_gesture) + 1) % len(gestures)]
#         print(f"Collecting images for: {current_gesture}")

#     # Quit the collection on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Quitting data collection.")
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Gesture categories
gestures = ['increase_volume', 'decrease_volume', 'mute', 'unmute']

# Create directories for each gesture
for gesture in gestures:
    os.makedirs(f'dataset/{gesture}', exist_ok=True)

# Initialize ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

current_gesture = gestures[0]  # Start with the first gesture
count = 0  # Initialize image count

print(f"Collecting images for: {current_gesture}")
print("Press 'n' to switch to the next gesture, or 'q' to quit.")

while True:
    try:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        # Convert the frame to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Draw hand landmarks on the frame for visualization
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box around the hand
                h, w, _ = frame.shape
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
                
                # Crop the hand region
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                if hand_region.size > 0:
                    # Convert the cropped hand region to grayscale
                    gray_frame = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                    resized_frame = cv2.resize(gray_frame, (400, 400))
                    
                    # Expand dimensions to match the required input shape (1, 400, 400, 1)
                    expanded_frame = np.expand_dims(resized_frame, axis=0)  # Shape: (1, 400, 400)
                    expanded_frame = np.expand_dims(expanded_frame, axis=-1)  # Shape: (1, 400, 400, 1)
                    
                    # Augment the image
                    augmented_images = datagen.flow(expanded_frame, batch_size=1)
                    
                    # Save the augmented frames to the corresponding gesture folder
                    for i in range(3):  # Save a few augmented images
                        aug_image = next(augmented_images)[0].astype('uint8')
                        file_path = f'dataset/{current_gesture}/hand_{hand_idx}_image_{count}_{i}.jpg'
                        cv2.imwrite(file_path, aug_image)
                        print(f"Saved: {file_path}")
                    
                    count += 1
                    
                    # Display the hand region being captured
                    cv2.imshow(f'Hand {hand_idx} Region', hand_region)
        
        # Display the frame with the drawn landmarks
        cv2.imshow('Collecting Gesture Data', frame)

        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('n'):
            count = 0
            current_gesture = gestures[(gestures.index(current_gesture) + 1) % len(gestures)]
            print(f"Collecting images for: {current_gesture}")
        elif key == ord('q'):
            print("Quitting data collection.")
            break
        elif key == 27:  # ESC key to quit (if needed)
            print("ESC key pressed, quitting data collection.")
            break

    except Exception as e:
        print(f"An error occurred: {e}")

cap.release()
cv2.destroyAllWindows()
