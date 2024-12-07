# from utils import load_data, prepare_data, gestures
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import tensorflow as tf

# # Define the image size
# IMG_SIZE = 400

# # Load and preprocess the data
# data, labels = load_data()
# X_train, X_val, y_train, y_val = prepare_data(data, labels)

# # Define a simple CNN model
# def create_model(input_shape, num_classes):
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Create the model
# model = create_model((IMG_SIZE, IMG_SIZE, 1), len(gestures))

# # Initialize callbacks
# checkpoint = tf.keras.callbacks.ModelCheckpoint('gesture_model_best.h5', monitor='val_loss', save_best_only=True)
# early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,  # Increase epochs if needed
#     batch_size=32,
#     callbacks=[early_stopping, checkpoint]
# )

# # Save the model
# model.save('gesture_model.keras')
from utils import load_data, prepare_data, gestures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the image size
IMG_SIZE = 400  # Ensure this matches with the datacoll.py

# Load and preprocess the data
data, labels = load_data()

# Split into training and validation sets
X_train, X_val, y_train, y_val = prepare_data(data, labels)

# Updated input shape for 400x400 grayscale images
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model((IMG_SIZE, IMG_SIZE, 1), len(gestures))

# Train the model with 10 epochs
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Save the model in the new .keras format
model.save('gesture_model.keras')
