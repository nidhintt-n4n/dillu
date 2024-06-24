import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv3D, Conv3DTranspose
from tensorflow.keras.optimizers import Adam
import cv2

# Constants
VIDEO_HEIGHT = 256   # Height of video frames
VIDEO_WIDTH = 256    # Width of video frames
VIDEO_CHANNELS = 3   # Number of channels in video frames
LATENT_DIM = 100     # Dimensionality of the latent space

# Function to build the Autoencoder model
def build_autoencoder():
    # Encoder
    input_layer = Input(shape=(VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS))
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(input_layer)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)
    x = Flatten()(x)
    encoded = Dense(LATENT_DIM, activation='relu')(x)
    
    # Decoder
    x = Dense(16 * (VIDEO_HEIGHT // 2) * (VIDEO_WIDTH // 2), activation='relu')(encoded)
    x = Reshape((VIDEO_HEIGHT // 2, VIDEO_WIDTH // 2, 16))(x)
    x = Conv3DTranspose(8, (3, 3, 3), activation='relu', padding='same', strides=(2, 2, 2))(x)
    decoded = Conv3DTranspose(VIDEO_CHANNELS, (3, 3, 3), activation='sigmoid', padding='same')(x)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Function to load and preprocess training data
def load_training_data(data_path):
    training_data = []
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith('.avi'):
            video_path = os.path.join(data_path, filename)
            frames = []
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
                frames.append(frame)
            cap.release()
            training_data.append(frames)
    return np.array(training_data)

if __name__ == '__main__':
    # Set data path
    data_path = './data/avenue/training_videos'
    
    # Load training data
    training_data = load_training_data(data_path)
    
    # Reshape training data to (num_videos * num_frames, height, width, channels)
    num_videos, num_frames, height, width, channels = training_data.shape
    training_data = np.reshape(training_data, (num_videos * num_frames, height, width, channels))
    
    # Build autoencoder model
    autoencoder = build_autoencoder()
    
    # Training parameters
    epochs = 50
    batch_size = 8
    
    # Train the autoencoder
    autoencoder.fit(training_data, training_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True)
    
    # Save the trained autoencoder model
    autoencoder.save('autoencoder_model.h5')
