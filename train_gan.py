import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv3D, Conv3DTranspose
from tensorflow.keras.optimizers import Adam

# Set GPU memory growth to avoid resource exhaustion
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
VIDEO_HEIGHT = 256  # Height of video frames
VIDEO_WIDTH = 256   # Width of video frames
VIDEO_CHANNELS = 3  # Number of channels in video frames
LATENT_DIM = 100    # Dimensionality of the latent space

# Generator function
def build_generator():
    model = Sequential([
        Dense(64 * 64 * 32, input_dim=LATENT_DIM),
        Reshape((64, 64, 32)),
        Conv3DTranspose(16, (3, 3, 3), strides=(2, 2, 2), padding='same'),
        Conv3DTranspose(8, (3, 3, 3), strides=(2, 2, 2), padding='same'),
        Conv3DTranspose(VIDEO_CHANNELS, (3, 3, 3), activation='sigmoid', padding='same')
    ])
    return model

# Discriminator function
def build_discriminator():
    model = Sequential([
        Conv3D(8, (3, 3, 3), strides=(2, 2, 2), padding='same', input_shape=(VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS)),
        Conv3D(16, (3, 3, 3), strides=(2, 2, 2), padding='same'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Combine Generator and Discriminator into GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(LATENT_DIM,))
    generated_video = generator(gan_input)
    gan_output = discriminator(generated_video)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return gan

# Load and preprocess training data
# def load_training_data(data_path):
#     training_data = []
#     for filename in sorted(os.listdir(data_path)):
#         if filename.endswith('.avi'):
#             video_path = os.path.join(data_path, filename)
#             # Implement code to load and preprocess video frames
#             # Example: Use OpenCV to read video frames and resize them
#             # Append preprocessed frames to training_data list
#     return np.array(training_data)

def load_training_data(data_path):
    training_data = []
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith('.avi'):
            video_path = os.path.join(data_path, filename)
            frames = []
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize frame to desired dimensions (VIDEO_HEIGHT x VIDEO_WIDTH)
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                # Preprocess frame (e.g., normalization, converting to float)
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
                # Append preprocessed frame to frames list
                frames.append(frame)
            cap.release()
            # Append frames of the video to training_data list
            training_data.append(frames)
    return np.array(training_data)

if __name__ == '__main__':
    # Set data path
    data_path = './data/avenue/training_videos'
    
    # Load training data
    training_data = load_training_data(data_path)
    
    # Build and compile models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # Training loop
    epochs = 100
    batch_size = 8
    half_batch = batch_size // 2
    
    for epoch in range(epochs):
        # Select a random half batch of real video frames
        idx = np.random.randint(0, training_data.shape[0], half_batch)
        real_videos = training_data[idx]
        
        # Generate a half batch of fake video frames
        noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
        fake_videos = generator.predict(noise)
        
        # Train the Discriminator (real classified as ones and fake as zeros)
        d_loss_real = discriminator.train_on_batch(real_videos, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_videos, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the Generator (via the GAN model, where the discriminator weights are frozen)
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, [D loss: {d_loss:.4f}, acc.: {100*d_loss_real[1]:.2f}], [G loss: {g_loss:.4f}]")
        
        # Save generated video frames at intervals
        
    # Save models at the end of training
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    gan.save('gan_model.h5')
