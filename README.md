# Video Anomaly Detection using GANs

## Overview
This project implements a video anomaly detection system using Generative Adversarial Networks (GANs). The GAN is trained on normal video data to learn the distribution of normal frames. During inference, it identifies anomalies by comparing generated frames with real frames and computing reconstruction errors.

## Project Structure
The project structure is organized as follows:
- `data/`: Directory containing training videos.
- `train_gan.py`: Script to train the GAN on the training data.
- `generate_frames.py`: Script to generate frames using the trained GAN.
- `autoencoder.py`: Script for anomaly detection using an autoencoder (not implemented).
- `detect_anomalies.py`: Script for detecting anomalies in video frames (not implemented).
- `xai_explanations.py`: Script for Explainable AI techniques (not implemented).
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy

## Usage
1. **Setup Environment**:
   - Install Python dependencies using `pip install -r requirements.txt`.
   
2. **Prepare Data**:
   - Place your training videos (in `.avi` format) inside the `data/avenue/training_videos` directory.
   
3. **Train the GAN**:
   - Run `python train_gan.py` to train the GAN on the training videos.
   
4. **Generate Frames**:
   - After training, run `python generate_frames.py` to generate frames using the trained GAN.
   
5. **Anomaly Detection (Future Work)**:
   - Implement `autoencoder.py` and `detect_anomalies.py` for anomaly detection and use `xai_explanations.py` for Explainable AI techniques.

## Notes
- Customize the preprocessing steps in `load_training_data` function in `train_gan.py` according to your dataset and requirements.
- Extend the functionality by implementing remaining scripts for comprehensive anomaly detection and explanation.

