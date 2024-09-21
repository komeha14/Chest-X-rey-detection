# Chest X-Ray Pneumonia Detection

[**Live App**](https://chest-x-rey-detection-8vcv4k4ykh5ecmnqez9hns.streamlit.app/)

This project is a deep learning application that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) model. The application is built with Streamlit to provide an interactive interface for users to upload X-ray images and get real-time predictions.

## Overview

Chest X-ray images are one of the most common imaging methods used to diagnose pneumonia. In this project, a machine learning model is trained to classify X-ray images as either pneumonia-positive or pneumonia-negative. The trained model is hosted and integrated into a Streamlit web app, which allows users to upload their X-ray images and get a prediction of whether or not pneumonia is detected.

## How it works

1. Users upload an X-ray image through the web interface.
2. The image is processed and analyzed by a trained CNN model.
3. The model returns a prediction indicating if the patient likely has pneumonia or not.

### Steps:
- **Data Preprocessing**: Data augmentation and scaling to normalize the images.
- **Model Training**: A CNN model was trained using a large dataset of chest X-ray images.
- **Evaluation**: The model was evaluated on a validation set to ensure accuracy and reduce overfitting.
- **Deployment**: The trained model was deployed using Streamlit to create an easy-to-use web interface.

## Team Roles

- **Mohamed Komeha**: Worked on the machine learning model and built the Streamlit application.
- **Hajar Emad**: Contributed to model development and dataset preprocessing.
- **Sara Sameh**: Assisted in model training, evaluation, and hyperparameter tuning.

## GitHub Profiles

- [**Mohamed Komeha**](https://github.com/komeha14)
- [**Sara Sameh**](https://github.com/S378818)
- **Hajar Emad**: (https://github.com/HajerEma)

## How to Run Locally

To run the app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/komeha14/Chest-X-ray-detection-Peunomonia.git
