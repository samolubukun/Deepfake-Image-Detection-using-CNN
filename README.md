# Deepfake Image Detection using CNN

This repository provides a deep learning-based solution for detecting deepfake images using Convolutional Neural Networks (CNNs). The system analyzes uploaded images and predicts whether the image is real or deepfake using a trained CNN model.

![Screenshot (263)](https://github.com/user-attachments/assets/1d0acef4-6d9c-47c7-810e-2d8e2f894087)

## Features
- **Streamlit Web App:** A user-friendly interface for uploading and analyzing images.
- **Trained CNN Model:** Achieves accurate predictions on real and deepfake images.

## Model Summary
The CNN model was trained to classify images as real or fake:
- **Architecture:** Sequential CNN with convolutional, pooling, and dense layers.
- **Input Shape:** 256x256 RGB images.
- **Performance Metrics:**
  - Accuracy: 94.7%
  - Precision: 93.9%
  - Recall: 95.2%

- **Access Training Notebook:** [Notebook](https://github.com/samolubukun/Deepfake-Image-Detection-using-CNN/tree/main/Notebook)
- **Dataset Used:** [140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

- **Model Download:** Download the trained model file and place it in the same folder as the Streamlit app (e.g., `app.py`) for execution: [deepfake_image_detection.h5](https://drive.google.com/file/d/13pMFr1UYuDJ_5wxRQmBgxMwc6XItXjBh/view?usp=sharing)

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/samolubukun/Deepfake-Image-Detection-using-CNN.git
   cd Deepfake-Image-Detection-using-CNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
