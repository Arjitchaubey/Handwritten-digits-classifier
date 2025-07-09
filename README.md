# Handwritten-digits-classifier

Python programmed handwritten digits classifier with mp3 audio reproduction as an output

Overview

I have implemented a machine learning system that classifies handwritten digits using CNNs trained on the MNIST dataset and provides audio feedback for the predicted digit.

Key Features:

  - High-accuracy digit recognition using a deep CNN (98.4% test accuracy).

  - Real-time audio output of predictions using Text-to-Speech (gTTS).

  - Clean, modular Python code using TensorFlow/Keras, NumPy, and Matplotlib.

  - Easily extensible for further research or deployment.

Dataset

  - MNIST: 70,000 grayscale images (28x28 pixels) of handwritten digits (0–9).

  - 60,000 images for training, 10,000 for testing.



Methodology

  - Data Loading & Preprocessing

  - Load MNIST using TensorFlow’s built-in loader.

  - Normalize pixel values to .

  - Reshape images to (28, 28, 1) for CNN input.

Model Architecture

  - Stacked Conv2D and MaxPooling2D layers for feature extraction.

  - Flatten layer followed by Dense (fully connected) layers.

  - Dropout layer to prevent overfitting.

  - Output layer with softmax activation for digit classification.

Results 
  - Accuracy: 98.4 percent

  - Robustness: Correctly classifies a wide range of handwriting styles.

  - Accessibility: Audio output enables use by visually impaired users.

How to Run

  1. Install dependencies:

```bash
pip install tensorflow numpy matplotlib gtts
```

  2. Run the main script to train the model and test audio output.

  3. For offline TTS, consider switching to pyttsx3.
