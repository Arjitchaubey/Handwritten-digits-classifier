import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import random

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Predict and audio output
sample_idx = random.randint(0, len(x_test)-1)
sample_image = x_test[sample_idx].reshape(1,28,28,1)
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions)

plt.imshow(x_test[sample_idx].reshape(28,28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_class}")
plt.axis('off')
plt.show()

tts = gTTS(text=f"The predicted digit is {predicted_class}", lang='en')
tts.save("predicted_digit.mp3")
os.system("mpg321 predicted_digit.mp3")  # On Windows, use "start predicted_digit.mp3"

# To download the audio file
from google.colab import files
files.download("predicted_digit.mp3")

