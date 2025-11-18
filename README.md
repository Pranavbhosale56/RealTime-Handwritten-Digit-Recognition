# RealTime-Handwritten-Digit-Recognition







MNIST Handwritten Digit Recognition – Flask + Realtime Canvas

Python · TensorFlow · Flask
License: MIT | GitHub Stars: ⭐ (add badge here if needed)

An interactive web application for recognizing handwritten digits using a CNN model trained on the MNIST dataset.
Users can draw directly on a browser canvas, and the model predicts the digit in realtime.

🖼️ Preview

(Add screenshot or GIF here)

✨ Features

CNN model trained on the MNIST dataset

Flask web application with /predict endpoint

Interactive browser canvas for drawing digits

Realtime prediction while drawing

Preprocessing pipeline: crop → scale → pad to match MNIST style

🚀 Installation & Running
1️⃣ Clone the repository
git clone https://github.com/gbennnn/realtime-digit-recognition.git
cd realtime-digit-recognition

2️⃣ Create a virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

3️⃣ Train the model
python train_mnist.py


This script downloads MNIST, trains the CNN, and saves the model:

models/mnist_cnn.h5

4️⃣ Run the web application
python app.py


Open your browser:

http://127.0.0.1:5000

🧠 Model Architecture

A simple CNN used for digit recognition:

Conv2D(32, 3×3, ReLU) → MaxPooling2D

Conv2D(64, 3×3, ReLU) → MaxPooling2D

Flatten

Dense(128, ReLU) → Dropout(0.3)

Dense(10, Softmax) (classification layer)

Accuracy on MNIST: ~98%
