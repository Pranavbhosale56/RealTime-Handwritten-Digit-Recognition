from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io, base64, re

app = Flask(__name__)

# Load trained MNIST CNN model
MODEL_PATH = 'models/mnist_cnn.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image_from_base64(b64data: str) -> np.ndarray:
    # Extract Base64 image content
    match = re.search(r"base64,(.*)", b64data)
    if not match:
        raise ValueError("Invalid image data")

    # Decode Base64 to bytes and open as grayscale image
    img_bytes = base64.b64decode(match.group(1))
    img = Image.open(io.BytesIO(img_bytes)).convert('L')

    # Invert (white digit on black background)
    img = ImageOps.invert(img)

    # Convert to numpy for cropping
    arr = np.array(img)

    # Find non-zero (digit) pixel locations
    coords = np.column_stack(np.where(arr > 0))

    # Crop around the digit
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        arr = arr[y_min:y_max+1, x_min:x_max+1]
    else:
        arr = np.zeros((28, 28), dtype=np.uint8)

    # Resize cropped digit to max 20x20
    img = Image.fromarray(arr)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create 28x28 black canvas and center the digit
    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))

    # Normalize and reshape for model input
    arr = np.array(canvas).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

@app.route('/')
def index():
    # Render the drawing UI page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON containing Base64 image
        data = request.get_json()
        img_data = data.get('image')

        # Preprocess for model
        x = preprocess_image_from_base64(img_data)

        # Perform prediction
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))

        # Return prediction + probabilities
        return jsonify({
            'prediction': pred,
            'probabilities': [float(p) for p in probs]
        })
    except Exception as e:
        # Handle errors safely
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
