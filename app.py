from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import os

app = Flask(__name__)

# Load model khi app khởi động
MODEL_PATH = "your_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data received'})

        img_data = base64.b64decode(data.split(',')[1])
        img = Image.open(io.BytesIO(img_data)).convert('L')  # Grayscale

        bbox = img.getbbox()
        if not bbox:
            return jsonify({'digit': 'Vui lòng vẽ chữ số!'})

        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        size = max(width, height) + 20
        img_processed = Image.new("L", (size, size), 255)
        img_processed.paste(img.crop(bbox), ((size - width) // 2, (size - height) // 2))
        img_processed = img_processed.resize((20, 20))

        img_28x28 = Image.new("L", (28, 28), 255)
        img_28x28.paste(img_processed, (4, 4))
        img_28x28 = ImageOps.invert(img_28x28)

        img_array = np.array(img_28x28) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Thêm channel cho model CNN

        prediction = model.predict(img_array, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({'digit': digit, 'confidence': f"{confidence:.2%}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
