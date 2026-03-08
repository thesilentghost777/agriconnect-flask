import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Charger le modèle une seule fois au démarrage
MODEL_PATH = os.path.join('model', 'model.h5')
model = load_model(MODEL_PATH)

# Classes de maladies (adapter selon ton modèle)
CLASS_NAMES = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Healthy'
]

# Taille d'entrée du modèle (adapter si nécessaire)
IMG_SIZE = (224, 224)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype('float32')  # ← pas de /255 !
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': 'Format non supporté'}), 400

    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        result = {
            'disease': CLASS_NAMES[predicted_index],
            'confidence': round(confidence, 2),
            'is_healthy': CLASS_NAMES[predicted_index] == 'Healthy'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)