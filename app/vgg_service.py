# vgg19_service.py

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS 


app = Flask(__name__)
CORS(app)

# Map numerical labels to genre names
label_to_genre = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz' ,
    6: 'metal',
    7 : 'pop' ,
    8: 'reggae',
    9: 'rock',
    # Add other genre mappings as needed
}
# Load the saved VGG19 model

model_vgg19 = tf.saved_model.load('vgg_model.pb')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array

@app.route('/classify_vgg19', methods=['POST'])
def classify_vgg19():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    file_path = '/tmp/uploaded_image.jpg'
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make prediction using the loaded model
    predictions = model_vgg19(img_array)
    decoded_predictions = tf.keras.applications.vgg19.decode_predictions(predictions.numpy(), top=3)[0]

    # Return predictions as JSON
    result = [{'label': label, 'probability': float(prob)} for (_, label, prob) in decoded_predictions]
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
