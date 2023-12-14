from flask import Flask, request, jsonify
import joblib
import librosa
import numpy as np
import os
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Load the SVM model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svm_model.pkl')
model = joblib.load(MODEL_PATH)

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

def extract_features(file_path):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_path, sr=None)

        # Extract features (modify as needed)
        # For example, let's extract the mean of the MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=58)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Concatenate other features if necessary

        return mfccs_mean

    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

@app.route('/svm_service', methods=['POST'])
def svm_service():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})

        file = request.files['file']

        # Save the file to the 'temp' directory
        file_path = os.path.join('temp', 'uploaded_file.wav')
        file.save(file_path)

        # Extract audio features (modify as needed)
        features = extract_features(file_path)

        # Make the prediction with the SVM model
        numerical_label = model.predict([features])[0]

        # Map numerical label to genre name
        predicted_genre = label_to_genre.get(numerical_label, 'Unknown')

        return jsonify({'genre': predicted_genre})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
