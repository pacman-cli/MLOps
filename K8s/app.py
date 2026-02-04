from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Create a Flask app
app= Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data= request.get_json()
    features = np.array(data['features']).reshape(1, -1) # Reshape for single prediction
    prediction = model.predict(features)[0] # Get the first predicted value
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


