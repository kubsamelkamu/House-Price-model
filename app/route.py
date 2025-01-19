from flask import request, jsonify
import joblib
from app import app

model = joblib.load('../models/gradient_boosting.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['feature1'], data['feature2'], ...]  
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})