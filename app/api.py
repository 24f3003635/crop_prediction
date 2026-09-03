import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

api_bp=Blueprint("api",__name__)

model=joblib.load('/mnt/c/Programs/Hackathon/Crop_prediction/Crop_Prediction/ML/crop_model.pkl')
preprocessing_pipeline = joblib.load('/mnt/c/Programs/Hackathon/Crop_prediction/Crop_Prediction/ML/preprocessing_pipeline.pkl')
encoder = joblib.load('/mnt/c/Programs/Hackathon/Crop_prediction/Crop_Prediction/ML/label_encoder.pkl')



@api_bp.route("/test")
def test():
    return jsonify({"message":"Backend Online"})

@api_bp.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    try:
        amount_of_nitrogen = float(data['N']) 
        amount_of_phosphorous = float(data['P'])
        amount_of_potassium = float(data['K'])
        Temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Missing or invalid input: {str(e)}'}), 400  # noqa: RUF010

    min_confidence = float(data.get('min_confidence', 10))
    top_n = int(data.get('top_n', 5))

    user_input = pd.DataFrame({
        'N': [amount_of_nitrogen],
        'P': [amount_of_phosphorous],
        'K': [amount_of_potassium],
        'temperature': [Temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    user_input_processed = preprocessing_pipeline.transform(user_input)

    probabilities = model.predict_proba(user_input_processed)[0]
    crop_names = encoder.classes_
    top_idx = np.argsort(probabilities)[::-1]

    recommendations = []
    for rank, idx in enumerate(top_idx, start=1):
        if rank > top_n:
            break
        confidence = probabilities[idx] * 100
        if confidence < min_confidence:
            break
        recommendations.append({
            'rank': rank,
            'crop': crop_names[idx],
            'confidence': round(float(confidence), 2)
        })

    return jsonify({'recommendations': recommendations})


