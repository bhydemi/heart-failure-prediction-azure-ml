import json
import logging
import os
import numpy as np
import joblib

def init():
    """
    Initialize the model for scoring.
    This function is called once when the web service starts.
    """
    global model

    # AZUREML_MODEL_DIR is set by Azure ML when deploying
    model_dir = os.getenv('AZUREML_MODEL_DIR')

    # Look for model.joblib in the model directory
    model_path = os.path.join(model_dir, 'model.joblib')

    # If not found at root, try searching
    if not os.path.exists(model_path):
        for root, dirs, files in os.walk(model_dir):
            if 'model.joblib' in files:
                model_path = os.path.join(root, 'model.joblib')
                break

    logging.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logging.info("Model loaded successfully")

def run(raw_data):
    """
    Score the input data using the loaded model.

    Expected input format:
    {
        "data": [[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                  high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                  sex, smoking, time], ...]
    }
    """
    try:
        # Parse the input data
        data = json.loads(raw_data)
        input_data = data.get('data', data)

        # Convert to numpy array
        input_array = np.array(input_data)

        # Make predictions
        predictions = model.predict(input_array)

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(input_array).tolist()
        except:
            probabilities = None

        result = {
            'predictions': predictions.tolist(),
            'status': 'success'
        }

        if probabilities:
            result['probabilities'] = probabilities

        return json.dumps(result)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return json.dumps({
            'error': str(e),
            'status': 'failed'
        })
