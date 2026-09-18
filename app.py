from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Allows your React frontend (different origin) to call this API

# ============================================
# Load model artifacts once at startup
# ============================================
model = joblib.load('odomai_model.pkl')
target_encoder = joblib.load('target_encoder.pkl')
model_columns = joblib.load('model_columns.pkl')

# Reference year used during training — must match the Kaggle notebook's
# CURRENT_YEAR value at the time the model was trained. This is a temporary
# hardcode (see Issue 1) until versioned with the model artifacts.
TRAINING_REFERENCE_YEAR = 2026

# Load cleaned data once, used only to build the manufacturer -> model dropdown map.
# This ensures the frontend only ever offers combinations the model was trained on.
try:
    df_clean = pd.read_csv('vehicles_clean.csv')
    MFR_MODELS = (
        df_clean.groupby('manufacturer')['model']
        .unique()
        .apply(lambda arr: sorted(arr.tolist()))
        .to_dict()
    )
except FileNotFoundError:
    MFR_MODELS = {}
    print("WARNING: vehicles_clean.csv not found — /metadata will return empty options.")


@app.route('/health', methods=['GET'])
def health():
    """Simple endpoint to confirm the service is up (also warms it after Render cold start)."""
    return jsonify({"status": "ok"})


@app.route('/metadata', methods=['GET'])
def metadata():
    """
    Returns manufacturer -> list of valid models, plus static dropdown options
    (fuel, transmission). The frontend uses this to populate its dropdowns so
    users can only submit combinations the model was actually trained on.
    """
    return jsonify({
        "manufacturers": sorted(MFR_MODELS.keys()),
        "models_by_manufacturer": MFR_MODELS,
        "fuel_types": ["gas", "diesel", "hybrid", "electric"],
        "transmissions": ["automatic", "manual"]
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON:
    {
        "manufacturer": "honda",
        "model": "civic",
        "fuel": "gas",
        "transmission": "automatic",
        "year": 2022,
        "odometer": 25000
    }
    Returns JSON:
    {
        "predicted_price": 18574.00,
        "input": { ...echoed input... }
    }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    required_fields = ['manufacturer', 'model', 'fuel', 'transmission', 'year', 'odometer']
    missing = [f for f in required_fields if f not in data or data[f] in (None, '')]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}"}), 400

    try:
        year = int(data['year'])
        odometer = float(data['odometer'])
    except (ValueError, TypeError):
        return jsonify({"error": "year must be an integer and odometer must be a number"}), 400

    manufacturer = str(data['manufacturer']).lower().strip()
    model_name = str(data['model']).lower().strip()
    fuel = str(data['fuel']).lower().strip()
    transmission = str(data['transmission']).lower().strip()

    # Validate against known training combinations (prevents garbage predictions
    # on manufacturer/model pairs the model never saw)
    if MFR_MODELS:
        if manufacturer not in MFR_MODELS:
            return jsonify({"error": f"Unknown manufacturer: '{manufacturer}'"}), 400
        if model_name not in MFR_MODELS[manufacturer]:
            return jsonify({
                "error": f"Unknown model '{model_name}' for manufacturer '{manufacturer}'"
            }), 400

    age = max(0, TRAINING_REFERENCE_YEAR - year)
    miles_per_year = odometer / (age + 1.0)

    input_dict = {
        'manufacturer': manufacturer,
        'model': model_name,
        'fuel': fuel,
        'transmission': transmission,
        'age': age,
        'odometer': odometer,
        'miles_per_year': miles_per_year
    }
    input_df = pd.DataFrame([input_dict])

    try:
        encoded = target_encoder.transform(input_df)
        encoded = encoded[model_columns]
        log_pred = model.predict(encoded)
        predicted_price = float(np.expm1(log_pred)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "predicted_price": round(predicted_price, 2),
        "input": {
            "manufacturer": manufacturer,
            "model": model_name,
            "fuel": fuel,
            "transmission": transmission,
            "year": year,
            "odometer": odometer
        }
    })

@app.route('/', methods=['GET'])
def index():
    """Serves the static vanilla JS test page."""
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)