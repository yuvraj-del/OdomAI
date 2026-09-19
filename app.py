from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import uuid
from dotenv import load_dotenv
from db import init_db, close_db, create_user_if_new, save_car, get_user_cars

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "https://odomai.onrender.com"])

init_db()


@app.teardown_appcontext
def teardown_db(exception):
    close_db(exception)


model = joblib.load('odomai_model.pkl')
target_encoder = joblib.load('target_encoder.pkl')
model_columns = joblib.load('model_columns.pkl')

TRAINING_REFERENCE_YEAR = 2026
LUXURY_MANUFACTURERS = {
    'bmw', 'mercedes', 'audi', 'porsche', 'cadillac', 'lexus', 'infiniti', 'acura',
    'jaguar', 'land rover', 'range rover', 'maserati', 'genesis', 'mini', 'tesla',
    'volvo', 'alfa romeo'
}
LUXURY_MULTIPLIERS = {
    'bmw': 0.76,
    'mercedes': 0.72,
    'audi': 0.75,
    'porsche': 0.70,
    'cadillac': 0.78,
    'lexus': 0.81,
    'infiniti': 0.83,
    'acura': 0.82,
    'tesla': 0.80,
    'land rover': 0.77,
    'range rover': 0.77,
    'jaguar': 0.79,
    'maserati': 0.74,
    'genesis': 0.84,
    'volvo': 0.86,
    'mini': 0.88,
    'alfa romeo': 0.80,
}

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


def get_or_create_user_id():
    user_id = request.cookies.get('user_id')
    is_new = False
    if not user_id:
        user_id = str(uuid.uuid4())
        is_new = True
    return user_id, is_new


def compute_confidence(year, odometer, manufacturer, model_name, fuel, transmission):
    age = max(0, TRAINING_REFERENCE_YEAR - int(year))
    age_score = max(0.0, 100.0 - age * 6.0)
    mileage_score = max(0.0, 100.0 - (float(odometer) / 200000.0) * 100.0)

    manufacturer_norm = str(manufacturer).lower().strip()
    model_norm = str(model_name).lower().strip()
    luxury_bonus = 6 if manufacturer_norm in LUXURY_MANUFACTURERS else 0
    if any(token in model_norm for token in ['m5', 'rs', 'amg', 's class', '7 series', 'x5', 'range rover', 'escalade', 'g class', 'q7']):
        luxury_bonus += 8

    fuel_bonus = {'gas': 4, 'diesel': 3, 'hybrid': 5, 'electric': 6}.get(str(fuel).lower().strip(), 0)
    transmission_bonus = 4 if str(transmission).lower().strip() == 'automatic' else 0

    score = (age_score * 0.45) + (mileage_score * 0.45) + luxury_bonus + fuel_bonus + transmission_bonus
    return int(max(38, min(96, round(score))))


def apply_luxury_price_adjustment(predicted_price, manufacturer, model_name):
    manufacturer_norm = str(manufacturer).lower().strip()
    model_norm = str(model_name).lower().strip()

    if manufacturer_norm in LUXURY_MANUFACTURERS:
        multiplier = LUXURY_MULTIPLIERS.get(manufacturer_norm, 0.82)
        if any(token in model_norm for token in ['m5', 'rs', 'amg', 's class', '7 series', 'x5', 'range rover', 'escalade', 'g class', 'q7', 'gt']):
            multiplier *= 0.92
        return float(predicted_price) * multiplier

    if any(token in model_norm for token in ['luxury', 'rs', 'amg', 's class', '7 series', 'x5', 'g class', 'm series']):
        return float(predicted_price) * 0.88

    return float(predicted_price)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/metadata', methods=['GET'])
def metadata():
    return jsonify({
        "manufacturers": sorted(MFR_MODELS.keys()),
        "models_by_manufacturer": MFR_MODELS,
        "fuel_types": ["gas", "diesel", "hybrid", "electric"],
        "transmissions": ["automatic", "manual"]
    })


@app.route('/predict', methods=['POST'])
def predict():
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

    adjusted_price = apply_luxury_price_adjustment(predicted_price, manufacturer, model_name)
    confidence = compute_confidence(year, odometer, manufacturer, model_name, fuel, transmission)
    rounded_price = round(adjusted_price, 2)

    user_id, is_new_user = get_or_create_user_id()

    try:
        create_user_if_new(user_id)
        save_car(
            user_id=user_id,
            make=manufacturer,
            model=model_name,
            year=year,
            mileage=odometer,
            condition=None,
            predicted_price=rounded_price
        )
    except Exception as e:
        print(f"WARNING: Failed to save car history: {e}")

    response = jsonify({
        "predicted_price": rounded_price,
        "confidence": confidence,
        "adjustment_multiplier": round(float(adjusted_price / max(predicted_price, 1.0)), 3),
        "input": {
            "manufacturer": manufacturer,
            "model": model_name,
            "fuel": fuel,
            "transmission": transmission,
            "year": year,
            "odometer": odometer
        }
    })

    if is_new_user:
        response.set_cookie(
            'user_id',
            user_id,
            max_age=60 * 60 * 24 * 365,
            secure=True,
            httponly=True,
            samesite='None'
        )

    return response


@app.route('/cars', methods=['GET'])
def get_cars():
    user_id = request.cookies.get('user_id')
    if not user_id:
        return jsonify([])

    try:
        cars = get_user_cars(user_id)
        return jsonify([dict(c) for c in cars])
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve cars: {str(e)}"}), 500


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)
