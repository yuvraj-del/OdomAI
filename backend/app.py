import logging
import re
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import uuid
from dotenv import load_dotenv
from backend.db import init_db, close_db, create_user_if_new, save_car, get_user_cars

logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
FRONTEND_DIR = BASE_DIR / 'frontend'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), template_folder=str(FRONTEND_DIR))
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "https://odomai.onrender.com"])

init_db()


@app.teardown_appcontext
def teardown_db(exception):
    close_db(exception)


model = joblib.load(MODELS_DIR / 'odomai_model.pkl')
target_encoder = joblib.load(MODELS_DIR / 'target_encoder.pkl')
model_columns = joblib.load(MODELS_DIR / 'model_columns.pkl')

TRAINING_REFERENCE_YEAR = 2026
MIN_YEAR = 1995
MAX_YEAR = 2024
MIN_ODOMETER = 1000
MAX_ODOMETER = 300000
ALLOWED_FUEL_TYPES = {"gas", "diesel", "hybrid", "electric"}
ALLOWED_TRANSMISSIONS = {"automatic", "manual"}

LUXURY_MANUFACTURERS = {
    'bmw', 'mercedes', 'mercedes-benz', 'audi', 'porsche', 'cadillac', 'lexus', 'infiniti', 'acura',
    'jaguar', 'land rover', 'range rover', 'maserati', 'genesis', 'mini', 'tesla',
    'volvo', 'alfa romeo', 'alfa-romeo'
}
LUXURY_MULTIPLIERS = {
    'bmw': 0.76,
    'mercedes': 0.72,
    'mercedes-benz': 0.72,
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
    'alfa-romeo': 0.80,
}
LUXURY_MANUFACTURER_ALIASES = {
    'mercedes-benz': {'mercedes-benz', 'mercedes benz', 'mercedes'},
    'alfa-romeo': {'alfa-romeo', 'alfa romeo', 'alfa'},
    'range rover': {'range rover', 'range-rover', 'rangerover'},
    'land rover': {'land rover', 'land-rover', 'landrover'},
}
MAINSTREAM_AGE_CENTER = 9.0
MAINSTREAM_ODOMETER_CENTER = 70000.0
MAINSTREAM_AGE_SPREAD = 12.0
MAINSTREAM_ODOMETER_SPREAD = 90000.0

try:
    df_clean = pd.read_csv(MODELS_DIR / 'vehicles_clean.csv')
    MFR_MODELS = (
        df_clean.groupby('manufacturer')['model']
        .unique()
        .apply(lambda arr: sorted(arr.tolist()))
        .to_dict()
    )
    if not df_clean.empty:
        df_clean = df_clean.dropna(subset=['year', 'odometer']).copy()
        df_clean['age'] = np.maximum(0, TRAINING_REFERENCE_YEAR - df_clean['year'].astype(float))
        MAINSTREAM_AGE_CENTER = float(df_clean['age'].median())
        MAINSTREAM_ODOMETER_CENTER = float(df_clean['odometer'].astype(float).median())
        age_q1, age_q3 = df_clean['age'].quantile([0.25, 0.75])
        miles_q1, miles_q3 = df_clean['odometer'].astype(float).quantile([0.25, 0.75])
        MAINSTREAM_AGE_SPREAD = max(float(age_q3 - age_q1), 10.0)
        MAINSTREAM_ODOMETER_SPREAD = max(float(miles_q3 - miles_q1), 50000.0)
except FileNotFoundError:
    MFR_MODELS = {}
    logger.error("vehicles_clean.csv not found — /metadata will return empty options.")


def normalize_manufacturer_key(value):
    return re.sub(r"[\s_-]+", ' ', str(value).lower().strip())


def canonicalize_manufacturer(value):
    normalized = normalize_manufacturer_key(value)
    for canonical, aliases in LUXURY_MANUFACTURER_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def contains_model_phrase(model_name, token):
    normalized_model = re.sub(r'[^a-z0-9]+', ' ', str(model_name).lower().strip())
    normalized_token = re.sub(r'[^a-z0-9]+', ' ', str(token).lower().strip())
    if not normalized_token:
        return False
    pattern = rf'(^|\s){re.escape(normalized_token)}(\s|$)'
    return re.search(pattern, normalized_model) is not None


def has_luxury_model_bump(model_name):
    return any(
        contains_model_phrase(model_name, token)
        for token in ['m5', 'rs', 'amg', 's class', '7 series', 'x5', 'range rover', 'escalade', 'g class', 'q7', 'gt']
    )


def get_or_create_user_id():
    user_id = request.cookies.get('user_id')
    is_new = False
    if not user_id:
        user_id = str(uuid.uuid4())
        is_new = True
    return user_id, is_new


def compute_confidence(year, odometer):
    age = max(0, TRAINING_REFERENCE_YEAR - int(year))
    mileage = max(0.0, float(odometer))

    age_distance = abs(age - MAINSTREAM_AGE_CENTER) / max(MAINSTREAM_AGE_SPREAD, 1.0)
    mileage_distance = abs(mileage - MAINSTREAM_ODOMETER_CENTER) / max(MAINSTREAM_ODOMETER_SPREAD, 1.0)
    score = 100.0 - (age_distance * 70.0) - (mileage_distance * 50.0)
    return int(max(38, min(96, round(score))))


def apply_progressive_price_reduction(predicted_price):
    price = float(predicted_price)
    if price <= 15000:
        return price
    if price <= 25000:
        # Continuous from 0% at $15,000 to 30% at $25,000.
        reduction_rate = 0.30 * ((price - 15000.0) / 10000.0)
    elif price <= 40000:
        reduction_rate = 0.30 + ((price - 25000.0) / 15000.0) * 0.06
    else:
        reduction_rate = 0.36 + min(0.12, ((price - 40000.0) / 50000.0) * 0.12)
    reduction_rate = max(0.0, min(reduction_rate, 0.48))
    return price * (1.0 - reduction_rate)


def apply_luxury_price_adjustment(predicted_price, manufacturer, model_name):
    # Order: luxury brand/model adjustment first, then the market-wide progressive price reduction.
    # This keeps the luxury factor as a brand-level modifier while the progressive curve remains a
    # single continuous cap for all vehicles above $15k, including non-luxury trucks.
    manufacturer_key = canonicalize_manufacturer(manufacturer)
    model_norm = str(model_name).lower().strip()

    if manufacturer_key in LUXURY_MANUFACTURERS:
        multiplier = LUXURY_MULTIPLIERS.get(manufacturer_key, 0.82)
        if has_luxury_model_bump(model_norm):
            multiplier *= 0.92
        adjusted = float(predicted_price) * multiplier
    elif any(contains_model_phrase(model_norm, token) for token in ['luxury', 'rs', 'amg', 's class', '7 series', 'x5', 'g class', 'm series']):
        adjusted = float(predicted_price) * 0.88
    else:
        adjusted = float(predicted_price)

    return apply_progressive_price_reduction(adjusted)


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
        year_raw = float(data['year'])
        odometer = float(data['odometer'])
    except (ValueError, TypeError):
        return jsonify({"error": "year must be an integer and odometer must be a number"}), 400

    if not year_raw.is_integer():
        return jsonify({"error": f"year must be an integer between {MIN_YEAR} and {MAX_YEAR}"}), 400

    year = int(year_raw)
    if year < MIN_YEAR or year > MAX_YEAR:
        return jsonify({"error": f"year must be between {MIN_YEAR} and {MAX_YEAR}"}), 400

    if odometer < MIN_ODOMETER or odometer > MAX_ODOMETER:
        return jsonify({"error": f"odometer must be between {MIN_ODOMETER:,} and {MAX_ODOMETER:,}"}), 400

    manufacturer = canonicalize_manufacturer(data['manufacturer'])
    model_name = str(data['model']).lower().strip()
    fuel = str(data['fuel']).lower().strip()
    transmission = str(data['transmission']).lower().strip()

    if fuel not in ALLOWED_FUEL_TYPES:
        return jsonify({"error": f"fuel must be one of: {', '.join(sorted(ALLOWED_FUEL_TYPES))}"}), 400

    if transmission not in ALLOWED_TRANSMISSIONS:
        return jsonify({"error": f"transmission must be one of: {', '.join(sorted(ALLOWED_TRANSMISSIONS))}"}), 400

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
    confidence = compute_confidence(year, odometer)
    rounded_price = round(adjusted_price, 2)

    user_id, is_new_user = get_or_create_user_id()

    try:
        create_user_if_new(user_id)
        save_car(
            user_id=user_id,
            make=manufacturer,
            model=model_name,
            year=year,
            mileage=int(odometer),
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
