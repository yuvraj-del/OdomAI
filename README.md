# OdomAI

OdomAI is a full-stack AI-powered used car valuation app that predicts a vehicle’s market price from key inputs like year, mileage, make, model, fuel type, and transmission. The project demonstrates the entire process of taking raw vehicle data, training a machine learning model, exposing it through a backend API, and integrating it into a front-end experience that helps users make faster, more informed car-buying decisions.

This project is designed to showcase practical AI and software engineering skills: from data cleaning and model development to REST API design, frontend integration, and persistence in a backend database.

## Project Summary

OdomAI solves a common real-world problem: used cars are difficult to price accurately because their value depends on age, mileage, brand, model, fuel type, and market demand. Instead of relying on a manual guess, this app uses a trained predictive model to estimate a fair price range in real time.

The app includes:
- A price prediction model trained from used-vehicle data
- A Flask API that loads the trained model and runs inferences
- A React-powered frontend for user input and results
- User-specific prediction history stored in a PostgreSQL database
- A confidence score based on how typical the vehicle is relative to the training data
- Business logic for luxury-brand pricing and progressive market adjustments

## Why This Project Is Strong for a Resume

This project highlights the kind of end-to-end product work employers look for in a computer science student:

- AI/ML workflow: cleaned and prepared real data, engineered features, and trained a prediction model
- Full-stack development: built a web app with a server, API, frontend, and database
- Data-driven decision making: used measurable vehicle characteristics instead of static pricing rules
- Real-world engineering: implemented validation, user tracking, and persistence
- API integration: connected a trained model to a frontend and surfaced results in a user-friendly interface
- Deployment-ready structure: app is organized for local execution and production deployment

This is a strong example of being able to move from raw data to a usable software product.

## Core Features

### AI Price Prediction
The backend evaluates a vehicle using inputs such as:
- Manufacturer
- Model
- Year
- Odometer reading
- Fuel type
- Transmission

It transforms the inputs into model-ready features, runs the prediction, and returns a price estimate along with a confidence score.

### Market-Aware Pricing Logic
The app does more than output a raw model prediction. It applies additional logic to produce more realistic pricing, including:
- Luxury brand adjustments
- Luxury model boosts
- Progressive price reduction based on vehicle market behavior
- Confidence scoring based on age and mileage profiles

### Frontend Experience
A frontend UI allows users to:
- Select manufacturer and model
- Enter vehicle details
- Submit the car for a prediction
- View the price estimate and confidence score
- Browse saved vehicle history tied to their user session

### Backend Storage
Each prediction can be saved for the current user in a database, making the app feel like a real product rather than a one-off model demo.

## Tech Stack

- Python
- Flask
- React (via CDN in the static frontend)
- PostgreSQL
- scikit-learn
- pandas and NumPy
- XGBoost
- category_encoders
- joblib
- HTML/CSS

## Architecture

```mermaid
flowchart LR
    A[Vehicle input from frontend] --> B[Flask API /backend/app.py]
    B --> C[Model inference using saved ML artifacts]
    C --> D[Price prediction + confidence score]
    D --> E[Frontend result display]
    D --> F[PostgreSQL user history storage]
    G[Vehicle data cleaning and feature engineering] --> C
```

## Project Structure

```text
OdomAI/
├── backend/
│   ├── __init__.py
│   ├── app.py
│   └── db.py
├── config/
│   ├── requirements.txt
│   └── runtime.txt
├── frontend/
│   └── index.html
├── models/
│   ├── odomai_model.json
│   └── vehicles_clean.csv
├── tests/
│   └── test_app_logic.py
├── .gitignore
├── README.md
└── ...
```

## How the AI Workflow Works

1. The project loads and cleans a dataset of used vehicles.
2. Features like vehicle age and mileage-based behavior are engineered.
3. Categorical values such as manufacturer, model, fuel type, and transmission are transformed for model use.
4. A machine learning model is trained and saved with joblib.
5. The Flask app loads the model and encoder artifacts at runtime.
6. The frontend sends a request with user-entered vehicle details.
7. The API validates the request, prepares the input data, runs inference, and returns a predicted value.
8. The result is displayed in the UI and stored in the database for the user.

## Important Business Logic in the App

The prediction engine is not just a raw model output. The app includes custom logic to improve realism:

- Progressive reduction curves for higher-priced vehicles
- Luxury brand and model multipliers
- Confidence scoring based on how “mainstream” a vehicle is compared to the training distribution
- User-specific prediction history stored in a database

This is a valuable part of the project because it shows that the system is built as a product, not just a notebook experiment.

## API Endpoints

The Flask app exposes several endpoints:

- GET /health
  - Health check for the API

- GET /metadata
  - Returns supported manufacturers, models, fuel types, and transmissions

- POST /predict
  - Accepts vehicle data and returns price prediction, confidence, and input summary

- GET /cars
  - Returns the saved prediction history for a user

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd OdomAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r config/requirements.txt
```

### 4. Set up environment variables

For the database-backed features, set a PostgreSQL connection string:

```bash
export DATABASE_URL="postgresql://username:password@host:port/database_name"
```

If DATABASE_URL is not set, the app still runs for prediction functionality, but database initialization and history storage are skipped.

### 5. Run the backend

From the project root:

```bash
python backend/app.py
```

The Flask app will serve the static frontend as well.

### 6. Open the app

Open your browser to:

```text
http://localhost:5000
```

## Running Tests

The project includes logic regression tests for validation:

```bash
python -m unittest discover -s tests -v
```

## Example Use Case

A user enters:
- Make: Mercedes-Benz
- Model: C-Class
- Year: 2018
- Mileage: 82,000
- Fuel: gas
- Transmission: automatic

The app loads the trained model, transforms the features, predicts a fair price, applies luxury-adjustment logic, and returns a realistic estimated value with a confidence score.

## Resume-Friendly Project Narrative

A concise summary you could use on a resume:

> Built OdomAI, a full-stack AI-powered used car pricing application. Trained a predictive model on vehicle data, integrated it into a Flask backend, and connected it to a frontend interface for real-time price estimation. Saved user predictions in PostgreSQL to support personalized car history tracking and demonstrate end-to-end product development from data to deployment.

## Future Improvements

- Add model retraining pipeline from a CSV or API source
- Improve feature engineering with trim level, condition, and location data
- Add authentication and user accounts
- Add a comparison dashboard for multiple vehicles
- Expand to more markets and pricing regions
- Deploy the backend and frontend separately for a more production-style architecture

## Final Note

OdomAI is more than a demo: it is a realistic example of an AI product that uses data science, software engineering, and user-facing design together. It is well suited for showcasing both technical capability and product thinking on a resume or in an interview.
