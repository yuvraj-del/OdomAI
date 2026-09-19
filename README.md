# OdomAI

**Beyond the odometer. Know your car's true value.**

An end-to-end machine learning product: raw used-car listings go in one end, and a live web app that prices any car in seconds comes out the other. I cleaned the data with pandas, trained and evaluated an XGBoost model, served it through a Flask API, built a React front end, and stored each visitor's history in PostgreSQL.

[![Live demo](https://img.shields.io/badge/live%20demo-odomai.onrender.com-534AB7?style=for-the-badge)](https://odomai.onrender.com)


![OdomAI screenshot](OdomAI_SS.png)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-SQL-4169E1?logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-model-EB5E28)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)

---

## Results at a glance

| | |
|---|---|
| **Training data** | 100,000 cleaned listings, 36 manufacturers, 830 distinct models, model years 1995-2022 |
| **Typical error** | **$2,949** mean absolute error, **16%** median error on 20,000 held-out listings |
| **Explained variance** | **R² = 0.82** |
| **vs. a naive baseline** | Predicting each model's median price gives $6,432 MAE, so the model **cuts the error by 54%** |
| **Stability** | 5-fold cross-validation: MAE **$2,927 ± 28**, R² 0.824 ± 0.009. Not a lucky split |
| **Tests** | 9 unit tests covering the pricing rules and confidence score |

---

## What I built

| Layer | What I did | Tools |
|---|---|---|
| **Data** | Turned messy classified ads into a clean modelling table: filtered impossible prices, mileage and years, normalized text, fixed listings filed under the wrong make, and collapsed hundreds of spelling variants of model names with a regex pipeline | pandas, NumPy, regex |
| **Model** | Engineered features, split the data *before* encoding to prevent leakage, target-encoded 830 models, trained on log-price, and evaluated against a baseline with cross-validation | scikit-learn, category_encoders, XGBoost |
| **API** | Designed a small REST API with strict input validation, model-alias handling, and business rules layered on top of the raw prediction | Flask, joblib |
| **Database** | Designed the schema, wrote an idempotent migration, and stored anonymous per-visitor history | PostgreSQL, psycopg2 |
| **Front end** | Built a responsive single-page app with live validation, an SVG confidence gauge, and a history view | React 18 |

```mermaid
flowchart LR
    subgraph Offline["Offline: data to model"]
        A["Raw listings"] --> B["pandas: clean and<br/>engineer features"]
        B --> C["Target encoding +<br/>XGBoost on log-price"]
        C --> D[("Saved model<br/>artifacts")]
    end

    subgraph Online["Online: request to prediction"]
        U["React UI"] -- "POST /predict" --> F["Flask API"]
        F --> V["Validate and<br/>normalize input"]
        V --> M["Model estimate"]
        M --> R["Pricing rules +<br/>confidence score"]
        R --> U
        R --> P[("PostgreSQL<br/>history")]
        U -- "GET /cars" --> P
    end

    D -. "loaded at startup" .-> M
```

---

## The model

### Data
The training data comes from the public [Craigslist Cars & Trucks dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data). Raw classified listings are noisy, so most of the accuracy is decided in the cleaning:

- Kept only listings with all seven needed fields, with prices of $1k-$120k, 1k-300k miles, and model years 1995 and later.
- Lower-cased and trimmed all text, and removed motorcycle listings (Harley-Davidson).
- Re-assigned iconic models (Camry, Civic, F-150, Wrangler, ...) that were filed under the wrong manufacturer.
- Standardized model names with an ordered regex pipeline, so variants like `f150`, `f 150` and `F-150 XLT 4x4` collapse into one model, and body-style and drivetrain words are stripped.
- Removed duplicates and models with fewer than 30 listings, so every model has enough examples for stable statistics, then sampled 100k rows with a fixed seed.

### Features and design choices

| Choice | Why |
|---|---|
| `age` and `miles_per_year` | More comparable than a raw year. Separates a 5-year-old car with 60k miles (normal) from one with 150k (heavily used) |
| **Log-transformed target** | Prices are heavily right-skewed. Training on `log(1 + price)` makes the model minimize *percentage* error, so a miss on a $6k sedan counts as much as a proportional miss on a $30k truck |
| **Target encoding** | `model` has 830 values. One-hot encoding would add 830 sparse columns; target encoding gives one numeric column, with smoothing to pull rarely-seen models toward the global average |
| **Split before encoding** | The encoder is fitted on the training set only. Fitting it on all the data would leak test prices into the features |
| **XGBoost** | 400 trees, depth 6, learning rate 0.05, 80% row and column subsampling |

Age is the strongest signal (about 45% of feature importance in a retrain with the same settings), followed by model name (about 19%) and mileage (about 15%).

### How accurate is it, really?

Overall numbers are in the table at the top. Breaking the held-out results down shows where it is strong and where it is not:

| Segment (held-out listings) | Within ±20% | Median error |
|---|---|---|
| Cars 8 years old or newer | 69% | 12.7% |
| Cars priced $20k-$40k | 69% | 13.1% |
| Cars priced $10k-$20k | 63% | 15.0% |
| Models with fewer than 120 listings | 53% | 18.5% |
| Cars over 20 years old | 40% | 25.6% |
| Cars priced under $5k | 40% | 26.4% |

It is weakest at the extremes: very cheap cars, where a small dollar miss is a large percentage, and very old cars. I kept this breakdown in the README on purpose, because knowing where a model fails matters as much as its headline score.

---

## The API

Flask serves both the API and the static front end.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /metadata` | Supported manufacturers, models, fuel types and transmissions (drives the UI dropdowns) |
| `POST /predict` | Validate a car, return the price estimate and confidence, and save it to the visitor's history |
| `GET /cars` | The visitor's saved predictions, newest first |

```bash
curl -X POST https://odomai.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"manufacturer": "toyota", "model": "camry", "fuel": "gas",
       "transmission": "automatic", "year": 2015, "odometer": 80000}'
```

```json
{
  "predicted_price": 12655.69,
  "confidence": 82,
  "adjustment_multiplier": 1.0,
  "input": { "manufacturer": "toyota", "model": "camry", "fuel": "gas",
             "transmission": "automatic", "year": 2015, "odometer": 80000.0 }
}
```

**Engineering details**
- **Strict validation** with clear 400 errors: required fields, integer year, mileage range, allowed fuel types and transmissions, and known manufacturer/model pairs. Inputs are limited to what the model can reasonably handle.
- **Manufacturer aliasing**: `Mercedes Benz`, `mercedes-benz` and `mercedes` all resolve to the same brand.
- **Whole-word matching** for model rules, so a rule for `rs` does not fire on `versa` or `traverse` (covered by a regression test).
- **Graceful degradation**: if the database is unavailable, the prediction is still returned and only the history save is skipped.
- **Pricing rules on top of the model**: per-brand luxury multipliers, an extra cut for flagship trims, and a progressive reduction that grows from 0% at $15k to 30% at $25k, up to a 48% cap. For example, a 2016 BMW X5 with 70,000 miles gets a combined multiplier of 0.667.

---

## The database

PostgreSQL stores each anonymous visitor's history. Identity is a random ID in an `HttpOnly`, `Secure`, `SameSite=None` cookie, so no sign-up is needed.

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY
);

CREATE TABLE cars (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    make            TEXT NOT NULL,
    model           TEXT NOT NULL,
    year            INTEGER NOT NULL,
    mileage         INTEGER NOT NULL,
    condition       TEXT,
    predicted_price REAL NOT NULL,
    confidence      INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- Tables are created on startup with `CREATE TABLE IF NOT EXISTS`, and later schema changes ship as idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migrations, so redeploys are safe.
- The history view recomputes confidence with the current formula, so a saved car never disagrees with what the prediction page showed.
- If `DATABASE_URL` is not set, the app still runs and predicts; only history is disabled.

---

## The front end

A responsive React 18 single-page app (loaded from a CDN, no build step):

- Dependent dropdowns (models filter by manufacturer), populated from the API.
- Live inline validation of year and mileage, and a submit button that only enables when the form is valid (the server validates again).
- An SVG **confidence gauge** with a needle and a color gradient, plus a price card.
- A **My Cars** history tab with saved date, price and confidence.
- Layout adapts to phones.

---

## Testing

```bash
python -m unittest discover -s tests -v
```

The suite covers the luxury-brand normalization, whole-word model matching, the progressive price curve, and the confidence score's behavior: it falls with age and mileage, is lower for cars newer than the data and for rare models, and always stays within its limits.

---

## Run it locally

Requires Python 3.12.

```bash
git clone https://github.com/yuvraj-del/OdomAI.git
cd OdomAI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r config/requirements.txt
python -m backend.app
```

Open **http://localhost:5000**. Predictions work right away.

To enable saved history, create a `.env` file in the project root with a PostgreSQL connection string:

```text
DATABASE_URL=postgresql://user:password@localhost:5432/odomai
```

---

## Project structure

```text
OdomAI/
├── backend/
│   ├── app.py            # Flask API: validation, model inference, pricing rules, confidence
│   └── db.py             # PostgreSQL schema, migrations and queries
├── frontend/
│   └── index.html        # React single-page app
├── models/
│   ├── odomai_model.pkl      # trained XGBoost model
│   ├── odomai_model.json     # same model in XGBoost's portable format
│   ├── target_encoder.pkl    # fitted target encoder
│   ├── model_columns.pkl     # feature order the model expects
│   └── vehicles_clean.csv    # cleaned dataset (catalog for dropdowns and model counts)
├── tests/
│   └── test_app_logic.py
└── config/
    ├── requirements.txt
    └── runtime.txt
```

---

## Known limitations

I would rather state these than have them found:

- **Asking prices, not sale prices.** The data comes from classified ads, so "accuracy" means agreement with listing prices. There is no ground truth for what cars actually sell for.
- **Limited coverage of new cars.** The training data runs through model year 2022. The app accepts years up to 2026 but shows a lower confidence for them, because those estimates are extrapolation.
- **Pricing rules are hand-set.** The luxury and progressive adjustments are deliberate business rules, not learned from data. On held-out listings they place the displayed price about 13% below the listing price on median.
- **Not strictly monotonic.** Tree models can produce small bumps, so a slightly older car occasionally gets a slightly higher estimate than a newer one.
- **No accounts.** History is tied to a browser cookie, not a user login.

## What I would do next

- Train on completed sales data to estimate real transaction prices.
- Add trim level, condition and location as features.
- Replace the hand-set pricing rules with quantile regression, to return a real price range.
- Move the front end to a Vite build with a component structure and end-to-end tests.
- Add user accounts and a side-by-side comparison of saved cars.

---

## Skills demonstrated

**Data and ML:** pandas data cleaning and feature engineering, regex text normalization, leakage-safe train/test splitting, target encoding, gradient boosting, cross-validation, baseline comparison, and error analysis by segment.

**Backend:** REST API design, input validation, error handling, environment-based configuration, and unit testing.

**Database:** relational schema design, foreign keys, parameterized queries, and idempotent migrations in PostgreSQL.

**Front end:** React state management, controlled forms with live validation, SVG data visualization, and responsive layout.
