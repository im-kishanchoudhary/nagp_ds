"""
Flask API for the churn model.

Run with:  python app.py
Then POST customer details as JSON to http://127.0.0.1:5000/predict
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# looks unused, but joblib needs it to unpickle the saved pipeline
from churn_features import ChurnFeatureAdder  # noqa: F401

MODEL_PATH = Path(__file__).parent / "model" / "churn_model.pkl"

# lower this for more recall, at the cost of more false alarms.
# section 7 of the notebook sweeps the cutoff: 0.3 lifts recall from 0.60 to 0.70,
# and drops precision from 0.62 to 0.53.
CHURN_THRESHOLD = 0.5

# Allowed values from the data dictionary. Checked here because the encoder ignores
# unknown categories, so a typo like "Month to month" would still return a prediction.
CATEGORICAL_FIELDS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

# numeric columns and the message used when a value fails its check
NUMERIC_FIELDS = {
    "SeniorCitizen": "must be 0 or 1",
    "tenure": "must be a whole number of months (0 or more)",
    "MonthlyCharges": "must be a number (0 or more)",
    "TotalCharges": "must be a number (0 or more), or blank for a new customer",
}

REQUIRED_FIELDS = list(CATEGORICAL_FIELDS) + list(NUMERIC_FIELDS)


def validate_customer(data: dict) -> list[str]:
    """Return every problem found. An empty list means the input is fine."""
    errors = []

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        errors.append(f"Missing fields: {', '.join(missing)}")

    for field, allowed in CATEGORICAL_FIELDS.items():
        if field in data and data[field] not in allowed:
            errors.append(f"'{field}' must be one of {allowed}, got {data[field]!r}")

    for field, message in NUMERIC_FIELDS.items():
        # a new customer has no TotalCharges yet, so blank is allowed
        if field not in data or (field == "TotalCharges" and data[field] in ("", " ", None)):
            continue
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            value = None
        if (value is None
                or isinstance(data[field], bool)   # a bool would pass float() as 1
                or value < 0
                or (field == "SeniorCitizen" and value not in (0, 1))
                or (field == "tenure" and not value.is_integer())):
            errors.append(f"'{field}' {message}")

    return errors


app = Flask(__name__)
app.json.sort_keys = False   # keep "prediction" first in the response

model = joblib.load(MODEL_PATH)   # load once at startup, reuse for every request


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)   # returns None on bad JSON instead of raising
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    errors = validate_customer(data)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    # keep only the trained columns; extras like customerID are dropped
    row = pd.DataFrame([{field: data[field] for field in REQUIRED_FIELDS}])
    probability = float(model.predict_proba(row)[0, 1])

    return jsonify({
        "prediction": "Yes" if probability >= CHURN_THRESHOLD else "No",
        "churn_probability": round(probability, 2),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.errorhandler(404)
@app.errorhandler(405)
def http_error(err):
    hints = {404: "Endpoint not found. Use POST /predict",
             405: "Method not allowed. /predict only accepts POST"}
    return jsonify({"error": hints[err.code]}), err.code


@app.errorhandler(Exception)
def unexpected_error(err):
    # always hand back JSON, never Flask's default HTML error page
    app.logger.exception(err)
    return jsonify({"error": "Something went wrong while making the prediction"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
