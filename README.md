# Customer Churn Prediction

A telecom company wants to find the customers who are likely to leave, so the retention team can call them early. This project starts from the raw IBM Telco Customer Churn data and ends with a Decision Tree model served by a small Flask API.

**Business problem → Data → Preparation → EDA → Feature Engineering → Model → Evaluation → Interpretation → Saved Model → API**

---

## Project structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv                  # the dataset (7,043 customers)
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb       # full analysis and modelling
├── model/
│   └── churn_model.pkl          # saved pipeline: features + encoding + tree
├── churn_features.py            # the custom feature step (shared by notebook and API)
├── app.py                       # Flask API: validation + the POST /predict endpoint
├── sample_request.json          # example customer to send to the API
├── sample_response.json         # what the API sends back for that customer
├── requirements.txt
└── README.md
```

The API is only two files:

| File | Job |
|---|---|
| `app.py` | Request format, validation and the HTTP endpoints |
| `churn_features.py` | The feature step saved inside the model |

One note on `churn_features.py`: the saved model has a custom feature step inside it. joblib only stores a *link* to that class and not its code, so this file has to sit next to `app.py` or the model will not load.

---

## Setup

You need Python 3.14. The versions in `requirements.txt` were tested with it.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 1. Run the notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

Run all cells from top to bottom (*Kernel → Restart & Run All*). The notebook takes about two minutes. The two grid searches in sections 6 and 7 are the slow part, around 30 seconds together.

The last section saves the trained pipeline to `model/churn_model.pkl`. A trained model is already in the repo, so run the notebook only if you want to train it again.

---

## 2. Start the API

From the project root:

```bash
python app.py
```

It runs on `http://127.0.0.1:5000`. Check it is up with:

```bash
curl http://127.0.0.1:5000/health
```
```json
{"status": "ok", "model_loaded": true}
```

### Endpoint: `POST /predict`

Send one customer as JSON, using the same column names as the dataset. `customerID` is optional and is ignored if you send it.

**Sample request** (`sample_request.json`). This is a real customer from the dataset: 8 months in, month-to-month contract, fiber optic, pays by electronic check, no tech support or online security. This customer did leave.

```json
{
  "customerID": "9305-CDSKC",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 8,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 99.65,
  "TotalCharges": 820.5
}
```

**Call it with curl**
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

**Or with PowerShell**
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body (Get-Content sample_request.json -Raw)
```

**Sample response** (`sample_response.json`)
```json
{
  "prediction": "Yes",
  "churn_probability": 0.74
}
```

- `prediction` is `"Yes"` when `churn_probability` is 0.5 or higher.
- `churn_probability` works best as a **risk score for ranking customers**. Class weighting was tested and not used (see *Tuning, class imbalance and benchmarks* below), and the tree is not calibrated. So treat the number as a ranking score, not as a true probability.

### Input rules and error handling

| Field(s) | What's accepted |
|---|---|
| `gender` | `Female`, `Male` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `Yes`, `No` |
| `MultipleLines` | `Yes`, `No`, `No phone service` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `Yes`, `No`, `No internet service` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `SeniorCitizen` | `0` or `1` |
| `tenure` | whole number, 0 or more |
| `MonthlyCharges` | number, 0 or more |
| `TotalCharges` | number, 0 or more. Can be blank (`" "`) for a new customer, as in the original data |

If something is wrong, the API returns **HTTP 400** and lists every problem, not just the first one:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"gender\": \"Female\", \"Contract\": \"Monthly\", \"tenure\": -3}"
```
```json
{
  "error": "Invalid input",
  "details": [
    "Missing fields: Partner, Dependents, PhoneService, ...",
    "'Contract' must be one of ['Month-to-month', 'One year', 'Two year'], got 'Monthly'",
    "'tenure' must be a whole number of months (0 or more)"
  ]
}
```

Other cases:
- Body isn't JSON → `400`
- `GET /predict` → `405`
- Unknown URL → `404`
- Unexpected server error → `500`

All of these come back as JSON.

---

## What's in the analysis (short version)

**Data preparation**
- 7,043 rows, 21 columns, no duplicates.
- `TotalCharges` was stored as text because 11 rows had a blank value. All 11 were new customers (tenure 0), so I filled them with 0.
- 26.5% of customers churned, so the classes are imbalanced. I used a stratified 70:30 split with `random_state=42`.
- All cleaning, feature engineering and one-hot encoding sit inside one scikit-learn `Pipeline`, fitted on the training data only. This avoids leakage, and the API runs the same steps on new customers.

**EDA** (4 charts in the notebook). Churn is highest for:
- month-to-month contracts, where nothing ties the customer in
- fiber optic customers, the premium product
- customers in their first year, with most churn in the first few months

**Engineered features**

| Feature | How it's made | What the data shows |
|---|---|---|
| `SpendPerService` | `MonthlyCharges` ÷ number of services held (minimum 1) | churn rises 7% → 21% → 28% → 50% across quartiles |
| `AvgMonthlySpend` | `TotalCharges` ÷ `tenure`, falling back to `MonthlyCharges` for a new customer | 11% → 25% → 36% → 33% across quartiles |
| `IsAutoPay` | `PaymentMethod` contains "automatic" | 16% churn on auto-pay vs 35% otherwise |

`AvgMonthlySpend` (importance 0.027) and `SpendPerService` (0.008) make the top 10. `IsAutoPay` does not, which makes sense: it is just `PaymentMethod` squashed into two values, and the tree gets more out of splitting on `PaymentMethod` itself. All three are kept because they are easy for the business to read, not because they improve the scores.

**Models compared** (test set, 2,113 customers none of the models saw during training)

| Model | Train acc | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| DT1 - no limits (overfits) | 0.998 | 0.734 | 0.499 | 0.524 | 0.511 |
| DT2 - `max_depth=5, min_samples_split=10` | 0.804 | 0.791 | 0.608 | 0.601 | 0.604 |
| DT3 - `max_depth=10, min_samples_split=10` | 0.873 | 0.762 | 0.556 | 0.510 | 0.532 |
| **Tuned by GridSearchCV (final)** | 0.801 | **0.794** | **0.615** | **0.602** | **0.608** |

The gap between train and test accuracy shows the problem. DT1 gets 99.8% on data it has seen and 73.4% on data it has not, so it has memorised the training rows instead of learning a pattern. DT3 is twice as deep as DT2 and still scores worse. The tuned tree in the last row is the final model, and the next section explains how it was chosen.

Final model: `DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=10, min_samples_split=50, random_state=42)`. It is 5 levels deep with 32 leaves.

**Test set results (final model)**

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.794 | 0.615 | 0.602 | 0.608 | 0.828 | 0.623 |

Confusion matrix: TN 1340, FP 212, FN 223, TP 338. The model catches **338 of the 561** churners in the test set and raises 212 false alarms, so about 6 in 10 flagged customers really did leave. Precision and recall end up close to equal. Lowering `CHURN_THRESHOLD` in `app.py` swaps precision for recall without retraining, and the sweep below shows what that costs.

**Top drivers:** month-to-month contract (importance 0.514), `tenure` (0.177), fiber optic internet (0.154), then electronic check and the spend features.

---

## Tuning, class imbalance and benchmarks

**Hyperparameter tuning.** The three trees above were compared on the test set. That is not a safe way to choose a model: once the test set has helped pick the winner, it no longer tells you how the model will do on fresh data. `GridSearchCV` makes the choice again over 224 parameter combinations, scored with 5-fold cross-validation **inside the training set only**:

| | |
|---|---|
| Best parameters | `criterion='gini', max_depth=5, min_samples_leaf=10, min_samples_split=50` |
| Best cross-validated F1 | 0.586 |
| Test accuracy / F1 | 0.794 / 0.608 |

The grid only just beat the hand-picked DT2 (0.791 to 0.794 accuracy). The original settings were already close to the best this grid could find. The real gain is that the choice is now backed by cross-validation instead of being a guess that happened to work.

**Class imbalance.** Two ways of handling the 26.5% churn rate were tried:

| Approach | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| No weighting, threshold 0.5 *(deployed)* | 0.615 | 0.602 | 0.608 | 0.828 |
| `class_weight='balanced'`, threshold 0.5 | 0.459 | 0.841 | 0.594 | 0.830 |
| No weighting, threshold 0.2 | 0.451 | 0.848 | 0.589 | 0.828 |

Compare the last two rows. `class_weight='balanced'` ends up almost exactly where the plain tree already sits at a 0.2 cutoff, and the ROC-AUC is the same. So the weighting does not add anything new, it just moves the model to a different point on the same curve. The plain tree is the one deployed, and `CHURN_THRESHOLD` in `app.py` is the way to trade for more recall. The full sweep is in section 7 of the notebook.

**Other model types** (same features, same split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Decision Tree (tuned, deployed) | 0.794 | 0.615 | 0.602 | 0.608 | 0.828 | 0.623 |
| Logistic Regression | **0.811** | **0.674** | 0.560 | **0.611** | **0.845** | 0.643 |
| Random Forest | 0.781 | 0.612 | 0.478 | 0.537 | 0.821 | 0.612 |
| Gradient Boosting | 0.795 | 0.647 | 0.503 | 0.566 | 0.841 | **0.648** |

Logistic regression beats the tuned tree on most columns, and gradient boosting has the best PR-AUC. The gaps are only a few points. The tree is still the one shipped, because a five-level tree can be printed and read by the retention team while 200 forest trees cannot. That trades a little accuracy for an explanation people can actually follow.
