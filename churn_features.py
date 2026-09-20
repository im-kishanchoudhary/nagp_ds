"""
Custom feature step for the churn model.

In its own file because joblib saves a reference to the class rather than its code,
so app.py has to import it before the saved pipeline will load.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds the three engineered features from section 3 of the notebook."""

    SERVICE_COLUMNS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        customers = X.copy()

        # blank for new customers, and both spend features below divide by it
        customers["TotalCharges"] = pd.to_numeric(customers["TotalCharges"], errors="coerce").fillna(0)

        service_count = (customers[self.SERVICE_COLUMNS] == "Yes").sum(axis=1)
        customers["SpendPerService"] = customers["MonthlyCharges"] / service_count.clip(lower=1)

        # tenure 0 means no billing history, so fall back to the current bill
        customers["AvgMonthlySpend"] = (
            customers["TotalCharges"] / customers["tenure"].where(customers["tenure"] > 0)
        ).fillna(customers["MonthlyCharges"])

        customers["IsAutoPay"] = (
            customers["PaymentMethod"].str.contains("automatic", na=False).astype(int)
        )

        return customers
