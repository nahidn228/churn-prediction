import os
import json
import joblib
import pytest
import pandas as pd


@pytest.fixture(scope="session")
def models_dir():
    """Returns the absolute path to the pre-trained models directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models")


@pytest.fixture(scope="session")
def loaded_artifacts(models_dir):
    """Loads and caches all pre-trained model artifacts for testing."""
    meta_path = os.path.join(models_dir, "pipeline_metadata.json")
    assert os.path.exists(meta_path), f"Metadata file not found at {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    supervised_models = joblib.load(os.path.join(models_dir, "supervised_models.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans.joblib"))
    iso_forest = joblib.load(os.path.join(models_dir, "isolation_forest.joblib"))
    pca_2d = joblib.load(os.path.join(models_dir, "pca_2d.joblib"))
    pca_3d = joblib.load(os.path.join(models_dir, "pca_3d.joblib"))

    return {
        "metadata": metadata,
        "models": supervised_models,
        "scaler": scaler,
        "kmeans": kmeans,
        "iso_forest": iso_forest,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d
    }


@pytest.fixture
def sample_raw_customer_dict():
    """A realistic single customer profile."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.20
    }


@pytest.fixture
def sample_high_risk_customer():
    """Customer profile engineered to represent high churn risk."""
    return {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 99.65,
        "TotalCharges": 199.30
    }


@pytest.fixture
def sample_loyal_customer():
    """Customer profile engineered to represent high retention / low risk."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.80,
        "TotalCharges": 4406.40
    }


@pytest.fixture
def sample_raw_dataframe():
    """A multi-row raw customer dataset with messy edge cases."""
    return pd.DataFrame([
        {
            "customerID": "0001-TEST1",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": "29.85",
            "Churn": "No"
        },
        {
            "customerID": "0002-TEST2",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": " ",  # whitespace missing value
            "Churn": "No"
        },
        {
            "customerID": "0003-TEST3",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.70,
            "TotalCharges": "151.65",
            "Churn": "Yes"
        }
    ])
