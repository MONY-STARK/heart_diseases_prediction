import pytest
import joblib
from fastapi.testclient import TestClient
from pathlib import Path
from app import app, load_model # adjust if your app path differs


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def model():
    return load_model("saved_models/base_logistic_model.pkl")


@pytest.fixture
def valid_input():
    return {
        "male": 1,
        "age": 50,
        "currentSmoker": 1,
        "cigsPerDay": 10,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "totChol": 240,
        "sysBP": 140,
        "diaBP": 90,
        "BMI": 27.5,
        "heartRate": 75,
        "glucose": 85
    }
