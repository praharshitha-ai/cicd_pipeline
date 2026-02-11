#backend/test/test_main.py
import pytest 
from fastapi.testclient import TestClient
from pathlib import Path


import pickle
from pathlib import Path


import numpy as np 

#import your app (adjust if you renamed it)
from  backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200 
    assert response.json() == {"status": "ok"}

def test_predict_valid_input():
    payload = {
        "area": 1200.5,
        "bedrooms": 3
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert isinstance(data["predicted_price"],(int, float))
    assert data["predicted_price"] > 0 #assuming house prices are positive 
