import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

def test_predict():
    client = app.test_client()
    response = client.post("/predict", json={
        "annual_income": 60,
        "spending_score": 50
    })

    assert response.status_code == 200
    assert "cluster" in response.json
