from app import app
import json

def test_predict():
    client = app.test_client()
    response = client.post(
        "/predict",
        data=json.dumps({
            "annual_income": 60,
            "spending_score": 70
        }),
        content_type="application/json"
    )
    assert response.status_code == 200
    assert "cluster" in response.get_json()
