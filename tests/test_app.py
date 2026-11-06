from fastapi.testclient import TestClient
from fraudshield.app import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_prediction():
    response = client.post("/predict", json={
        "amount": 500,
        "hour": 2,
        "international": True,
        "new_device": True,
        "previous_transactions": 1
    })
    assert response.status_code == 200
    assert 0 <= response.json()["fraud_probability"] <= 1

def test_invalid_hour():
    response = client.post("/predict", json={
        "amount": 10,
        "hour": 30,
        "previous_transactions": 5
    })
    assert response.status_code == 422
