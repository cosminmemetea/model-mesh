# tests/test_fastapi.py
from fastapi.testclient import TestClient
from api.fastapi_app import app

client = TestClient(app)

def test_predict_sentiment():
    response = client.post("/predict", json={"text": "I am happy", "model": "bert"})
    assert response.status_code == 200
    assert "result" in response.json()
