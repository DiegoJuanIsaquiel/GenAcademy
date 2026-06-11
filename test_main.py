from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "GenAcademy API está online" in response.json()["message"]

def test_read_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "embedding_model" in response.json()

def test_read_model():
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json()["status"] == "active"