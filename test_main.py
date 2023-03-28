from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "This is neural network's main page"}

def test_read_main_demo():
    response = client.get("/demo/")
    assert response.status_code == 200

def test_read_main_predict():
    response = client.post("/predict/", json={"text":"Hello world!"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert isinstance(response.json()["generated_text"], str)