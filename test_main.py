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
    #так как seed в нейросети одно и тоже, то ответ на запрос всегда один и тот же
    assert len(response.json()["generated_text"]) == 217
    assert response.json()["generated_text"] == "Hello world! We're sorry to announce that you'll never miss our latest episode of the podcast again. Just follow us on Twitter here. For our upcoming podcast – live and free!\n\nDownload our show!\n\nWe'll also be sending"