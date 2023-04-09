from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Функция проверяет исходный путь (root)
# 1 тест: проверяет статус кода. Должен быть 200
# 2 тест: проверяет тип ответа. Должен быть dict
# 3 тест: проверяет ответ на пустоту. Должен быть не пустым
# 4 тест: проверяет тип текста в ответе. Должен быть str
# 5 тест: проверяет содержимое ответа на соответствие.
# Должен быть словарь с текстом


def test_read_main_root():
    response = client.get("/")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert len(response.json()["message"]) != 0
    assert isinstance(response.json()["message"], str)
    assert response.json() == {"message": "This is neural network's main page"}

# Функция проверяет демо режим модели нейросети
# 1 тест: проверяет статус кода. Должен быть 200
# 2 тест: проверяет тип ответа. Должен быть dict
# 3 тест: проверяет ответ на пустоту. Должен быть не пустым
# 4 тест: проверяет тип сгенерированного текста в ответе. Должен быть str


def test_read_main_demo():
    response = client.get("/demo/")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert len(response.json()["generated_text"]) != 0
    assert isinstance(response.json()["generated_text"], str)

# Функция проверяет предсказание модели нейросети
# 1 тест: проверяет статус кода. Должен быть 200
# 2 тест: проверяет тип ответа. Должен быть dict
# 3 тест: проверяет ответ на пустоту. Должен быть не пустым
# 4 тест: проверяет тип сгенерированного текста в ответе. Должен быть str


def test_read_main_predict():
    response = client.post("/predict/", json={"text": "Hello world!"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert len(response.json()["generated_text"]) != 0
    assert isinstance(response.json()["generated_text"], str)

# Функция проверяет предсказание модели нейросети
# 1 тест: проверяет статус кода. Должен быть 200
# 2 тест: проверяет тип ответа. Должен быть list
# 3 тест: проверяет ответ на пустоту. Должен быть не пустым
# 4 тест: проверяет тип сгенерированного текста в ответе. Должен быть str


def test_read_main_predict_detail():
    response = client.post("/predict/detail/",
                           json={"text": "Hello world!",
                                 "max_length": 30,
                                 "num_return_sequences": 1})
    assert response.status_code == 200
    assert len(response.json()[0]) == 1
    assert isinstance(response.json(), list)
    assert len(response.json()[0]["generated_text"]) != 0
    assert isinstance(response.json()[0]["generated_text"], str)
