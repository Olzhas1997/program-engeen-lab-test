from transformers import pipeline, set_seed
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    text: str


class ItemDetail(Item):
    max_length: int = 30
    num_return_sequences: int = 1

# Модель нейросети, способная сгенерировать текст. Принимает:
# Либо какое-то начало текста и пишет продолжение
# Либо пишет текст с нуля


app = FastAPI()
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Функция отрабатывающаяся на корневом пути
# Возвращает словарь с заранее определенным текстом


@app.get("/")
def root():
    return {"message": "This is neural network's main page"}

# Функция отрабатывающаяся на пути /demo/
# Исходя из названия модель будет генерировать текст по готовому началу


@app.get("/demo/")
def demo():
    return generator("This is demo")[0]

# Функция отрабатывающаяся на пути /predict/
# Модель генерирует текст по началу введенному пользователем


@app.post("/predict/")
def predict(item: Item):
    set_seed(42)
    return generator(item.text)[0]

# Функция отрабатывающаяся на пути /predict/detail/
# Функция принимает:
# - Макс. количество символов
# - Макс. количество сгенерированных ответов
# Модель генерирует текст по началу введенному пользователем


@app.post("/predict/detail/")
def predict_detail(item: ItemDetail):
    set_seed(42)
    return generator(item.text,
                     max_length=item.max_length,
                     num_return_sequences=item.num_return_sequences)
