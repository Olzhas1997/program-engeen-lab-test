from transformers import pipeline, set_seed
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
generator = pipeline('text-generation', model='gpt2')
set_seed(42)


@app.get("/")
def root():
    return {"message": "This is neural network's main page"}


@app.get("/demo/")
def demo():
    return generator("This is demo")[0]


@app.post("/predict/")
def predict(item: Item):
    set_seed(42)
    return generator(item.text)[0]
