from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()


model = pickle.load(open("logistic_model.pkl", "rb"))

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

encoder = pickle.load(open("encoder.pkl", "rb"))


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Emotion Detection API is running"
    }


@app.post("/predict")
def predict(data: TextInput):

    text_vector = vectorizer.transform([data.text])

    prediction = model.predict(text_vector)[0]

    emotion = encoder.inverse_transform([prediction])[0]

    return {
        "input": data.text,
        "prediction": emotion,
        "value": int(prediction)
    }