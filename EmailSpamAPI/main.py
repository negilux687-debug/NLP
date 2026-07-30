from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="Spam Detection API")

# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class MailRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Spam Detection API is Running"
    }


@app.post("/predict")
def predict(data: MailRequest):

    text = [data.message]

    features = vectorizer.transform(text)

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Ham"
    else:
        result = "Spam"

    return {
        "prediction": result,
        "value": int(prediction)
    }