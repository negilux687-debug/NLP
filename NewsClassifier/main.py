from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model files
model = pickle.load(open("news_classifier.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))


class News(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
def predict(data: News):

    vector = tfidf.transform([data.text])

    result = model.predict(vector)

    category = encoder.inverse_transform(result)[0]

    return {
        "input":data.text,
        "category": category
    }