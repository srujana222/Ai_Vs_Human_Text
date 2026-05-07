from fastapi import FastAPI, Form
import pickle
from pydantic import BaseModel

app = FastAPI( )

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class AivsHuman(BaseModel):
   text_content: str

@app.get("/")
def home():
    return {
        "message": "AI vs Human Text Classifier API is Running"
    }

@app.post("/predict")
def predict(text: str = Form(...)):

    text_vector = vectorizer.transform([text_content])
    result = model.predict(text_vector)[0]

    if prediction == 0:
        result = "Human Written Text"
    else:
        result = "AI Generated Text"

    return {
        "prediction": result
    }