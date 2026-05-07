from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import re
import string

app = FastAPI()


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)



class AivsHuman(BaseModel):
    text_content: str


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


@app.get("/")
def home():
    return {
        "message": "AI vs Human Text Classifier API is Running"
    }

#
@app.post("/predict")
def predict(data: AivsHuman):

   
    cleaned_text = clean_text(data.text_content)

    text_vector = vectorizer.transform([cleaned_text])

   
    prediction = model.predict(text_vector)[0]


    if prediction == 0:
        result = "Human Written Text"
    else:
        result = "AI Generated Text"

    return {
        "prediction": result
    }