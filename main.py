from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Initialize Hugging Face sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Define the input schema
class TextInput(BaseModel):
    text: str

# Define the sentiment analysis endpoint
@app.post("/analyze-sentiment")
async def analyze_sentiment(input: TextInput):
    # Perform sentiment analysis on the input text
    result = sentiment_pipeline(input.text)
    return {"label": result[0]['label'], "score": result[0]['score']}

# To run this FastAPI backend, use: uvicorn main:app --reload
