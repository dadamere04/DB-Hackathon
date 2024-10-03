from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import string
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize the sentiment analysis pipeline
sentiment_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

# Initialize sarcasm detection model and tokenizer
sarcasm_model_path = "helinivan/multilingual-sarcasm-detector"
sarcasm_tokenizer = AutoTokenizer.from_pretrained(sarcasm_model_path)
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_path)

# Preprocessing function for sarcasm detection
def preprocess_data(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

# Function for sarcasm detection
def detect_sarcasm(text,sentiment):
    preprocessed_text = preprocess_data(text)
    tokenized_text = sarcasm_tokenizer([preprocessed_text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    output = sarcasm_model(**tokenized_text)
    probs = output.logits.softmax(dim=-1).tolist()[0]
    confidence = max(probs)
    prediction = probs.index(confidence)

    print(f"og prediction: {prediction}, confidence: {confidence}")
    if (prediction == 0 and confidence <0.99):
        return {"is_sarcastic": True, "confidence": confidence}
    if (prediction == 1 and confidence <0.60):
        return {"is_sarcastic": False, "confidence": confidence}
    return {"is_sarcastic": bool(prediction), "confidence": confidence}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text_input = data.get('text')
        if not text_input:
            return jsonify({'error': 'No text input provided'}), 400

        try:
            # Sentiment Analysis
            sentiment_result = sentiment_task(text_input)
            app.logger.info("Sentiment Analysis Response: %s", sentiment_result)

            if isinstance(sentiment_result, list) and len(sentiment_result) > 0 and 'label' in sentiment_result[0]:
                sentiment_label = sentiment_result[0]['label']
                sentiment_score = sentiment_result[0]['score']
            else:
                return jsonify({'error': 'Unexpected sentiment response format'}), 500

            # Sarcasm Detection
            sarcasm_result = detect_sarcasm(text_input, sentiment_label)
            app.logger.info("Sarcasm Detection Response: %s", sarcasm_result)

            # Combine the results
            return jsonify({
                'sentiment': {'label': sentiment_label, 'score': sentiment_score},
                'sarcasm': sarcasm_result
            })

        except Exception as e:
            app.logger.error("Error occurred in /analyze_text: %s", e)
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error("Error occurred in /analyze_text: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
