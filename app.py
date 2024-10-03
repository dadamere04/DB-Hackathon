from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
from transformers import pipeline
import json
from requests.exceptions import HTTPError
import time

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize the sentiment analysis pipeline with the new model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        text_input = data.get('text')
        if not text_input:
            return jsonify({'error': 'No text input provided'}), 400

        try:
            # Use the Transformers pipeline for sentiment analysis
            result = sentiment_task(text_input)

            # Log the entire response to troubleshoot unexpected formats
            app.logger.info("Transformers API Response: %s", result)

            # Check if the result is in the expected format
            if isinstance(result, list) and len(result) > 0 and 'label' in result[0]:
                sentiment = result[0]
                return jsonify({'label': sentiment['label'], 'score': sentiment['score']})
            else:
                return jsonify({'error': 'Unexpected response format from Transformers API', 'response': result}), 500

        except Exception as e:
            app.logger.error("Error occurred in /analyze_sentiment: %s", e)
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error("Error occurred in /analyze_sentiment: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
