from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
from requests.exceptions import HTTPError
import time

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS
hf = InferenceClient(token=os.getenv("HUGGINGFACE_API_KEY"))

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

        retries = 3
        for i in range(retries):
            try:
                # Use the Hugging Face API for text classification (Sentiment analysis)
                result = hf.post(model="distilbert-base-uncased-finetuned-sst-2-english", data={"inputs": text_input})

                # If the response is in bytes, decode it
                if isinstance(result, bytes):
                    result = result.decode('utf-8')

                # Convert the result to JSON
                result_json = json.loads(result)

                # Log the entire response to troubleshoot unexpected formats
                app.logger.info("Hugging Face API Response: %s", result_json)

                # Handle the nested list response
                if isinstance(result_json, list) and len(result_json) > 0 and isinstance(result_json[0], list):
                    sentiment_scores = result_json[0]

                    # Find the sentiment with the highest score
                    highest_sentiment = max(sentiment_scores, key=lambda x: x['score'])

                    return jsonify({'label': highest_sentiment['label'], 'score': highest_sentiment['score']})
                else:
                    return jsonify({'error': 'Unexpected response format from Hugging Face API', 'response': result_json}), 500

            except HTTPError as http_err:
                if http_err.response.status_code == 500:
                    if i < retries - 1:
                        wait_time = 2 ** i
                        app.logger.warning(f"Server error, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        return jsonify({'error': 'Server error, please try again later.'}), 500
                elif http_err.response.status_code == 429:
                    if i < retries - 1:
                        wait_time = 2 ** i
                        app.logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        return jsonify({'error': 'Rate limit exceeded, please try again later.'}), 429
                else:
                    return jsonify({'error': str(http_err)}), http_err.response.status_code
            except Exception as e:
                app.logger.error("Error occurred in /analyze_sentiment: %s", e)
                return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error("Error occurred in /analyze_sentiment: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
