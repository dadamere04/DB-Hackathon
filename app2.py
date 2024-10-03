from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)


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


# Load the SerpApi key from environment variables
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
if not SERPAPI_KEY:
    raise ValueError("No SERPAPI_KEY found in environment variables!")

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Endpoint for handling the Google News API request
@app.route('/search_news', methods=['POST'])
def search_news():
    data = request.json
    company = data.get('company')
    
    if not company:
        return jsonify({'error': 'Company name is required'}), 400

    try:
        # SerpApi Google News API request
        url = f'https://serpapi.com/search.json?q={company}&tbm=nws&api_key={SERPAPI_KEY}'
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        news_data = response.json()

        if 'news_results' in news_data:
            return jsonify(news_data['news_results'])
        else:
            return jsonify({'error': 'No news results found'}), 404

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Request failed: {e}'}), 500

# Example sentiment and sarcasm analysis (replace this with actual logic)
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text input is required'}), 400

    # Dummy response for sentiment and sarcasm analysis
    sentiment_result = {'label': 'POSITIVE', 'score': 0.95}
    sarcasm_result = {'is_sarcastic': False}

    return jsonify({
        'sentiment': sentiment_result,
        'sarcasm': sarcasm_result
    })

if __name__ == '__main__':
    app.run(debug=True)
