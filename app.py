from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import string
import os
from dotenv import load_dotenv
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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

# #keywords
# model_name = "agentlans/flan-t5-small-keywords"
# keyword_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# keyword_tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    if (prediction == 0 and confidence <0.9):
        return {"is_sarcastic": True, "confidence": confidence}
    if (prediction == 1 and confidence <0.60):
        return {"is_sarcastic": False, "confidence": confidence}
    return {"is_sarcastic": bool(prediction), "confidence": confidence}


# Load the SerpApi key from environment variables
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
if not SERPAPI_KEY:
    raise ValueError("No SERPAPI_KEY found in environment variables!")

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            app.logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Read the file content for analysis
            file_content = file.read().decode('utf-8')
            app.logger.info(f"File content read successfully: {file.filename}")
            
            # Perform sentiment analysis and sarcasm detection
            sentiment_result = sentiment_task(file_content)
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            
            sarcasm_result = detect_sarcasm(file_content, sentiment_label)
            
            return jsonify({
                'message': f'File {file.filename} uploaded and analyzed successfully!',
                'sentiment': {'label': sentiment_label, 'score': sentiment_score},
                'sarcasm': sarcasm_result
            })
        else:
            app.logger.error("File type not allowed")
            return jsonify({'error': 'File not allowed. Only .txt files are accepted.'}), 400

    except Exception as e:
        app.logger.error(f"Error occurred during file upload: {e}")
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

# @app.route('/get_keywords', methods=['POST'])
# def get_keywords():
#     try:
#         data = request.json
#         text = data.get('text')
        
#         if not text:
#             return jsonify({'error': 'No text provided'}), 400

#         # Tokenize and generate keywords using the FLAN-T5 model
#         inputs = keyword_tokenizer(text, return_tensors="pt")
#         outputs = keyword_model.generate(**inputs, max_length=512)
#         decoded_output = keyword_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Process the output to get a list of keywords (split and remove duplicates)
#         keywords = list(set(decoded_output.split('||')))
        
#         # Only return the top 3 keywords
#         top_keywords = keywords[:3]

#         return jsonify({'keywords': top_keywords})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

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
