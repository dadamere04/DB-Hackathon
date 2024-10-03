import React, { useState } from 'react';

function App() {
  const [inputText, setInputText] = useState('');
  const [sentiment, setSentiment] = useState(null);

  // Initialize Hugging Face sentiment pipeline
  const initSentimentModel = async () => {
    const sentimentPipeline = await window.transformers.pipeline('sentiment-analysis');
    return sentimentPipeline;
  };

  // Function to perform sentiment analysis on the input text
  const analyzeSentiment = async () => {
    if (!inputText) {
      alert('Please enter some text');
      return;
    }

    try {
      // Initialize the Hugging Face sentiment model
      const sentimentPipeline = await initSentimentModel();

      // Perform sentiment analysis on the input text
      const sentimentData = await sentimentPipeline(inputText);

      // Set the sentiment result
      setSentiment(sentimentData[0]);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
    }
  };

  return (
    <div className="App">
      <h1>Sentiment Analysis Tool</h1>

      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter text to analyze sentiment"
        rows="5"
        cols="50"
      ></textarea>
      <br />

      <button onClick={analyzeSentiment}>Analyze Sentiment</button>

      <div>
        {sentiment ? (
          <div>
            <h2>Sentiment Analysis Result:</h2>
            <p><strong>Label:</strong> {sentiment.label}</p>
            <p><strong>Confidence:</strong> {Math.round(sentiment.score * 100)}%</p>
          </div>
        ) : (
          <p>Enter some text and click "Analyze Sentiment" to see the result.</p>
        )}
      </div>
    </div>
  );
}

export default App;
