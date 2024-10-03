import React, { useState } from 'react';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:3000/get-reviews'; // Your backend server

function App() {
  const [placeName, setPlaceName] = useState('');
  const [reviews, setReviews] = useState([]);
  const [sentiments, setSentiments] = useState([]);

  // Initialize Hugging Face sentiment pipeline
  const initSentimentModel = async () => {
    const sentimentPipeline = await window.transformers.pipeline('sentiment-analysis');
    return sentimentPipeline;
  };

  // Function to fetch place ID using Google Places API (through backend)
  const getPlaceId = async () => {
    const response = await axios.get(`https://maps.googleapis.com/maps/api/place/findplacefromtext/json`, {
      params: {
        input: placeName,
        inputtype: 'textquery',
        fields: 'place_id',
        key: 'YOUR_GOOGLE_API_KEY', // Replace with your key
      },
    });

    if (response.data.candidates.length > 0) {
      return response.data.candidates[0].place_id;
    } else {
      alert('Place not found!');
      return null;
    }
  };

  // Function to fetch reviews from backend and perform sentiment analysis
  const getPlaceReviews = async () => {
    const placeId = await getPlaceId();

    if (!placeId) return;

    try {
      const response = await axios.get(`${BACKEND_URL}?placeId=${placeId}`);
      const reviewsData = response.data.result.reviews;

      setReviews(reviewsData);

      // Initialize sentiment model and perform sentiment analysis on reviews
      const sentimentPipeline = await initSentimentModel();
      const sentimentsData = await Promise.all(
        reviewsData.map((review) => sentimentPipeline(review.text))
      );

      setSentiments(sentimentsData);
    } catch (error) {
      console.error('Error fetching reviews:', error);
    }
  };

  return (
    <div className="App">
      <h1>Google Reviews Sentiment Analysis</h1>
      <input
        type="text"
        value={placeName}
        onChange={(e) => setPlaceName(e.target.value)}
        placeholder="Enter place name"
      />
      <button onClick={getPlaceReviews}>Get Reviews and Analyze Sentiment</button>

      <div>
        <h2>Reviews:</h2>
        {reviews.length > 0 ? (
          reviews.map((review, index) => (
            <div key={index}>
              <p><strong>Author:</strong> {review.author_name}</p>
              <p><strong>Review:</strong> {review.text}</p>
              <p><strong>Rating:</strong> {review.rating}</p>
              <p><strong>Sentiment:</strong> {sentiments[index] ? sentiments[index][0].label : 'Analyzing...'}</p>
              <hr />
            </div>
          ))
        ) : (
          <p>No reviews available</p>
        )}
      </div>
    </div>
  );
}

export default App;
