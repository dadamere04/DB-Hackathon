const express = require('express');
const axios = require('axios');
const app = express();
const port = 3000;

// Google API Key (replace with your key)
const API_KEY = env.GOOGLE_API_KEY;

// Endpoint to get reviews from Google Places API
app.get('/get-reviews', async (req, res) => {
    const placeId = req.query.placeId;

    try {
        const response = await axios.get(`https://maps.googleapis.com/maps/api/place/details/json`, {
            params: {
                place_id: placeId,
                fields: 'name,rating,reviews',
                key: API_KEY
            }
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error fetching reviews:', error);
        res.status(500).send('Error fetching reviews');
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
