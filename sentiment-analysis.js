// Function to get Place ID

API_KEY = "AIzaSyDHceAQWnV85MuN57COejgl1BfHCWDNaRk"
async function getPlaceId(placeName) {
    const url = `https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input=${encodeURIComponent(placeName)}&inputtype=textquery&fields=place_id&key=${API_KEY}`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (data.candidates && data.candidates.length > 0) {
            return data.candidates[0].place_id;
        } else {
            console.error('Place not found');
            return null;
        }
    } catch (error) {
        console.error('Error fetching place ID:', error);
        return null;
    }
}

// Function to get Reviews
async function getPlaceReviews(placeId) {
    const url = `https://maps.googleapis.com/maps/api/place/details/json?place_id=${placeId}&fields=name,rating,reviews&key=${API_KEY}`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (data.result && data.result.reviews) {
            return data.result.reviews;
        } else {
            console.log('No reviews found for this place.');
            return [];
        }
    } catch (error) {
        console.error('Error fetching reviews:', error);
        return [];
    }
}
