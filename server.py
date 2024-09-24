import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
# Ensuring that all reviews have a 'sentiment' key
for review in reviews:
    if 'sentiment' not in review:
        review['sentiment'] = sia.polarity_scores(review['ReviewBody'])

# Valid locations from the README
VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def filter_reviews(self, location, start_date, end_date):
        filtered_reviews = reviews

        if location:
            location = location[0]
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
        
        if start_date:
            start_date = datetime.strptime(start_date[0], "%Y-%m-%d")
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date]

        if end_date:
            end_date = datetime.strptime(end_date[0], "%Y-%m-%d")
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date]

        return sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        global reviews

        if environ["REQUEST_METHOD"] == "GET":
            query_str = environ.get("QUERY_STRING")
            location = parse_qs(query_str).get("location")
            start_date = parse_qs(query_str).get("start_date")
            end_date = parse_qs(query_str).get("end_date")

            filtered_reviews = self.filter_reviews(location, start_date, end_date)
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            cont_len = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(cont_len).decode("utf-8")

            location = parse_qs(body).get("Location", [""])[0]
            review_body = parse_qs(body).get("ReviewBody", [""])[0]

            if not location or not review_body or location not in VALID_LOCATIONS:
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", "0")
                ])
                return [b'']

            sentiment = self.analyze_sentiment(review_body)

            new_rev = {
                "ReviewId": str(uuid.uuid4()),
                "Location": location,
                "ReviewBody": review_body,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment
            }

            reviews.append(new_rev)
            response_body = json.dumps(new_rev, indent=2).encode("utf-8")

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()