# Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)
![API](https://img.shields.io/badge/API-TMDB-fcba03)

A Flask web app that recommends movies using content-based filtering and analyzes user review sentiment with a Naive Bayes NLP model.

## Live Demo

https://movie-recommendation-system-0u9a.onrender.com/

## Features

- Movie search with autocomplete
- TMDB-powered movie details (poster, cast, rating, runtime, etc.)
- Content-based movie recommendations
- **Sentiment analysis** on user reviews (Good/Bad with emoji)
- Clickable recommended movie cards

## How It Works

1. **Recommendations** — Cosine similarity on movie metadata from `main_data.csv`
2. **Movie details** — Fetched from [TMDB API](https://www.themoviedb.org/documentation/api)
3. **Reviews** — Pulled from TMDB reviews API (IMDb scraping as fallback)
4. **Sentiment** — Multinomial Naive Bayes model trained on IMDB review dataset

## Local Setup

```bash
git clone https://github.com/kashyaprohit2005/-movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
python train_sentiment.py
python app.py
```

Open http://127.0.0.1:5000/

## Deploy on Render

1. Connect this GitHub repo to Render
2. Set environment variable: `TMDB_API_KEY` = your TMDB API key
3. Build command: `pip install -r requirements.txt && python train_sentiment.py`
4. Start command: `gunicorn app:app`

Or use the included `render.yaml` Blueprint.

## Get a TMDB API Key

1. Create an account at https://www.themoviedb.org/
2. Go to Settings → API → Request an API key
3. Use "NA" for website URL if you don't have one

## Project Structure

```
app.py              # Flask app (routes, TMDB, sentiment)
train_sentiment.py  # Trains NLP model → nlp_model.pkl, tranform.pkl
main_data.csv       # Movie dataset for recommendations
templates/          # HTML templates
static/             # CSS, JS, images
datasets/           # Raw datasets
```

## Developer

Developed by Rohit, BCA DS, 241539
