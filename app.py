import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import os
import concurrent.futures
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bs4 as bs
import pickle
import requests
import google.generativeai as genai  # <-- Gemini library

# Load Sentiment Analysis Model & Vectorizer (lazy, with fallback training)
SENTIMENT_MODEL = None
SENTIMENT_VECTORIZER = None

def load_sentiment_models():
    global SENTIMENT_MODEL, SENTIMENT_VECTORIZER
    if SENTIMENT_MODEL is not None and SENTIMENT_VECTORIZER is not None:
        return SENTIMENT_MODEL, SENTIMENT_VECTORIZER

    model_path = 'nlp_model.pkl'
    vectorizer_path = 'tranform.pkl'
    try:
        SENTIMENT_MODEL = pickle.load(open(model_path, 'rb'))
        SENTIMENT_VECTORIZER = pickle.load(open(vectorizer_path, 'rb'))
    except Exception as exc:
        print(f"Sentiment model load failed ({exc}). Retraining sentiment model...")
        from train_sentiment import train_and_save
        SENTIMENT_MODEL, SENTIMENT_VECTORIZER = train_and_save()
    return SENTIMENT_MODEL, SENTIMENT_VECTORIZER

def analyze_sentiment(review_text):
    clf, vectorizer = load_sentiment_models()
    vector = vectorizer.transform(np.array([review_text]))
    pred = clf.predict(vector)
    return 'Good' if pred[0] == 1 else 'Bad'

# TMDB API Key (set TMDB_API_KEY in Render environment variables)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "your api key here")

DATA = None
COUNT_MATRIX = None

def get_data_and_matrix():
    global DATA, COUNT_MATRIX
    if DATA is None or COUNT_MATRIX is None:
        DATA = pd.read_csv('main_data.csv')
        DATA['movie_title'] = DATA['movie_title'].astype(str).str.lower().str.strip()
        DATA['comb'] = DATA['comb'].fillna('')
        cv = CountVectorizer()
        COUNT_MATRIX = cv.fit_transform(DATA['comb'])
    return DATA, COUNT_MATRIX

def rcmd(m):
    m = str(m).lower().strip()
    data, count_matrix = get_data_and_matrix()
    
    # Exact or substring match in dataset
    match = data.loc[data['movie_title'] == m]
    if match.empty:
        match = data[data['movie_title'].str.contains(m, case=False, regex=False, na=False)]
    
    if match.empty:
        return None
    
    i = match.index[0]
    similarity_scores = cosine_similarity(count_matrix[i], count_matrix).flatten()
    lst = list(enumerate(similarity_scores))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]
    return [data['movie_title'].iloc[item[0]] for item in lst]

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].astype(str).str.capitalize())

def fetch_single_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {'api_key': TMDB_API_KEY, 'query': title}
        resp = requests.get(url, params=params, timeout=3).json()
        if resp.get('results') and len(resp['results']) > 0 and resp['results'][0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/original{resp['results'][0]['poster_path']}"
    except Exception:
        pass
    return "/static/movie_placeholder.jpeg"

def fetch_person_bio(cast_id):
    try:
        url = f"https://api.themoviedb.org/3/person/{cast_id}"
        p = requests.get(url, params={'api_key': TMDB_API_KEY}, timeout=3).json()
        bdy = p.get('birthday', 'N/A')
        if bdy and bdy != 'N/A':
            bdy = pd.to_datetime(bdy).strftime('%b %d, %Y')
        return {
            'bdy': bdy or 'N/A',
            'bio': p.get('biography', 'N/A') or 'N/A',
            'place': p.get('place_of_birth', 'N/A') or 'N/A'
        }
    except Exception:
        return {'bdy': 'N/A', 'bio': 'N/A', 'place': 'N/A'}

def fetch_tmdb_reviews(movie_id, limit=8):
    """Fetch user reviews from TMDB API (reliable on cloud hosts)."""
    reviews = []
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
        resp = requests.get(url, params={'api_key': TMDB_API_KEY}, timeout=5).json()
        for item in resp.get('results', [])[:limit]:
            content = (item.get('content') or '').strip()
            if content:
                author = item.get('author', 'Anonymous')
                reviews.append(f"{content}\n— {author}")
    except Exception as exc:
        print(f"TMDB Reviews Error: {exc}")
    return reviews

def fetch_imdb_reviews(imdb_id, limit=8):
    """Fallback: scrape IMDb reviews when TMDB has none."""
    reviews = []
    if not imdb_id or imdb_id == 'N/A':
        return reviews
    try:
        imdb_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        imdb_resp = requests.get(
            f'https://www.imdb.com/title/{imdb_id}/reviews',
            headers=imdb_headers,
            timeout=8,
        )
        if imdb_resp.status_code != 200:
            return reviews

        soup = bs.BeautifulSoup(imdb_resp.text, 'lxml')
        selectors = [
            'div[data-testid="review-text"]',
            'div.ipc-html-content-inner-div',
            'div.text.show-more__control',
        ]
        for selector in selectors:
            for node in soup.select(selector):
                text = node.get_text(strip=True)
                if text and len(text) > 20:
                    reviews.append(text)
                if len(reviews) >= limit:
                    return reviews[:limit]
    except Exception as exc:
        print(f"IMDb Scraping Error: {exc}")
    return reviews[:limit]

def get_reviews_with_sentiment(movie_id, imdb_id):
    """Fetch reviews and run sentiment analysis on each."""
    reviews_list = fetch_tmdb_reviews(movie_id)
    if not reviews_list:
        reviews_list = fetch_imdb_reviews(imdb_id)

    movie_reviews = {}
    for review_text in reviews_list:
        try:
            movie_reviews[review_text] = analyze_sentiment(review_text)
        except Exception as exc:
            print(f"Sentiment analysis error: {exc}")
            movie_reviews[review_text] = 'Unknown'
    return movie_reviews

app = Flask(__name__)

# --- DEVELOPER INFO ---
DEV_INFO = "Developed by Rohit, BCA DS, 241539"

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions, developer_info=DEV_INFO)

@app.route("/get_all_movie_data", methods=["POST"])
def get_all_movie_data():
    movie_title_input = request.form.get('name', '').strip()
    if not movie_title_input:
        return jsonify({'status': 'fail', 'message': 'Please enter a movie title!'}), 400

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # 1. Search Movie on TMDB using URL params
    search_url = "https://api.themoviedb.org/3/search/movie"
    target_movie = None
    try:
        search_res = requests.get(search_url, params={'api_key': TMDB_API_KEY, 'query': movie_title_input}, headers=headers, timeout=5).json()
        if search_res.get('results') and len(search_res['results']) > 0:
            target_movie = search_res['results'][0]
    except Exception as e:
        print("TMDB Search Error:", str(e))

    # 2. Movie Similarity Calculation
    search_title = target_movie.get('title') if target_movie else movie_title_input
    rec_movies = rcmd(search_title)
    if not rec_movies:
        rec_movies = rcmd(movie_title_input)
    
    # If still not found, fallback to top 10 from dataset
    if not rec_movies:
        data, _ = get_data_and_matrix()
        rec_movies = list(data['movie_title'].head(10))

    # If TMDB search didn't find the exact movie, attempt search with the first recommended title
    if not target_movie:
        try:
            fallback_res = requests.get(search_url, params={'api_key': TMDB_API_KEY, 'query': rec_movies[0]}, headers=headers, timeout=5).json()
            if fallback_res.get('results') and len(fallback_res['results']) > 0:
                target_movie = fallback_res['results'][0]
        except Exception:
            pass

    if not target_movie:
        return jsonify({'status': 'fail', 'message': 'Sorry! The movie you requested was not found. Please try another movie title.'}), 404

    movie_id = target_movie['id']
    original_title = target_movie.get('title') or target_movie.get('original_title')

    # 3. Parallel fetch: Movie Details, Credits, and 10 Recommended Posters
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_details = executor.submit(lambda: requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={'api_key': TMDB_API_KEY}, headers=headers, timeout=4).json())
        future_credits = executor.submit(lambda: requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits", params={'api_key': TMDB_API_KEY}, headers=headers, timeout=4).json())
        poster_futures = [executor.submit(fetch_single_poster, m) for m in rec_movies]

        movie_details = future_details.result()
        movie_credits = future_credits.result()
        rec_posters = [f.result() for f in poster_futures]

    # Process Movie Info
    imdb_id = movie_details.get('imdb_id', 'N/A')
    poster = f"https://image.tmdb.org/t/p/original{movie_details.get('poster_path')}" if movie_details.get('poster_path') else "/static/movie_placeholder.jpeg"
    overview = movie_details.get('overview', '')
    genres = ", ".join([g['name'] for g in movie_details.get('genres', [])])
    rating = movie_details.get('vote_average', 0)
    vote_count = "{:,}".format(movie_details.get('vote_count', 0))
    
    rel_date = movie_details.get('release_date', '')
    release_date = pd.to_datetime(rel_date).strftime('%b %d, %Y') if rel_date else "N/A"
    
    runtime_min = int(movie_details.get('runtime') or 0)
    if runtime_min % 60 == 0:
        runtime = f"{runtime_min // 60} hour(s)"
    else:
        runtime = f"{runtime_min // 60} hour(s) {runtime_min % 60} min(s)"
    status = movie_details.get('status', 'Released')

    # Process Cast
    raw_cast = movie_credits.get('cast', [])[:8]
    cast_names, cast_chars, cast_profiles, cast_ids = [], [], [], []
    for c in raw_cast:
        cast_ids.append(c['id'])
        cast_names.append(c['name'])
        cast_chars.append(c['character'])
        cast_profiles.append(f"https://image.tmdb.org/t/p/original{c['profile_path']}" if c.get('profile_path') else "/static/default.jpg")

    # Fetch Person Bios in Parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        bio_results = list(executor.map(fetch_person_bio, cast_ids))

    cast_bdys = [b['bdy'] for b in bio_results]
    cast_bios = [b['bio'] for b in bio_results]
    cast_places = [b['place'] for b in bio_results]

    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdys[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # 4. Fetch reviews and run sentiment analysis
    movie_reviews = get_reviews_with_sentiment(movie_id, imdb_id)

    # 5. Render HTML Output
    rendered_html = render_template('recommend.html', title=original_title, poster=poster, overview=overview,
                                    vote_average=rating, vote_count=vote_count, release_date=release_date,
                                    runtime=runtime, status=status, genres=genres, movie_cards=movie_cards,
                                    reviews=movie_reviews, casts=casts, cast_details=cast_details,
                                    developer_info=DEV_INFO, movie_id=movie_id)

    return jsonify({'status': 'success', 'html': rendered_html})

# --- NEW TRAILER ENDPOINT ---
@app.route('/api/trailer/<int:movie_id>', methods=['GET'])
def get_trailer(movie_id):
    if not TMDB_API_KEY:
        return jsonify({"success": False, "message": "API key configuration error"}), 500

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])

        yt_videos = [v for v in results if v.get('site') == 'YouTube']

        if not yt_videos:
            return jsonify({"success": False, "message": "Trailer unavailable for this movie."}), 404

        def get_weight(video):
            weight = 0
            if video.get('type') == 'Trailer': weight += 3
            elif video.get('type') == 'Teaser': weight += 1
            if video.get('official'): weight += 1
            return weight

        yt_videos.sort(key=get_weight, reverse=True)
        best_video = yt_videos[0]

        return jsonify({
            "success": True,
            "youtube_key": best_video.get('key'),
            "title": best_video.get('name')
        })

    except requests.RequestException:
        return jsonify({"success": False, "message": "Failed to retrieve trailer data."}), 502

# ==========================================
# NEW: ISOLATED CHATBOT ROUTES (Zero-Risk)
# ==========================================

# 1. Grab the API key safely
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. Configure the Model Dynamically (EXPLICIT FIX)
chat_model = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # We explicitly set the exact version Google requested in the error logs
        chat_model = genai.GenerativeModel('gemini-3.6-flash')
        print("SUCCESS: Connected to Gemini model -> gemini-3.6-flash")
    else:
        print("Warning: GEMINI_API_KEY not found.")
except Exception as init_err:
    print(f"Failed to initialize Gemini Model. Error: {init_err}")

# Route to load the standalone chat HTML page
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

# Route to handle the chat logic secretly in the background
@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not chat_model:
        return jsonify({"success": False, "message": "Chatbot offline. Failed to detect a valid AI model."}), 500
        
    data = request.get_json()
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"success": False, "message": "Please enter a message."}), 400
        
    try:
        # PERSONA INJECTION: Guarantees it acts like a movie expert regardless of the server
        prompt = (
            "You are a cinematic expert and movie recommendation assistant. "
            "You must ONLY answer questions related to movies, TV shows, actors, directors, and the entertainment industry. "
            "Keep your answers concise and engaging. "
            "If a user asks about an unrelated topic, politely decline and steer the conversation back to movies.\n\n"
            f"User Question: {user_message}"
        )
        
        response = chat_model.generate_content(prompt)
        return jsonify({"success": True, "reply": response.text})
    except Exception as e:
        exact_error = str(e)
        print(f"CRITICAL GEMINI ERROR: {exact_error}")
        return jsonify({"success": False, "message": f"SYSTEM ERROR: {exact_error}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
