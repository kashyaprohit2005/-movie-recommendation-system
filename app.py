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

# Load Sentiment Analysis Model & Vectorizer
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# TMDB API Key - Replace with your actual 32-character key
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "'1e9a8541b13e1d9dff9ac2bda6d982e5'")

DATA = None
COUNT_MATRIX = None

def get_data_and_matrix():
    global DATA, COUNT_MATRIX
    if DATA is None or COUNT_MATRIX is None:
        DATA = pd.read_csv('main_data.csv')
        DATA['comb'] = DATA['comb'].fillna('')
        cv = CountVectorizer()
        COUNT_MATRIX = cv.fit_transform(DATA['comb'])
    return DATA, COUNT_MATRIX

def rcmd(m):
    m = str(m).lower().strip()
    data, count_matrix = get_data_and_matrix()
    
    if m not in data['movie_title'].unique():
        return None
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        similarity_scores = cosine_similarity(count_matrix[i], count_matrix).flatten()
        lst = list(enumerate(similarity_scores))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        return [data['movie_title'][item[0]] for item in lst]

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

def fetch_single_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        resp = requests.get(url, timeout=3).json()
        if resp.get('results') and len(resp['results']) > 0 and resp['results'][0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/original{resp['results'][0]['poster_path']}"
    except Exception:
        pass
    return "/static/movie_placeholder.jpeg"

def fetch_person_bio(cast_id):
    try:
        url = f"https://api.themoviedb.org/3/person/{cast_id}?api_key={TMDB_API_KEY}"
        p = requests.get(url, timeout=3).json()
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

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/get_all_movie_data", methods=["POST"])
def get_all_movie_data():
    movie_title_input = request.form.get('name', '').strip()
    if not movie_title_input:
        return jsonify({'status': 'fail', 'message': 'Please enter a movie title!'}), 400

    # 1. Search Movie on TMDB
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title_input}"
    try:
        search_res = requests.get(search_url, timeout=4).json()
    except Exception as e:
        return jsonify({'status': 'fail', 'message': 'Error contacting movie database.'}), 500

    if not search_res.get('results') or len(search_res['results']) == 0:
        return jsonify({'status': 'fail', 'message': 'Sorry! The movie you requested is not found. Please check the spelling.'}), 404

    target_movie = search_res['results'][0]
    movie_id = target_movie['id']
    original_title = target_movie['original_title']

    # 2. Compute similarity recommendations locally
    rec_movies = rcmd(original_title)
    if not rec_movies:
        rec_movies = rcmd(movie_title_input)
    if not rec_movies:
        return jsonify({'status': 'fail', 'message': 'Sorry! The movie is not in our recommendation database.'}), 404

    # 3. Parallel fetch: Movie Details, Credits, and 10 Recommended Posters
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_details = executor.submit(lambda: requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}", timeout=4).json())
        future_credits = executor.submit(lambda: requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}", timeout=4).json())
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

    # 4. Scrape IMDb Reviews Fast (Non-blocking)
    reviews_list = []
    reviews_status = []
    if imdb_id and imdb_id != 'N/A':
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            imdb_resp = requests.get(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt', headers=headers, timeout=2)
            if imdb_resp.status_code == 200:
                soup = bs.BeautifulSoup(imdb_resp.text, 'lxml')
                soup_result = soup.find_all("div", {"class": "text show-more__control"})
                for review in soup_result[:8]:
                    if review.string:
                        reviews_list.append(review.string)
                        vector = vectorizer.transform(np.array([review.string]))
                        pred = clf.predict(vector)
                        reviews_status.append('Good' if pred[0] == 1 else 'Bad')
        except Exception:
            pass

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    # 5. Render directly to HTML
    rendered_html = render_template('recommend.html', title=original_title, poster=poster, overview=overview,
                                    vote_average=rating, vote_count=vote_count, release_date=release_date,
                                    runtime=runtime, status=status, genres=genres, movie_cards=movie_cards,
                                    reviews=movie_reviews, casts=casts, cast_details=cast_details)

    return jsonify({'status': 'success', 'html': rendered_html})

if __name__ == '__main__':
    app.run(debug=True)
