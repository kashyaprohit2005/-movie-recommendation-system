import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bs4 as bs
import urllib.request
import pickle
import requests

# 1. Load ML Model and Vectorizer
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# 2. TMDB API Key for Backend Proxying
# Replace this string with your active 32-character TMDB API key
TMDB_API_KEY = "1e9a8541b13e1d9dff9ac2bda6d982e5"

# 3. Global Memory Cache (Low RAM footprint)
DATA = None
COUNT_MATRIX = None

def get_data_and_matrix():
    global DATA, COUNT_MATRIX
    if DATA is None or COUNT_MATRIX is None:
        DATA = pd.read_csv('main_data.csv')
        DATA['comb'] = DATA['comb'].fillna('')
        cv = CountVectorizer()
        # Stores only sparse matrix (~5MB) instead of full dense similarity matrix
        COUNT_MATRIX = cv.fit_transform(DATA['comb'])
    return DATA, COUNT_MATRIX

def rcmd(m):
    m = str(m).lower().strip()
    data, count_matrix = get_data_and_matrix()
    
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies.'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        # Calculate similarity ONLY for the requested movie row on-demand (<1MB RAM)
        similarity_scores = cosine_similarity(count_matrix[i], count_matrix).flatten()
        lst = list(enumerate(similarity_scores))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = []
        for item in lst:
            a = item[0]
            l.append(data['movie_title'][a])
        return l

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity_route():
    try:
        movie = request.form.get('name', '')
        rc = rcmd(movie)
        if isinstance(rc, str):
            return rc
        else:
            m_str = "---".join(rc)
            return m_str
    except Exception as e:
        print("Error in /similarity:", str(e))
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies."

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        title = request.form['title']
        cast_ids = request.form['cast_ids']
        cast_names = request.form['cast_names']
        cast_chars = request.form['cast_chars']
        cast_bdys = request.form['cast_bdys']
        cast_bios = request.form['cast_bios']
        cast_places = request.form['cast_places']
        cast_profiles = request.form['cast_profiles']
        imdb_id = request.form['imdb_id']
        poster = request.form['poster']
        genres = request.form['genres']
        overview = request.form['overview']
        vote_average = request.form['rating']
        vote_count = request.form['vote_count']
        release_date = request.form['release_date']
        runtime = request.form['runtime']
        status = request.form['status']
        rec_movies = request.form['rec_movies']
        rec_posters = request.form['rec_posters']

        rec_movies = convert_to_list(rec_movies)
        rec_posters = convert_to_list(rec_posters)
        cast_names = convert_to_list(cast_names)
        cast_chars = convert_to_list(cast_chars)
        cast_profiles = convert_to_list(cast_profiles)
        cast_bdys = convert_to_list(cast_bdys)
        cast_bios = convert_to_list(cast_bios)
        cast_places = convert_to_list(cast_places)

        cast_ids = cast_ids.split(',')
        cast_ids[0] = cast_ids[0].replace("[", "")
        cast_ids[-1] = cast_ids[-1].replace("]", "")

        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
        casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
        cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdys[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

        reviews_list = []
        reviews_status = []
        
        try:
            req = urllib.request.Request(
                f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            sauce = urllib.request.urlopen(req).read()
            soup = bs.BeautifulSoup(sauce, 'lxml')
            soup_result = soup.find_all("div", {"class": "text show-more__control"})

            for reviews in soup_result:
                if reviews.string:
                    reviews_list.append(reviews.string)
                    movie_review_list = np.array([reviews.string])
                    movie_vector = vectorizer.transform(movie_review_list)
                    pred = clf.predict(movie_vector)
                    reviews_status.append('Good' if pred else 'Bad')
        except Exception as e:
            print("Error scraping IMDb reviews:", str(e))

        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

        return render_template('recommend.html', title=title, poster=poster, overview=overview,
                               vote_average=vote_average, vote_count=vote_count, release_date=release_date,
                               runtime=runtime, status=status, genres=genres, movie_cards=movie_cards,
                               reviews=movie_reviews, casts=casts, cast_details=cast_details)
    except Exception as e:
        print("Error in /recommend:", str(e))
        return "An error occurred while generating recommendation cards.", 500

# ==================== TMDB PROXY ENDPOINTS ====================
@app.route("/api/search_movie")
def proxy_search_movie():
    title = request.args.get('title', '')
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    resp = requests.get(url)
    return jsonify(resp.json()), resp.status_code

@app.route("/api/movie_details/<movie_id>")
def proxy_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    resp = requests.get(url)
    return jsonify(resp.json()), resp.status_code

@app.route("/api/movie_credits/<movie_id>")
def proxy_movie_credits(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    resp = requests.get(url)
    return jsonify(resp.json()), resp.status_code

@app.route("/api/person/<person_id>")
def proxy_person_details(person_id):
    url = f"https://api.themoviedb.org/3/person/{person_id}?api_key={TMDB_API_KEY}"
    resp = requests.get(url)
    return jsonify(resp.json()), resp.status_code

if __name__ == '__main__':
    app.run(debug=True)
