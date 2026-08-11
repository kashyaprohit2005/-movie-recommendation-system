import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

# Load the NLP model and CountVectorizer
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# TMDB API Key for Backend Proxying
TMDB_API_KEY = "YOUR_TMDB_API_KEY_HERE"  # <-- Paste your 32-character TMDB API Key here

def create_similarity():
    data = pd.read_csv('main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies.'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
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
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
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

    # String transformations
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
    
    # Cast dictionary rendering
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdys[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping IMDb reviews
    sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []
    reviews_status = []
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    return render_template('recommend.html', title=title, poster=poster, overview=overview,
                           vote_average=vote_average, vote_count=vote_count, release_date=release_date,
                           runtime=runtime, status=status, genres=genres, movie_cards=movie_cards,
                           reviews=movie_reviews, casts=casts, cast_details=cast_details)

# ==================== TMDB PROXY ENDPOINTS ====================
@app.route("/api/search_movie")
def proxy_search_movie():
    title = request.args.get('title')
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
