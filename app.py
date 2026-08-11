import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# 1. Load ML Model and Vectorizer for Review Sentiment
@st.cache_resource
def load_nlp():
    clf = pickle.load(open('nlp_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tranform.pkl', 'rb'))
    return clf, vectorizer

# 2. Load dataset and count matrix ONLY
@st.cache_data
def load_data_and_vectorizer():
    data = pd.read_csv('main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    return data, count_matrix

clf, vectorizer = load_nlp()
data, count_matrix = load_data_and_vectorizer()

# On-demand recommendation calculation
def rcmd(m):
    m = m.lower()
    if m not in data['movie_title'].unique():
        return None
    idx = data.loc[data['movie_title'] == m].index[0]
    
    sim_scores = cosine_similarity(count_matrix[idx], count_matrix).flatten()
    lst = list(enumerate(sim_scores))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]
    
    return [data['movie_title'][item[0]].capitalize() for item in lst]

# UI Layout
st.title("🎬 Movie Recommendation System & Review Sentiment")
movie_list = sorted(list(data['movie_title'].str.capitalize().unique()))

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    movie_list
)

if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        recommendations = rcmd(selected_movie)
        if recommendations:
            st.subheader(f"Top 10 Recommendations for '{selected_movie}':")
            cols = st.columns(5)
            for idx, rec in enumerate(recommendations[:5]):
                with cols[idx]:
                    st.info(rec)
            cols_bottom = st.columns(5)
            for idx, rec in enumerate(recommendations[5:]):
                with cols_bottom[idx]:
                    st.info(rec)
        else:
            st.warning("Sorry! Movie details not found in database.")

st.markdown("---")
st.subheader("📝 Analyze a Movie Review (NLP Sentiment Analysis)")
user_review = st.text_area("Enter your review about a movie:")

if st.button("Predict Sentiment"):
    if user_review.strip():
        movie_vector = vectorizer.transform(np.array([user_review]))
        pred = clf.predict(movie_vector)
        sentiment = "Good / Positive 😊" if pred[0] == 1 else "Bad / Negative 😞"
        if pred[0] == 1:
            st.success(f"Sentiment: **{sentiment}**")
        else:
            st.error(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please type a review first.")
