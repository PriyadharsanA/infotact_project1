import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load shared data (if exists)
with open('shared_data.pkl', 'rb') as f:
    shared_data = pickle.load(f)

st.write(f"User ID passed from notebook: {shared_data['user_id']}")

# Load CSV data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocess titles
movies['clean_movie_title'] = movies['movie_title'].astype(str).str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# Content-based similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['clean_movie_title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_to_index = pd.Series(movies.index, index=movies['clean_movie_title'].str.lower())

def get_content_recommendations(title, n=10):
    title = title.lower()
    if title not in title_to_index:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['movie_id', 'movie_title']].iloc[movie_indices].reset_index(drop=True)

# Collaborative filtering using SVD
user_item_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='user_rating').fillna(0)
svd = TruncatedSVD(n_components=50, random_state=42)
user_features = svd.fit_transform(user_item_matrix)
item_features = svd.components_
predicted_df = pd.DataFrame(np.dot(user_features, item_features), index=user_item_matrix.index, columns=user_item_matrix.columns)

def recommend_svd(user_id, predicted_df, movies_df, user_item_df, n=5):
    if user_id not in predicted_df.index:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    user_row = predicted_df.loc[user_id]
    user_row = user_row[~(user_item_df.loc[user_id] > 0)]
    top_ids = user_row.nlargest(n).index
    recs = movies_df[movies_df['movie_id'].isin(top_ids)][['movie_id', 'movie_title']]
    return recs.set_index('movie_id').loc[top_ids].reset_index()

def hybrid_recommend(user_id, predicted_df, movies_df, user_item_df, tfidf_sim=cosine_sim, n=5):
    if user_id not in predicted_df.index:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    user_row = predicted_df.loc[user_id]
    unrated = user_row[~(user_item_df.loc[user_id] > 0)]
    top_svd_ids = unrated.nlargest(30).index
    movie_idxs = movies[movies['movie_id'].isin(top_svd_ids)].index
    sim_avg = cosine_sim[movie_idxs].mean(axis=1)
    top_idx = sim_avg.argsort()[::-1][:n]
    movie_ids = movies.iloc[movie_idxs[top_idx]]['movie_id']
    return movies[movies['movie_id'].isin(movie_ids)][['movie_id', 'movie_title']].reset_index(drop=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("---")

option = st.radio("Choose Recommendation Type", ['Content-Based', 'SVD (Collaborative)', 'Hybrid'], horizontal=True)
movie_input = st.text_input("Enter a movie title (for content-based):")
user_input = st.number_input("Enter User ID (for SVD/Hybrid):", min_value=1, step=1)

# Function to get a placeholder poster image from OMDb API or similar
def fetch_movie_poster(title):
    return f"https://via.placeholder.com/150?text={'+'.join(title.split())}"

# Function to show movies with posters
def display_movies(df):
    for _, row in df.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(fetch_movie_poster(row['movie_title']), width=100)
        with col2:
            st.subheader(row['movie_title'])
        st.markdown("---")

if st.button("Recommend"):
    if option == 'Content-Based':
        st.subheader("Recommendations (Content-Based):")
        recs = get_content_recommendations(movie_input)
        display_movies(recs)
    elif option == 'SVD (Collaborative)':
        st.subheader("Recommendations (Collaborative - SVD):")
        recs = recommend_svd(user_input, predicted_df, movies, user_item_matrix)
        display_movies(recs)
    else:
        st.subheader("Recommendations (Hybrid):")
        recs = hybrid_recommend(user_input, predicted_df, movies, user_item_matrix)
        display_movies(recs)
