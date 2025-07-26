import pickle
import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse  # for poster text encoding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ----------------------------
# Caching-heavy operations
# ----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

@st.cache_data
def preprocess_movies(movies):
    movies['clean_movie_title'] = movies['movie_title'].astype(str).str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    return movies

@st.cache_data
def compute_tfidf(clean_titles):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(clean_titles)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

@st.cache_data
def compute_user_item_matrix(ratings):
    return ratings.pivot_table(index='user_id', columns='movie_id', values='user_rating').fillna(0)

@st.cache_data
def compute_svd_matrix(user_item_matrix):
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_features = svd.fit_transform(user_item_matrix)
    item_features = svd.components_
    predicted_df = pd.DataFrame(np.dot(user_features, item_features), index=user_item_matrix.index, columns=user_item_matrix.columns)
    return predicted_df

# ----------------------------
# Recommendation Logic
# ----------------------------
def get_content_recommendations(title, title_to_index, cosine_sim, movies, n=10):
    title = title.lower()
    if title not in title_to_index:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['movie_id', 'movie_title']].iloc[movie_indices].reset_index(drop=True)

def recommend_svd(user_id, predicted_df, movies_df, user_item_df, n=5):
    if user_id not in predicted_df.index:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    user_row = predicted_df.loc[user_id]
    if user_id in user_item_df.index:
        user_row = user_row[~(user_item_df.loc[user_id] > 0)]
    if user_row.empty:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    top_ids = user_row.nlargest(n).index
    recs = movies_df[movies_df['movie_id'].isin(top_ids)][['movie_id', 'movie_title']]
    return recs.set_index('movie_id').loc[top_ids].reset_index()

def hybrid_recommend(user_id, predicted_df, movies_df, user_item_df, cosine_sim, n=5):
    if user_id not in predicted_df.index:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    user_row = predicted_df.loc[user_id]
    if user_id in user_item_df.index:
        unrated = user_row[~(user_item_df.loc[user_id] > 0)]
    else:
        unrated = user_row
    if unrated.empty:
        return pd.DataFrame(columns=["movie_id", "movie_title"])
    top_svd_ids = unrated.nlargest(30).index
    movie_idxs = movies_df[movies_df['movie_id'].isin(top_svd_ids)].index.to_numpy()
    sim_avg = cosine_sim[movie_idxs].mean(axis=1)
    top_idx = sim_avg.argsort()[::-1][:n]
    movie_ids = movies_df.iloc[movie_idxs[top_idx]]['movie_id']
    return movies_df[movies_df['movie_id'].isin(movie_ids)][['movie_id', 'movie_title']].reset_index(drop=True)

def fetch_movie_poster(title):
    encoded_title = urllib.parse.quote_plus(title)
    return f"https://via.placeholder.com/150x220.png?text={encoded_title}"

def display_movies(df):
    for _, row in df.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(fetch_movie_poster(row['movie_title']), width=100)
        with col2:
            st.subheader(row['movie_title'])
        st.markdown("---")

# ----------------------------
# App Initialization
# ----------------------------
with open('shared_data.pkl', 'rb') as f:
    shared_data = pickle.load(f)

st.set_page_config(page_title="Fast Movi_
