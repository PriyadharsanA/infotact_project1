import pickle
import streamlit as st
import pandas as pd
import numpy as np
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
    # Placeholder image
    return f"https://via.placeholder.com/150?text={'+'.join(title.split())}"

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

st.set_page_config(page_title="Fast Movie Recommender", layout="centered")
st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>⚡ Fast Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("---")
st.write(f"User ID passed from notebook: {shared_data['user_id']}")

# Load and prepare everything
movies, ratings = load_data()
movies = preprocess_movies(movies)
cosine_sim = compute_tfidf(movies['clean_movie_title'])
user_item_matrix = compute_user_item_matrix(ratings)
predicted_df = compute_svd_matrix(user_item_matrix)

title_to_index = pd.Series(movies.index, index=movies['clean_movie_title'].str.lower())

# ----------------------------
# Streamlit UI
# ----------------------------
option = st.radio("Choose Recommendation Type", ['Content-Based', 'SVD (Collaborative)', 'Hybrid'], horizontal=True)
movie_input = st.text_input("Enter a movie title (for content-based):")
user_input = st.number_input("Enter User ID (for SVD/Hybrid):", min_value=1, step=1)

if st.button("Recommend"):
    if option == 'Content-Based':
        if not movie_input:
            st.warning("Please enter a movie title.")
        else:
            st.subheader("Recommendations (Content-Based):")
            recs = get_content_recommendations(movie_input, title_to_index, cosine_sim, movies)
            if recs.empty:
                st.info("No recommendations found for the entered movie title.")
            else:
                display_movies(recs)

    elif option == 'SVD (Collaborative)':
        if user_input not in user_item_matrix.index:
            st.warning("User ID not found in the dataset.")
        else:
            st.subheader("Recommendations (Collaborative - SVD):")
            recs = recommend_svd(user_input, predicted_df, movies, user_item_matrix)
            if recs.empty:
                st.info("No collaborative recommendations could be generated for this user.")
            else:
                display_movies(recs)

    else:  # Hybrid
        if user_input not in user_item_matrix.index:
            st.warning("User ID not found in the dataset.")
        else:
            st.subheader("Recommendations (Hybrid):")
            recs = hybrid_recommend(user_input, predicted_df, movies, user_item_matrix, cosine_sim)
            if recs.empty:
                st.info("No hybrid recommendations could be generated for this user.")
            else:
                display_movies(recs)
