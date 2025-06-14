import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print("Loading datasets...")
print("Loading movies data...")
movies = pd.read_csv('infotact_project1/movies.csv')

print("Loading ratings data (this might take a moment)...")
# Read ratings in chunks to be more memory efficient
ratings_chunks = pd.read_csv('infotact_project1/ratings.csv', chunksize=100000)
ratings = pd.concat(ratings_chunks)

print("Loading users data...")
users = pd.read_csv('infotact_project1/users.csv')

# Basic Dataset Information
print("\n=== Basic Dataset Information ===")
print("-" * 50)
print(f"Number of movies: {len(movies)}")
print(f"Number of ratings: {len(ratings)}")
print(f"Number of users: {len(users)}")

print("\nMovies Dataset Sample:")
print(movies.head())

# Movie Genres Analysis
print("\n=== Movie Genres Analysis ===")
print("-" * 50)

# Extract unique genres
print("Analyzing movie genres...")
all_genres = set()
for genres in movies['movie_genres']:
    if isinstance(genres, str):
        genre_list = [g.strip() for g in genres.split(',')]
        all_genres.update(genre_list)

# Count movies per genre
genre_counts = {}
for genres in movies['movie_genres']:
    if isinstance(genres, str):
        for genre in genres.split(','):
            genre = genre.strip()
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

# Plot genre distribution
print("Creating genre distribution plot...")
plt.figure(figsize=(12, 6))
genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
genre_df = genre_df.sort_values('Count', ascending=False)
sns.barplot(x='Count', y='Genre', data=genre_df)
plt.title('Distribution of Movies by Genre')
plt.xlabel('Number of Movies')
plt.tight_layout()
plt.savefig('infotact_project1/genre_distribution.png')
plt.close()

# Ratings Analysis
print("\n=== Ratings Analysis ===")
print("-" * 50)

# Distribution of ratings
print("Creating ratings distribution plot...")
plt.figure(figsize=(10, 6))
sns.histplot(data=ratings, x='user_rating', bins=20)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('infotact_project1/ratings_distribution.png')
plt.close()

# Average rating per movie
print("Calculating average ratings per movie...")
movie_ratings = ratings.groupby('movie_id')['user_rating'].agg(['mean', 'count']).reset_index()
movie_ratings = movie_ratings.merge(movies[['movie_id', 'movie_title']], on='movie_id')

print("\nTop 10 Highest Rated Movies (with at least 100 ratings):")
print(movie_ratings[movie_ratings['count'] >= 100].sort_values('mean', ascending=False).head(10))

# User Behavior Analysis
print("\n=== User Behavior Analysis ===")
print("-" * 50)

# Number of ratings per user
print("Analyzing user rating patterns...")
user_ratings = ratings.groupby('user_id').size().reset_index(name='rating_count')

plt.figure(figsize=(10, 6))
sns.histplot(data=user_ratings, x='rating_count', bins=50)
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.savefig('infotact_project1/user_ratings_distribution.png')
plt.close()

print("\nUser Rating Statistics:")
print(user_ratings['rating_count'].describe())

# Save results to CSV files
print("\nSaving analysis results...")
movie_ratings.to_csv('infotact_project1/movie_ratings_analysis.csv', index=False)
user_ratings.to_csv('infotact_project1/user_ratings_analysis.csv', index=False)
genre_df.to_csv('infotact_project1/genre_analysis.csv', index=False)

print("\nAnalysis complete! Check the generated files:")
print("1. infotact_project1/genre_distribution.png")
print("2. infotact_project1/ratings_distribution.png")
print("3. infotact_project1/user_ratings_distribution.png")
print("4. infotact_project1/movie_ratings_analysis.csv")
print("5. infotact_project1/user_ratings_analysis.csv")
print("6. infotact_project1/genre_analysis.csv") 