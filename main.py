

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Pivot table for collaborative filtering
rating_matrix = data.pivot_table(index='title', columns='userId', values='rating')

# Fill NaN values with 0 (alternative: use movie's mean rating)
rating_matrix.fillna(0, inplace=True)

# Compute movie similarity using cosine similarity
movie_similarity = cosine_similarity(rating_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=rating_matrix.index, columns=rating_matrix.index)


### 1. COLLABORATIVE FILTERING (ITEM-BASED) ###
def recommend_movies_collaborative(movie_title, num_recommendations=5):
    matches = [title for title in movie_similarity_df.index if movie_title.lower() in title.lower()]
    
    if not matches:
        return f"Movie '{movie_title}' not found. Try another title."
    
    best_match = matches[0]
    similar_movies = movie_similarity_df[best_match].sort_values(ascending=False)[1:num_recommendations + 1]
    
    return similar_movies


### 2. CONTENT-BASED FILTERING (GENRE SIMILARITY) ###
# Convert genres to strings
movies['genres'] = movies['genres'].replace("(no genres listed)", "").astype(str)

# Create a TF-IDF Vectorizer model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity for content-based filtering
content_similarity = cosine_similarity(tfidf_matrix)
content_similarity_df = pd.DataFrame(content_similarity, index=movies['title'], columns=movies['title'])

def recommend_movies_content(movie_title, num_recommendations=5):
    matches = [title for title in content_similarity_df.index if movie_title.lower() in title.lower()]
    
    if not matches:
        return f"Movie '{movie_title}' not found. Try another title."
    
    best_match = matches[0]
    similar_movies = content_similarity_df[best_match].sort_values(ascending=False)[1:num_recommendations + 1]
    
    return similar_movies


### 3. HYBRID MODEL (COLLABORATIVE + CONTENT-BASED) ###
def recommend_movies_hybrid(movie_title, num_recommendations=5):
    matches = [title for title in movie_similarity_df.index if movie_title.lower() in title.lower()]
    
    if not matches:
        return f"Movie '{movie_title}' not found. Try another title."
    
    best_match = matches[0]
    
    # Get top 10 recommendations from both methods
    collaborative_movies = movie_similarity_df[best_match].sort_values(ascending=False)[1:11]
    content_movies = content_similarity_df[best_match].sort_values(ascending=False)[1:11]
    
    # Combine scores
    hybrid_recommendations = pd.concat([collaborative_movies, content_movies]).groupby(level=0).mean()
    
    return hybrid_recommendations.sort_values(ascending=False).head(num_recommendations)


### 4. USER-BASED COLLABORATIVE FILTERING (kNN) ###
user_rating_matrix = rating_matrix.T  # Transpose for user-based approach

# Fit kNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_rating_matrix)

def recommend_movies_user_based(user_id, num_recommendations=5):
    if user_id not in user_rating_matrix.index:
        return f"User {user_id} not found!"
    
    # Find similar users
    distances, indices = knn.kneighbors([user_rating_matrix.loc[user_id]], n_neighbors=6)
    
    similar_users = indices.flatten()[1:]  # Exclude the user itself
    similar_users_ratings = user_rating_matrix.iloc[similar_users].mean().sort_values(ascending=False)
    
    return similar_users_ratings.head(num_recommendations)


### MAIN FUNCTION ###
print("\nChoose recommendation model:")
print("1 - Collaborative Filtering (Item-Based)")
print("2 - Content-Based Filtering")
print("3 - Hybrid Model")
print("4 - User-Based Collaborative Filtering (kNN)")

choice = int(input("Enter model choice (1-4): "))

if choice in [1, 2, 3]:
    movie_title = input("Enter the movie title (e.g., Toy Story (1995)): ")
    num_recommendations = int(input("Enter number of recommendations: "))

    if choice == 1:
        print(recommend_movies_collaborative(movie_title, num_recommendations))
    elif choice == 2:
        print(recommend_movies_content(movie_title, num_recommendations))
    else:
        print(recommend_movies_hybrid(movie_title, num_recommendations))

elif choice == 4:
    user_id = int(input("Enter User ID: "))
    num_recommendations = int(input("Enter number of recommendations: "))
    print(recommend_movies_user_based(user_id, num_recommendations))

else:
    print("Invalid choice. Please enter a number between 1 and 4.")
