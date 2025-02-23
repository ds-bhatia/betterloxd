import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load datasets
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Split dataset into train and test
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Pivot table for collaborative filtering
rating_matrix = train_data.pivot_table(index='title', columns='userId', values='rating')
rating_matrix.fillna(0, inplace=True)  # Fill NaN values with 0

# Compute movie similarity using cosine similarity
movie_similarity = cosine_similarity(rating_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=rating_matrix.index, columns=rating_matrix.index)

### 1. COLLABORATIVE FILTERING (ITEM-BASED) ###
def recommend_movies_collaborative(movie_title, num_recommendations=5, user_id=None):
    matches = [title for title in movie_similarity_df.index if movie_title.lower() in title.lower()]
    
    if not matches:
        return f"Movie '{movie_title}' not found. Try another title."
    
    best_match = matches[0]
    similar_movies = movie_similarity_df[best_match].sort_values(ascending=False)
    
    # Exclude movies already rated by the user
    if user_id is not None:
        user_rated_movies = set(train_data[train_data['userId'] == user_id]['title'])
        similar_movies = similar_movies[~similar_movies.index.isin(user_rated_movies)]
    
    return similar_movies.head(num_recommendations)

def recall_at_k(k=5):
    recall_scores = []
    
    user = random.choice(test_data['userId'].unique())
    user_test_movies = set(test_data[test_data['userId'] == user]['title'])
    
    if len(user_test_movies) == 0:
        return 0  # Avoid division by zero if user has no relevant movies in test set

    test_movie = np.random.choice(list(user_test_movies))
    recommended_movies = recommend_movies_collaborative(test_movie, k, user_id=user)

    recommended_titles = set(recommended_movies.index)
    relevant_recommendations = recommended_titles.intersection(user_test_movies)
    
    recall = len(relevant_recommendations) / len(user_test_movies)
    recall_scores.append(recall)
    
    print(f"User {user}: Testing with '{test_movie}'")
    
    a = list(recommended_titles)
    print("Recommended: ")
    for i in a:
        print(i)
    print("-------------")
    
    print(f"Relevant Recommendations:")
    a = list(relevant_recommendations)
    for i in a:
        print(i)
    print("-------------")
    
    

    return np.mean(recall_scores) if recall_scores else 0


# Compute and print Recall@K
recall_k = recall_at_k(7)
print(f"\nOverall Recall@7: {recall_k:.4f}")


### PRECISION@K EVALUATION ###
def precision_at_k(k=5):
    precision_scores = []
    
    user = random.choice(test_data['userId'].unique())
    # Get the user's movies in the test set (relevant movies)
    user_test_movies = set(test_data[test_data['userId'] == user]['title'])

    # Pick a random movie from the test set as the reference movie
    test_movie = np.random.choice(list(user_test_movies))
    
    # Get recommendations, but allow some overlap with the test set
    recommended_movies = recommend_movies_collaborative(test_movie, k, user_id=user)

    recommended_titles = set(recommended_movies.index)  # Extract recommended movie titles
    
    # Calculate how many recommended movies are in the test set (relevant)
    relevant_recommendations = recommended_titles.intersection(user_test_movies)
    
    # Compute precision
    precision = len(relevant_recommendations) / k
    precision_scores.append(precision)
    
    print(f"User {user}: Testing with '{test_movie}'")
    a = list(recommended_titles)
    print("Recommended: ")
    for i in a:
        print(i)
    print("-------------")
    print(f"Relevant Recommendations:")
    a = list(relevant_recommendations)
    for i in a:
        print(i)
    print("-------------")

 
    

    return np.mean(precision_scores) if precision_scores else 0

# Compute and print Precision@K
precision_k = precision_at_k(7)
print(f"\nOverall Precision@7: {precision_k:.4f}")
