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
def evaluate_at_k(k=5):
    precision_scores = []
    recall_scores = []
    hit_scores = []

    first_user = True  # To print details only for the first user

    # Loop over all users in the test set
    for user in test_data['userId'].unique():
        user_test_movies = set(test_data[test_data['userId'] == user]['title'])

        if len(user_test_movies) == 0:
            continue  # Skip users with no relevant movies in the test set

        for test_movie in user_test_movies:  
            # Get recommendations, ensuring the test movie is not included
            recommended_movies = recommend_movies_collaborative(test_movie, k+1, user_id=user)

            if isinstance(recommended_movies, str):  # Check if the result is an error message
                continue

            recommended_titles = set(recommended_movies.index) - {test_movie}  # Remove test movie if present

            # Keep only top-k movies after removing test_movie
            recommended_titles = list(recommended_titles)[:k]
            relevant_recommendations = set(recommended_titles).intersection(user_test_movies)

            # Compute Precision, Recall, and Hit Rate
            precision = len(relevant_recommendations) / k
            recall = len(relevant_recommendations) / len(user_test_movies) if user_test_movies else 0
            hit = 1 if len(relevant_recommendations) > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)
            hit_scores.append(hit)

            # Print details only for the first user in the loop
            if first_user:
                print(f"User {user}: Testing with '{test_movie}'")
                print("Recommended: ")
                for movie in recommended_titles:
                    print(movie)
                print("-------------")

                print("Relevant Recommendations:")
                for movie in relevant_recommendations:
                    print(movie)
                print("-------------")

                first_user = False  # Ensure we print details only once

    # Compute final averaged scores
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_hit_rate = np.mean(hit_scores) if hit_scores else 0  # Should not always be 1 now

    print(f"\nOverall Metrics at K={k}:")
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print(f"Average Hit Rate@{k}: {avg_hit_rate:.4f}")  # Should not always be 1 now

# Run evaluation with K=7
evaluate_at_k(k=7)
