# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity

# ratings = pd.read_csv('ratings.csv')
# movies = pd.read_csv('movies.csv')

# num_ratings = len(ratings)
# num_movies = ratings['movieId'].nunique()
# num_users = ratings['userId'].nunique()
# avg_rating = ratings['rating'].mean()

# print(f"No. of ratings: {num_ratings}")
# print(f"No. of movies: {num_movies}")
# print(f"No. of users: {num_users}")

# plt.figure()
# sns.countplot(x="rating", data=ratings, palette="terrain_r")
# plt.title("Distribution of movie ratings", fontsize=14)

# plt.figure()
# movies['genres'] = movies['genres'].str.split('|')
# genres_exploded = movies.explode('genres')
# genre_counts = genres_exploded['genres'].value_counts()
# sns.barplot(x=genre_counts.values, y=genre_counts.index)
# plt.title("Distribution of Movie Genres")
# plt.xlabel("Number of Movies")
# plt.ylabel("Genres")

# #plt.show()


# data = pd.merge(ratings, movies, on='movieId')

# global_avg_rating = data['rating'].mean()

# movie_user_matrix = data.pivot_table(index='title', columns='userId', values='rating', fill_value=global_avg_rating - 2.5)

# movie_similarity = cosine_similarity(movie_user_matrix)
# movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# def recommend_movies(movie_title, num_recommendations):
#     if movie_title not in movie_similarity_df.index:
#         return "Movie not found!"
    
#     similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations + 1]
    
#     return similar_movies

# num_recomendations = int(input("Enter number of movie recomendations: "))
# movie_title = input("Enter the movie title - eg: Toy Story (1995): ")
# a = recommend_movies(movie_title, num_recomendations)
# print(a.head(num_recomendations))

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load MovieLens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge ratings with movie details
data = pd.merge(ratings, movies, on='movieId')

# Normalize user ratings by subtracting mean rating per user
user_mean_ratings = data.groupby('userId')['rating'].transform('mean')
data['normalized_rating'] = data['rating'] - user_mean_ratings

# Create Movie-User matrix with normalized ratings
movie_user_matrix = data.pivot_table(index='title', columns='userId', values='normalized_rating', fill_value=0)

# Compute Movie-Movie similarity using Pearson correlation
movie_similarity_df = movie_user_matrix.T.corr(method='pearson')

def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movie_similarity_df.index:
        return "Movie not found!"
    
    # Get similar movies based on similarity scores
    similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False).index[1:num_recommendations+1]
    
    return similar_movies

# Example Usage
movie_title = "Toy Story (1995)"  # Change to any movie title from the dataset
print(recommend_movies(movie_title, num_recommendations=5))
