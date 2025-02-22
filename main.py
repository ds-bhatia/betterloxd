import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

num_ratings = len(ratings)
num_movies = ratings['movieId'].nunique()
num_users = ratings['userId'].nunique()
avg_rating = ratings['rating'].mean()

print(f"No. of ratings: {num_ratings}")
print(f"No. of movies: {num_movies}")
print(f"No. of users: {num_users}")

plt.figure()
sns.countplot(x="rating", data=ratings, palette="terrain_r")
plt.title("Distribution of movie ratings", fontsize=14)

plt.figure()
movies['genres'] = movies['genres'].str.split('|')
genres_exploded = movies.explode('genres')
genre_counts = genres_exploded['genres'].value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title("Distribution of Movie Genres")
plt.xlabel("Number of Movies")
plt.ylabel("Genres")

plt.show()