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
# movie_user_matrix = data.pivot_table(index='title', columns='userId', values='rating', fill_value=global_avg_rating)

# movie_similarity = cosine_similarity(movie_user_matrix)
# movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# def recommend_movies(movie_title, num_recommendations):
#     if movie_title not in movie_similarity_df.index:
#         return "Movie not found!"
    
#     similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations + 1]
    
#     return similar_movies

# num_recomendations = int(input("Enter number of movie recomendations you want: "))
# movie_title = input("Enter the movie title - eg: Toy Story (1995): ")
# a = recommend_movies(movie_title, num_recomendations)
# print(a.head(num_recomendations))





# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.stats import pearsonr

# # Load datasets
# ratings = pd.read_csv('ratings.csv')
# movies = pd.read_csv('movies.csv')

# # Basic statistics
# num_ratings = len(ratings)
# num_movies = ratings['movieId'].nunique()
# num_users = ratings['userId'].nunique()
# avg_rating = ratings['rating'].mean()

# print(f"No. of ratings: {num_ratings}")
# print(f"No. of movies: {num_movies}")
# print(f"No. of users: {num_users}")

# # Visualizing rating distribution
# plt.figure()
# sns.countplot(x="rating", data=ratings, palette="terrain_r")
# plt.title("Distribution of Movie Ratings", fontsize=14)

# plt.figure()
# movies['genres'] = movies['genres'].str.split('|')
# genres_exploded = movies.explode('genres')
# genre_counts = genres_exploded['genres'].value_counts()
# sns.barplot(x=genre_counts.values, y=genre_counts.index)
# plt.title("Distribution of Movie Genres")
# plt.xlabel("Number of Movies")
# plt.ylabel("Genres")

# #plt.show()

# # Merge datasets
# data = pd.merge(ratings, movies, on='movieId')

# # Compute mean-centered rating matrix for similarity calculation
# rating_matrix = data.pivot_table(index='title', columns='userId', values='rating')
# mean_ratings = rating_matrix.mean(axis=1)
# centered_ratings = rating_matrix.sub(mean_ratings, axis=0).fillna(0)

# # Function to compute Pearson correlation-based similarity
# def compute_similarity(movie_matrix):
#     similarity_df = pd.DataFrame(index=movie_matrix.index, columns=movie_matrix.index)
    
#     for movie1 in movie_matrix.index:
#         for movie2 in movie_matrix.index:
#             if movie1 != movie2:
#                 corr, _ = pearsonr(movie_matrix.loc[movie1], movie_matrix.loc[movie2])
#                 similarity_df.loc[movie1, movie2] = corr

#     return similarity_df.astype(float).fillna(0)

# # Compute similarity matrix
# movie_similarity_df = compute_similarity(centered_ratings)

# # Function to recommend movies
# def recommend_movies(movie_title, num_recommendations=5, min_ratings=20):
#     movie_title = movie_title.strip().lower()

#     # Find closest matching movie title
#     matches = [title for title in movie_similarity_df.index if movie_title in title.lower()]
    
#     if not matches:
#         return f"Movie '{movie_title}' not found! Try another title."

#     best_match = matches[0]  # Taking the first match
    
#     # Filter similar movies with sufficient ratings
#     similar_movies = movie_similarity_df[best_match].dropna().sort_values(ascending=False)
#     similar_movies = similar_movies[similar_movies.index.isin(centered_ratings[centered_ratings.count(axis=1) > min_ratings].index)]
    
#     if similar_movies.empty:
#         return f"No strong recommendations for '{best_match}'. Try another movie."

#     return similar_movies.head(num_recommendations)

# # Get user input
# num_recommendations = int(input("Enter number of movie recommendations you want: "))
# movie_title = input("Enter the movie title (e.g., Toy Story (1995)): ")
# recommendations = recommend_movies(movie_title, num_recommendations)
# print(recommendations)





# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity

# # Load datasets
# ratings = pd.read_csv('ratings.csv')
# movies = pd.read_csv('movies.csv')

# # Merge datasets
# data = pd.merge(ratings, movies, on='movieId')

# # Create a user-item matrix (sparse format)
# rating_matrix = data.pivot_table(index='title', columns='userId', values='rating')

# # Fill NaN values with 0 (alternative: fill with movie's mean rating)
# rating_matrix.fillna(0, inplace=True)

# # Compute movie similarity using cosine similarity
# movie_similarity = cosine_similarity(rating_matrix)

# # Convert to DataFrame for easy lookup
# movie_similarity_df = pd.DataFrame(movie_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# # Function to find similar movies
# def recommend_movies(movie_title, num_recommendations=5):
#     # Find closest matching title (case-insensitive)
#     matches = [title for title in movie_similarity_df.index if movie_title.lower() in title.lower()]
    
#     if not matches:
#         return f"Movie '{movie_title}' not found. Try another title."

#     best_match = matches[0]  # Take the first close match
    
#     # Get top similar movies (excluding itself)
#     similar_movies = movie_similarity_df[best_match].sort_values(ascending=False)[1:num_recommendations + 1]
    
#     return similar_movies

# # Get user input
# num_recommendations = int(input("Enter number of movie recommendations you want: "))
# movie_title = input("Enter the movie title (e.g., Toy Story (1995)): ")
# recommendations = recommend_movies(movie_title, num_recommendations)
# print(recommendations)





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
