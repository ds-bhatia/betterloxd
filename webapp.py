import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="Data Analysis & Recommendation Web App", layout="wide")
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

data = pd.merge(ratings, movies, on='movieId')

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

rating_matrix = train_data.pivot_table(index='title', columns='userId', values='rating')
rating_matrix.fillna(0, inplace=True)

movie_similarity_df = pd.DataFrame(cosine_similarity(rating_matrix), index=rating_matrix.index, columns=rating_matrix.index)

# User-based collaborative filtering
user_rating_matrix = rating_matrix.T
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_rating_matrix)

def recommend_movies_user_based(user_id, num_recommendations=5):
    if user_id not in user_rating_matrix.index:
        return f"User {user_id} not found!"
    
    distances, indices = knn.kneighbors([user_rating_matrix.loc[user_id]], n_neighbors=6)
    similar_users = indices.flatten()[1:]
    similar_users_ratings = user_rating_matrix.iloc[similar_users].mean().sort_values(ascending=False)
    
    return similar_users_ratings.head(num_recommendations)

def precision_at_k(k=5, num_users=10):
    precision_scores = []
    
    sampled_users = random.sample(list(test_data['userId'].unique()), min(num_users, len(test_data['userId'].unique())))
    
    user = random.choice(test_data['userId'].unique())
    user_test_movies = set(test_data[test_data['userId'] == user]['title'])
    test_movie = np.random.choice(list(user_test_movies))
    
    recommended_movies = recommend_movies_user_based(user, k)
    recommended_titles = set(recommended_movies.index)
    relevant_recommendations = recommended_titles.intersection(user_test_movies)
    if(list(relevant_recommendations) == []):
        return precision_at_k(k)
    precision = len(relevant_recommendations) / k
    precision_scores.append(precision)   
    st.write(f"User {user}:")
    a = list(recommended_titles)
    st.write("Recommended: ")
    for i in a:
        st.write(i)
    st.write("-----------------")
    a = list(relevant_recommendations)
    st.write("Relevant Recommendations: ")
    for i in a:
        st.write(i)
    st.write("-----------------")
    return np.mean(precision_scores) if precision_scores else 0

def map(k=5, num_users=10):
    precision_scores = []
    
    sampled_users = random.sample(list(test_data['userId'].unique()), min(num_users, len(test_data['userId'].unique())))
    
    for user in sampled_users:
        user_test_movies = set(test_data[test_data['userId'] == user]['title'])
        if not user_test_movies:
            continue
        test_movie = np.random.choice(list(user_test_movies))
        
        recommended_movies = recommend_movies_user_based(user, k)
        recommended_titles = set(recommended_movies.index)
        relevant_recommendations = recommended_titles.intersection(user_test_movies)
        
        precision = len(relevant_recommendations) / k
        precision_scores.append(precision)   
    return np.mean(precision_scores) if precision_scores else 0


def recall_at_k(k=5, num_users=10):
    recall_scores = []
    
    sampled_users = random.sample(list(test_data['userId'].unique()), min(num_users, len(test_data['userId'].unique())))
    
    for user in sampled_users:
        user_test_movies = set(test_data[test_data['userId'] == user]['title'])
        if not user_test_movies:
            continue
        test_movie = np.random.choice(list(user_test_movies))
        
        recommended_movies = recommend_movies_user_based(user, k)
        recommended_titles = set(recommended_movies.index)
        relevant_recommendations = recommended_titles.intersection(user_test_movies)
        
        recall = len(relevant_recommendations) / len(user_test_movies) if user_test_movies else 0
        recall_scores.append(recall)
        
    return np.mean(recall_scores) if recall_scores else 0

def hit_rate(k=5, num_users=10):
    hit_count = 0
    
    sampled_users = random.sample(list(test_data['userId'].unique()), min(num_users, len(test_data['userId'].unique())))
    
    for user in sampled_users:
        user_test_movies = set(test_data[test_data['userId'] == user]['title'])
        if not user_test_movies:
            continue
        test_movie = np.random.choice(list(user_test_movies))
        
        recommended_movies = recommend_movies_user_based(user, k)
        recommended_titles = set(recommended_movies.index)
        
        if recommended_titles.intersection(user_test_movies):
            hit_count += 1
    
    return hit_count / num_users if num_users else 0

def dataset_summary():
    st.title("📄 Dataset Summary")
    num_ratings = len(ratings)
    num_movies = ratings['movieId'].nunique()
    num_users = ratings['userId'].nunique()
    avg_rating = ratings['rating'].mean()
    
    st.write(f"**No. of ratings:** {num_ratings}")
    st.write(f"**No. of movies:** {num_movies}")
    st.write(f"**No. of users:** {num_users}")
    st.write(f"**Average rating:** {avg_rating:.2f}")

def movies_infographics():
    st.title("🎬 Movies Dataset Infographics")
    st.write("### Preview of Dataset")
    st.write(movies.head())
    st.subheader("Distribution of Movie Genres")
    movies['genres'] = movies['genres'].str.split('|')
    genres_exploded = movies.explode('genres')
    genre_counts = genres_exploded['genres'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
    plt.xlabel("Number of Movies")
    plt.ylabel("Genres")
    st.pyplot(fig)

def ratings_infographics():
    st.title("⭐ Ratings Dataset Infographics")
    st.write("### Preview of Dataset")
    st.write(ratings.head())
    st.subheader("Distribution of Movie Ratings")
    fig, ax = plt.subplots()
    sns.countplot(x="rating", data=ratings, palette="terrain_r", ax=ax)
    st.pyplot(fig)

def movie_recommendation():
    st.title("🎥 Movie Recommendation System")
    st.subheader("Get Movie Recommendations")
    num_recommendations = st.slider("Number of recommendations (K):", 1, 20, 7)
    model_type = st.radio("Choose Recommendation Model:", ["User-Based Filtering", "Collaborative Filtering", "Content-Based Filtering", "Hybrid Model"])
    
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    data = pd.merge(ratings, movies, on='movieId')
    rating_matrix = data.pivot_table(index='title', columns='userId', values='rating')
    rating_matrix.fillna(0, inplace=True)
    movie_similarity_df = pd.DataFrame(cosine_similarity(rating_matrix), index=rating_matrix.index, columns=rating_matrix.index)
    
    movies['genres'] = movies['genres'].replace("(no genres listed)", "").astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    content_similarity_df = pd.DataFrame(cosine_similarity(tfidf_matrix), index=movies['title'], columns=movies['title'])
    
    if model_type == "User-Based Filtering":
        if st.button("Compute Metrics"):
            precision_k = precision_at_k(num_recommendations)
            recall_k = recall_at_k(num_recommendations)
            hit_rate_k = hit_rate(num_recommendations)
            #rmse_value = rmse()
            map_ = map(num_recommendations)
            st.write(f"Precision@{num_recommendations}: {precision_k:.4f}")
            st.write(f"Overall Recall@{num_recommendations}: {recall_k:.4f}")
            st.write(f"Overall Hit Rate@{num_recommendations}: {hit_rate_k:.4f}")
           # st.write(f"RMSE: {rmse_value:.4f}")
            st.write(f"Mean Average Precision: {map_}")

    else:
        movie_title = st.selectbox("Select a movie:", movies['title'].values)
        if st.button("Get Recommendations"):
            if model_type == "Collaborative Filtering":
                recommendations = movie_similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations + 1]
                recommendations = recommendations.index.tolist()
            elif model_type == "Content-Based Filtering":
                recommendations = content_similarity_df[movie_title].sort_values(ascending=False)[1:num_recommendations + 1]
                recommendations = recommendations.index.tolist()
            elif model_type == "Hybrid Model":
                collaborative_movies = movie_similarity_df[movie_title].sort_values(ascending=False)[1:11]
                content_movies = content_similarity_df[movie_title].sort_values(ascending=False)[1:11]
                hybrid_recommendations = pd.concat([collaborative_movies, content_movies]).groupby(level=0).mean()
                recommendations = hybrid_recommendations.sort_values(ascending=False)[1:num_recommendations + 1]
                recommendations = recommendations.index.tolist()
            i = 1
            st.write("Recommended Movies: ")
            for movie in recommendations:
                st.write(i, movie) 
                i += 1
    

def main():
    
    
    st.sidebar.title("📊 Betterloxd")
    st.sidebar.markdown("---")
    main_option = st.sidebar.radio("**Select a section:**", ["📄 Dataset Summary", "📂 Dataset Details", "🎥 Movie Recommendation"])
    
    if main_option == "📄 Dataset Summary":
        dataset_summary()
    elif main_option == "📂 Dataset Details":
        sub_option = st.sidebar.radio("**Select a dataset:**", ["🎬 Movies Dataset", "⭐ Ratings Dataset"])
        if sub_option == "🎬 Movies Dataset":
            movies_infographics()
        elif sub_option == "⭐ Ratings Dataset":
            ratings_infographics()
    elif main_option == "🎥 Movie Recommendation":
        movie_recommendation()

if __name__ == "__main__":
    main()
