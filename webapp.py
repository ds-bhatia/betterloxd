import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    st.set_page_config(page_title="Data Analysis & Recommendation Web App", layout="wide")
    
    st.sidebar.title("📊 Data Analysis & Recommendation Web App")
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

def dataset_summary():
    st.title("📄 Dataset Summary")
    ratings = pd.read_csv("ratings.csv")
    
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
    movies = pd.read_csv("movies.csv")
    
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
    ratings = pd.read_csv("ratings.csv")
    
    st.write("### Preview of Dataset")
    st.write(ratings.head())
    
    st.subheader("Distribution of Movie Ratings")
    fig, ax = plt.subplots()
    sns.countplot(x="rating", data=ratings, palette="terrain_r", ax=ax)
    st.pyplot(fig)

def movie_recommendation():
    st.title("🎥 Movie Recommendation System")
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
    
    user_rating_matrix = rating_matrix.T
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_rating_matrix)
    
    st.subheader("Get Movie Recommendations")
    movie_title = st.selectbox("Select a movie:", movies['title'].values)
    num_recommendations = st.slider("Number of recommendations:", 1, 20, 5)
    model_type = st.radio("Choose Recommendation Model:", ["Collaborative Filtering", "Content-Based Filtering", "Hybrid Model", "User-Based Filtering"])
    if model_type == "User-Based Filtering":
            with st.form('getInp'):
                user_id = st.number_input('Input User ID: ', min_value = 1, step =1)
                submit = st.form_submit_button('Submit')
            if submit:
                if user_id in user_rating_matrix.index:
                    distances, indices = knn.kneighbors([user_rating_matrix.loc[user_id]], n_neighbors=6)
                    similar_users = indices.flatten()[1:]
                    similar_users_ratings = user_rating_matrix.iloc[similar_users].mean().sort_values(ascending=False)[1:num_recommendations + 1]
                    recommendations = similar_users_ratings.index.tolist()
                else:
                    recommendations = "User not found!"
                i = 1
                st.write("Recommended Movies: ")
                for movie in recommendations:
                    st.write(i, movie) 
                    i += 1
    else:
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

if __name__ == "__main__":
    main()