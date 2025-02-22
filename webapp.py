import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(page_title="Data Analysis Web App", layout="wide")
    
    st.sidebar.title("Betterloxd")
    st.sidebar.markdown("---")
    main_option = st.sidebar.radio("**Select a section:**", ["📄 Dataset Summary", "📂 Dataset Details"])
    
    if main_option == "📄 Dataset Summary":
        dataset_summary()
    elif main_option == "📂 Dataset Details":
        sub_option = st.sidebar.radio("**Select a dataset:**", ["🎬 Movies Dataset", "⭐ Ratings Dataset"])
        if sub_option == "🎬 Movies Dataset":
            movies_infographics()
        elif sub_option == "⭐ Ratings Dataset":
            ratings_infographics()

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
    st.write(movies.head(20))
    
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

if __name__ == "__main__":
    main()
