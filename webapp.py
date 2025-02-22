import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(page_title="Data Analysis Web App", layout="wide")
    
    st.sidebar.title("📊 Data Analysis Web App")
    st.sidebar.markdown("---")
    option = st.sidebar.radio("**Select a section:**", ["🎬 Movies Dataset", "⭐ Ratings Dataset"])
    
    if option == "🎬 Movies Dataset":
        movies_infographics()
    elif option == "⭐ Ratings Dataset":
        ratings_infographics()

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

if __name__ == "__main__":
    main()
