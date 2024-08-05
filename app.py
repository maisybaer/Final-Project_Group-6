#import files
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data and models
@st.cache
def load_data():
    movies_metadata = pd.read_csv('movies_metadata.csv')
    ratings = pd.read_csv('ratings_small.csv')
    with open('knn_model.pkl', 'rb') as file:
        knn = pickle.load(file)
    return movies_metadata, ratings, knn

movies_metadata, ratings, knn = load_data()

# Prepare the data
ratings.drop("timestamp", axis=1, inplace=True)
final_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
csr_data = csr_matrix(final_ratings.values)
user_matrix = final_ratings.copy()

# Define the recommendation function
def movie_recommender_engine(movie_name, matrix, cf_model, n_recs):
    # Fit the model with the matrix
    cf_model.fit(matrix)

    # Extract input movie title using fuzzy matching
    matched_movie = process.extractOne(movie_name, movies_metadata['title'])
    if matched_movie:
        movie_id = matched_movie[2]
    else:
        return "Movie not found. Please check your input."

    # Check if movie_id exists in matrix
    if movie_id not in matrix.columns:
        return "Movie not found in ratings. Please check your input."

    # Create a vector for the movie
    movie_vector = np.zeros(matrix.shape[1])  # Shape (7454,)
    movie_vector[matrix.columns.get_loc(movie_id)] = matrix[movie_id].values.mean()  # Fill in the rating

    # Reshape to (1, 7454)
    movie_vector = movie_vector.reshape(1, -1)

    # Calculate neighbor distances
    distances, indices = cf_model.kneighbors(movie_vector, n_neighbors=n_recs)

    # Create a list for recommendations
    movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:n_recs]

    movie_titles = [movies_metadata['title'][i[0]] for i in movie_rec_ids]

    return movie_titles

# Streamlit UI
st.title("Movie Recommender System")
st.write("Enter a movie you like and get recommendations!")

movie_name = st.text_input("Enter a movie name:")
if movie_name:
    recommendations = movie_recommender_engine(movie_name, user_matrix, knn, 10)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("Here are some movies you might like:")
        for movie in recommendations:
            st.write(movie)

