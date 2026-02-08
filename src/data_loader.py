import pandas as pd
import os

def load_data():
    # Get absolute path to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ratings_path = os.path.join(base_dir, "data", "ratings.csv")
    movies_path = os.path.join(base_dir, "data", "movies.csv")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    data = ratings.merge(movies, on="movieId", how="left")
    return data
