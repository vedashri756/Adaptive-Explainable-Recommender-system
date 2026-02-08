import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def popularity_recommender(data, top_k=5, min_ratings=50):
    """
    Recommends top_k popular items based on average rating,
    filtered by minimum number of ratings.
    """

    popularity_df = (
        data.groupby("title")
        .agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    popularity_df = popularity_df[popularity_df["rating_count"] >= min_ratings]

    popularity_df = popularity_df.sort_values(
        by=["mean_rating", "rating_count"],
        ascending=False
    )

    return popularity_df.head(top_k)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_based_cf(data, user_id, top_k=5, min_sim=0.3):
    """
    User-based collaborative filtering using cosine similarity.
    """

    # Create user-item matrix
    user_item = data.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    # Fill missing values with 0 (explicit feedback assumption)
    user_item_filled = user_item.fillna(0)

    # Compute user-user similarity
    similarity = cosine_similarity(user_item_filled)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item.index,
        columns=user_item.index
    )

    # Similar users
    sim_users = similarity_df[user_id]
    sim_users = sim_users[sim_users > min_sim].sort_values(ascending=False)

    # Remove self
    sim_users = sim_users.drop(user_id, errors="ignore")

    # Movies already watched by user
    watched = user_item.loc[user_id]
    watched = watched[watched.notna()].index

    # Weighted ratings
    weighted_scores = {}

    for sim_user, sim_score in sim_users.items():
        sim_user_ratings = user_item.loc[sim_user]
        for movie_id, rating in sim_user_ratings.dropna().items():
            if movie_id not in watched:
                weighted_scores.setdefault(movie_id, 0)
                weighted_scores[movie_id] += sim_score * rating

    if not weighted_scores:
        return pd.DataFrame(columns=["movieId", "score", "title"])


    # Sort recommendations
    recs = (
        pd.Series(weighted_scores)
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index()
        .rename(columns={"index": "movieId", 0: "score"})
    )

    # Attach movie titles
    movies = data[["movieId", "title"]].drop_duplicates()
    recs = recs.merge(movies, on="movieId", how="left")

    return recs

