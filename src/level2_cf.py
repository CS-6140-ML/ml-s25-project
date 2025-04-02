import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings_df):
    """
    Build a user-item matrix from the ratings dataframe.

    Parameters:
        ratings_df (pd.DataFrame): Should contain 'user_id', 'business_id', and 'rating'.

    Returns:
        pd.DataFrame: Pivot table with users as rows and businesses as columns.
    """
    user_item_matrix = ratings_df.pivot(index='user_id', columns='business_id', values='rating')
    return user_item_matrix


def user_based_recommendations(user_id, user_item_matrix, top_n=5):
    """
    Recommend items to a user using user-based collaborative filtering.

    For each unrated item, a weighted average rating is computed from similar users.

    Parameters:
        user_id: The target user.
        user_item_matrix: Pivot table of user ratings.
        top_n: Number of recommendations to return.

    Returns:
        list: List of recommended business_ids.
    """
    # Fill NaNs with 0 for similarity computation
    matrix_filled = user_item_matrix.fillna(0)
    # Compute cosine similarity between users
    similarity = cosine_similarity(matrix_filled)
    user_ids = list(user_item_matrix.index)

    try:
        idx = user_ids.index(user_id)
    except ValueError:
        print("User ID not found in the matrix.")
        return []

    user_similarities = similarity[idx]
    # For items not rated by the user, compute a weighted sum of ratings from similar users.
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index
    predicted_ratings = {}
    for item in unrated_items:
        ratings = user_item_matrix[item]
        # Consider only users who have rated the item.
        mask = ratings.notna()
        if mask.sum() == 0:
            predicted_ratings[item] = 0
        else:
            # Weighted sum of ratings from similar users
            predicted_ratings[item] = np.dot(ratings[mask], user_similarities[mask]) / np.sum(user_similarities[mask])

    # Return top_n items sorted by predicted rating.
    recommended_items = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item for item, score in recommended_items]


if __name__ == "__main__":
    # Load ratings data (assumes preprocessed ratings file exists)
    ratings_df = pd.read_csv("data/processed/ratings_processed.csv")
    user_item_matrix = build_user_item_matrix(ratings_df)

    # Pick a sample user_id from the ratings data
    sample_user_id = user_item_matrix.index[0]
    print(f"Recommendations for user {sample_user_id}:")
    recs = user_based_recommendations(sample_user_id, user_item_matrix, top_n=5)
    print(recs)
