import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def build_user_item_matrix(ratings_df):
    """
    Build a user-item matrix from ratings data.

    Parameters:
        ratings_df (pd.DataFrame): Contains 'user_id', 'business_id', and 'rating'.

    Returns:
        pd.DataFrame: Pivot table with users as rows and businesses as columns.
    """
    return ratings_df.pivot(index='user_id', columns='business_id', values='rating')


def matrix_factorization_recommendations(user_id, user_item_matrix, top_n=5, n_factors=20):
    """
    Recommend items using matrix factorization (SVD).

    Parameters:
        user_id: The target user.
        user_item_matrix: Pivot table of user ratings.
        top_n: Number of recommendations.
        n_factors: Number of latent factors for SVD.

    Returns:
        list: Recommended business_ids.
    """
    # Fill missing values with 0 for SVD
    matrix_filled = user_item_matrix.fillna(0)

    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    U = svd.fit_transform(matrix_filled)
    Vt = svd.components_

    # Reconstruct the approximate ratings matrix
    reconstructed_matrix = np.dot(U, Vt)

    # Create a DataFrame for predicted ratings
    predicted_ratings = pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index,
                                     columns=user_item_matrix.columns)

    # For the target user, filter out items already rated
    user_actual_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_actual_ratings[user_actual_ratings.isna()].index
    predictions = predicted_ratings.loc[user_id, unrated_items]

    # Get top_n recommendations based on predicted ratings
    recommended_items = predictions.sort_values(ascending=False).head(top_n).index.tolist()
    return recommended_items


if __name__ == "__main__":
    # Load ratings data
    ratings_df = pd.read_csv("data/processed/ratings_processed.csv")
    user_item_matrix = build_user_item_matrix(ratings_df)

    # Choose a sample user_id
    sample_user_id = user_item_matrix.index[0]
    print(f"Matrix Factorization Recommendations for user {sample_user_id}:")
    recommendations = matrix_factorization_recommendations(sample_user_id, user_item_matrix, top_n=5, n_factors=20)
    print(recommendations)
