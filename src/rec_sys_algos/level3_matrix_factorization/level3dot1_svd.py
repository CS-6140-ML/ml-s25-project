import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.common.cache import cache_results
from util.paths import DATA_PROCESSED_PATH
from src.common.user_item_matrix_components import build_user_item_matrix_components


@cache_results("svd_model_cache.pkl", force_recompute=False)
def train_svd(sparse_matrix, n_factors=50):
    """
    Train SVD on the sparse user-item matrix.
    """
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    U = svd.fit_transform(sparse_matrix)
    Vt = svd.components_

    return svd, U, Vt


def matrix_factorization_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
    """
    Recommend items for a given user using the SVD model.

    For the target user, it computes predicted ratings from the SVD factors, excludes items already rated,
    and returns the top_n items with the highest predicted ratings.
    """
    sparse_matrix, user_ids, business_ids = matrix_components
    svd, U, Vt = svd_model_components

    try:
        target_idx = user_ids.index(user_id)
    except ValueError:
        print(f"User ID {user_id} not found.")
        return []

    # Compute predicted ratings for the target user
    user_factors = U[target_idx]
    item_factors = Vt
    predicted_ratings = np.dot(user_factors, item_factors)

    # Get items the user hasn't rated yet
    target_ratings = sparse_matrix[target_idx].toarray().flatten()

    # Only consider items that the target user hasn't rated
    candidate_indices = np.where(target_ratings == 0)[0]
    candidate_predictions = predicted_ratings[candidate_indices]

    # Get the indices of the top predicted items
    top_candidate_indices = candidate_indices[np.argsort(candidate_predictions)[::-1][:top_n]]
    recommended_items = [business_ids[i] for i in top_candidate_indices]
    return recommended_items


if __name__ == "__main__":
    ratings_csv = os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    sample_user_id = user_ids[0]

    # Train SVD
    svd_model_components = train_svd(sparse_matrix, n_factors=50)
    recommendations = matrix_factorization_recommendations(sample_user_id, matrix_components, svd_model_components,
                                                           top_n=5)
    print(f"Matrix Factorization Recommendations for user {sample_user_id}:")
    print(recommendations)
