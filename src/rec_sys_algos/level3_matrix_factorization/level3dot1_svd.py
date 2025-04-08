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
    Train SVD on the centered sparse user-item matrix.
    """
    # Calculate mean ratings for each user
    row_means = np.zeros(sparse_matrix.shape[0])
    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix[i]
        if row.nnz > 0:  # If user has any ratings
            row_means[i] = row.sum() / row.nnz

    # Center the matrix by subtracting the row means
    centered_matrix = sparse_matrix.tolil()
    for i in range(sparse_matrix.shape[0]):
        if row_means[i] != 0:
            rows, cols = centered_matrix[i].nonzero()
            for j in cols:
                centered_matrix[i, j] -= row_means[i]
    centered_matrix = centered_matrix.tocsr()

    # Perform SVD
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    U = svd.fit_transform(centered_matrix)
    Vt = svd.components_

    return svd, U, Vt, row_means


def matrix_factorization_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
    """
    Recommend items for a given user using the SVD model.

    For the target user, it computes predicted ratings from the SVD factors, excludes items already rated,
    and returns the top_n items with the highest predicted ratings.
    """
    sparse_matrix, user_ids, business_ids = matrix_components
    svd, U, Vt, row_means = svd_model_components

    try:
        target_idx = user_ids.index(user_id)
    except ValueError:
        print(f"User ID {user_id} not found.")
        return []

    # Compute predicted ratings for the target user
    user_factors = U[target_idx]
    item_factors = Vt
    predicted_ratings = np.dot(user_factors, item_factors)

    # Add back the user mean to the predicted ratings
    predicted_ratings += row_means[target_idx]

    # Clip predicted ratings to ensure they are non-negative
    predicted_ratings = np.clip(predicted_ratings, 0, None)

    # Get items the user hasn't rated yet
    target_ratings = sparse_matrix[target_idx].toarray().flatten()

    # Only consider items that the target user hasn't rated
    candidate_indices = np.where(target_ratings == 0)[0]
    candidate_predictions = predicted_ratings[candidate_indices]

    # Get the indices of the top predicted items
    top_candidate_indices = candidate_indices[np.argsort(candidate_predictions)[::-1][:top_n]]
    recommendations = [(business_ids[i], candidate_predictions[i]) for i in top_candidate_indices]
    return sorted(recommendations, key=lambda x: x[1], reverse=True)


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
