import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

from src.common.cache import cache_results
from util.paths import DATA_PROCESSED_PATH
from src.common.user_item_matrix_components import build_user_item_matrix_components


@cache_results("svd_pca_model_cache.pkl", force_recompute=False)
def train_svd_pca(sparse_matrix, n_factors=50, variance_threshold=0.8):
    """
    Train SVD on the centered sparse user-item matrix and keep only principal components
    that explain sufficient variance.

    Args:
        sparse_matrix: Sparse user-item matrix
        n_factors: Maximum number of factors to consider
        variance_threshold: Target explained variance (0.0-1.0)

    Returns:
        tuple: (svd_model, U matrix, Vt matrix, row_means)
    """
    # Center the data by subtracting row means (user means)
    matrix_dense = sparse_matrix.toarray()
    mask = matrix_dense > 0
    row_means = np.zeros(matrix_dense.shape[0])

    # Calculate mean ratings for each user
    for i in range(matrix_dense.shape[0]):
        if np.any(mask[i]):
            row_means[i] = matrix_dense[i, mask[i]].mean()

    # Create a matrix of repeated row means only for rated items
    row_means_matrix = np.zeros_like(matrix_dense)
    for i, mean in enumerate(row_means):
        rated_items = sparse_matrix[i].nonzero()[1]
        row_means_matrix[i, rated_items] = mean

    # Center the matrix by subtracting user means
    centered_matrix = sparse_matrix - csr_matrix(row_means_matrix)

    # Initialize SVD with maximum components
    max_components = min(n_factors, centered_matrix.shape[1] - 1, centered_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(centered_matrix)

    # Calculate cumulative explained variance
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find minimum components needed to reach variance threshold
    n_components_kept = 1  # Default to at least 1 component
    if np.any(cumulative_variance >= variance_threshold):
        n_components_kept = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Retrain with exactly n_components_kept
    svd = TruncatedSVD(n_components=n_components_kept, random_state=42)
    U = svd.fit_transform(centered_matrix)
    Vt = svd.components_

    print(
        f"Using {n_components_kept} principal components that explain {svd.explained_variance_ratio_.sum():.2%} of variance")

    return svd, U, Vt, row_means


def matrix_factorization_with_pca_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
    """
    Recommend items for a given user using the SVD-PCA model with centered data.

    Args:
        user_id: ID of the target user
        matrix_components: Tuple of (sparse_matrix, user_ids, business_ids)
        svd_model_components: Tuple of (svd_model, U, Vt, row_means)
        top_n: Number of recommendations to return

    Returns:
        list: Business IDs of recommended items
    """
    sparse_matrix, user_ids, business_ids = matrix_components
    svd, U, Vt, row_means = svd_model_components

    try:
        target_idx = user_ids.index(user_id)
    except ValueError:
        print(f"User ID {user_id} not found.")
        return []

    # Compute predicted ratings using the dimensionally reduced matrices
    # No slicing needed as U and Vt already have the correct dimensions after retraining
    user_factors = U[target_idx]
    item_factors = Vt
    predicted_ratings = np.dot(user_factors, item_factors)

    # Add back the user mean to get the final predictions
    predicted_ratings += row_means[target_idx]

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

    # Train SVD with 80% variance threshold (based on Pareto's Principle)
    svd_model_components = train_svd_pca(sparse_matrix, n_factors=50, variance_threshold=0.8)

    recommendations = matrix_factorization_with_pca_recommendations(sample_user_id, matrix_components,
                                                                    svd_model_components,
                                                                    top_n=5)
    print(f"Matrix Factorization with PCA Recommendations for user {sample_user_id}:")
    print(recommendations)
