import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.common.cache import cache_results
from util.paths import DATA_PROCESSED_PATH
from src.common.user_item_matrix_components import build_user_item_matrix_components


@cache_results("svd_pca_model_cache.pkl", force_recompute=False)
def train_svd(sparse_matrix, n_factors=50, variance_threshold=0.8):
    """
    Train SVD on the centered sparse user-item matrix and keep only principal components
    that explain sufficient variance.
    """
    # Calculate mean ratings for each user directly from sparse matrix
    row_means = np.zeros(sparse_matrix.shape[0])
    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix[i]
        if row.nnz > 0:  # if user has any ratings
            row_means[i] = row.sum() / row.nnz

    # Convert to LIL format for efficient row-wise operations
    centered_matrix = sparse_matrix.tolil()

    # Center the matrix by subtracting means from non-zero elements only
    for i in range(sparse_matrix.shape[0]):
        if row_means[i] != 0:
            rows, cols = centered_matrix[i].nonzero()
            for j in cols:
                centered_matrix[i, j] -= row_means[i]

    centered_matrix = centered_matrix.tocsr()

    max_components = min(n_factors, centered_matrix.shape[1] - 1, centered_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(centered_matrix)

    # Calculate total explained variance and show it
    total_variance = svd.explained_variance_ratio_.sum()
    print(f"Total variance explained by all {max_components} components: {total_variance:.2%}")

    # Calculate cumulative explained variance
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find components needed to reach variance threshold (relative to total available)
    adjusted_threshold = variance_threshold * total_variance
    n_components_kept = max_components

    below_threshold = cumulative_variance < adjusted_threshold
    if np.any(below_threshold):
        n_components_kept = np.sum(below_threshold)
    else:
        # If even 1 component exceeds threshold, use 1 component
        n_components_kept = 1

    # Print variance explained by the selected components
    variance_explained = cumulative_variance[n_components_kept - 1]
    print(f"Selected {n_components_kept} out of {max_components} components")
    print(
        f"These components explain {variance_explained:.2%} of total variance ({variance_explained / total_variance:.2%} of available variance)")

    # Retrain with exactly n_components_kept
    svd = TruncatedSVD(n_components=n_components_kept, random_state=42)
    U = svd.fit_transform(centered_matrix)
    Vt = svd.components_

    print(
        f"Using {n_components_kept} principal components that explain {svd.explained_variance_ratio_.sum():.2%} of variance")

    return svd, U, Vt, row_means


def matrix_factorization_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
    """
    Recommend items for a given user using the SVD-PCA model.

    For the target user, it computes predicted ratings from the SVD factors after PCA, excludes items already rated,
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
    svd_model_components = train_svd(sparse_matrix, n_factors=50, variance_threshold=0.8)
    recommendations = matrix_factorization_recommendations(sample_user_id, matrix_components, svd_model_components,
                                                           top_n=5)
    print(f"Matrix Factorization with PCA Recommendations for user {sample_user_id}:")
    print(recommendations)
