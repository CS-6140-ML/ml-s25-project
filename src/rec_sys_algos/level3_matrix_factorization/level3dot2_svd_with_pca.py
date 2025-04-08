import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.common.cache import cache_results
from src.common.user_item_matrix_components import build_user_item_matrix_components
from src.rec_sys_algos.level3_matrix_factorization import base
from util.paths import DATA_PROCESSED_PATH


@cache_results("svd_pca_model_cache.pkl", force_recompute=False)
def train_svd_with_pca(sparse_matrix, n_factors, variance_threshold):
    svd, U, Vt, row_means = base.train_base_svd(sparse_matrix, n_factors)

    # Center the matrix by subtracting the row means
    centered_matrix = sparse_matrix.tolil()

    for i in range(sparse_matrix.shape[0]):
        if row_means[i] != 0:
            rows, cols = centered_matrix[i].nonzero()
            for j in cols:
                centered_matrix[i, j] -= row_means[i]
    centered_matrix = centered_matrix.tocsr()

    # Calculate total explained variance and show it
    total_variance = svd.explained_variance_ratio_.sum()

    # Calculate cumulative explained variance
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find components needed to reach variance threshold (relative to total available)
    adjusted_threshold = variance_threshold * total_variance
    n_components_kept = n_factors

    below_threshold = cumulative_variance <= adjusted_threshold
    if variance_threshold == 1.0:
        # If threshold is 1.0, keep all components
        pass
    elif np.any(below_threshold):
        n_components_kept = np.sum(below_threshold)
    else:
        # If even 1 component exceeds threshold, use 1 component
        n_components_kept = 1

    # Print variance explained by the selected components
    variance_explained = cumulative_variance[n_components_kept - 1]

    # Retrain with exactly n_components_kept
    svd = TruncatedSVD(n_components=n_components_kept, random_state=42)
    U = svd.fit_transform(centered_matrix)
    Vt = svd.components_

    print(
        f"Using {n_components_kept} principal components that explain {svd.explained_variance_ratio_.sum():.2%} of variance")

    return svd, U, Vt, row_means, n_components_kept, variance_explained, total_variance


def train_svd(sparse_matrix, n_factors=50, variance_threshold=0.8):
    """
    Train SVD on the centered sparse user-item matrix and keep only principal components
    that explain sufficient variance.
    """
    svd_with_pca_model_components = train_svd_with_pca(sparse_matrix, n_factors, variance_threshold)

    svd, U, Vt, row_means, n_components_kept, variance_explained, total_variance = svd_with_pca_model_components

    print(f"Total variance explained by all {n_factors} components: {total_variance:.2%}")
    print(f"Selected {n_components_kept} out of {n_factors} components")
    print(
        f"These components explain {variance_explained:.2%} of total variance ({variance_explained / total_variance:.2%} of available variance)")

    return svd, U, Vt, row_means


def matrix_factorization_based_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
    print("Matrix Factorization-based Recommendations using SVD with PCA...")
    return base.matrix_factorization_based_recommendations(user_id, matrix_components, svd_model_components, top_n)


if __name__ == "__main__":
    ratings_csv = os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    sample_user_id = user_ids[0]

    # Train SVD with 80% variance threshold (based on Pareto's Principle)
    svd_model_components = train_svd(sparse_matrix, n_factors=50, variance_threshold=0.8)
    recommendations = matrix_factorization_based_recommendations(sample_user_id, matrix_components,
                                                                 svd_model_components, top_n=5)
    print(f"Matrix Factorization with PCA Recommendations for user {sample_user_id}:")
    print(recommendations)
