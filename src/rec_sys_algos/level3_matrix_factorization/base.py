import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.common.cache import cache_results


@cache_results("svd_model_base_cache.pkl", force_recompute=False)
def train_base_svd(sparse_matrix, n_factors=50):
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


def matrix_factorization_based_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
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

    # Add back the user mean to get the final predictions
    predicted_ratings += row_means[target_idx]

    # Get items the user hasn't rated yet
    target_ratings = sparse_matrix[target_idx].toarray().flatten()

    # Only consider items that the target user hasn't rated
    candidate_indices = np.where(target_ratings == 0)[0]
    candidate_predictions = predicted_ratings[candidate_indices]

    # Get the indices of the top predicted items
    top_candidate_indices = candidate_indices[np.argsort(candidate_predictions)[::-1][:top_n]]
    recommendations = [(business_ids[i], candidate_predictions[i]) for i in top_candidate_indices]
    return sorted(recommendations, key=lambda x: (-x[1], x[0]))
