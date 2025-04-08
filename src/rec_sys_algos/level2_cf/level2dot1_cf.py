import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from util.paths import DATA_PROCESSED_PATH
from src.common.user_item_matrix_components import build_user_item_matrix_components


def user_based_recommendations(user_id, matrix_components, top_n=5, num_similar=10):
    sparse_matrix, user_ids, business_ids = matrix_components

    try:
        target_idx = user_ids.index(user_id)
    except ValueError:
        print("User ID not found in the matrix.")
        return []

    # Compute cosine similarity for the target user row vs. all users.
    target_vector = sparse_matrix[target_idx]
    # This returns a 1-D array of similarities for the target user.
    sim_scores = cosine_similarity(target_vector, sparse_matrix)[0]

    # Get indices of similar users, excluding the target user.
    similar_indices = sim_scores.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != target_idx]
    top_similar_indices = similar_indices[:num_similar]

    # Candidate items: items rated by top similar users but not by the target user.
    target_rated = set(np.where(target_vector.toarray().flatten() > 0)[0])
    candidate_indices = set()
    for i in top_similar_indices:
        candidate_indices.update(np.where(sparse_matrix[i].toarray().flatten() > 0)[0])
    candidate_indices = candidate_indices - target_rated

    predicted_ratings = {}
    for j in candidate_indices:
        numerator = 0.0
        denominator = 0.0
        for i in top_similar_indices:
            rating = sparse_matrix[i, j]
            if rating > 0:
                numerator += sim_scores[i] * rating
                denominator += sim_scores[i]
        if denominator > 0:
            predicted_ratings[business_ids[j]] = numerator / denominator

    recommended_items = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item for item, score in recommended_items]


if __name__ == "__main__":
    ratings_csv = DATA_PROCESSED_PATH + "/ratings_processed.csv"
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    sample_user_id = user_ids[0]
    recommendations = user_based_recommendations(sample_user_id, matrix_components, top_n=5)
    print(f"Collaborative Filtering Recommendations for user {sample_user_id}:")
    print(recommendations)
