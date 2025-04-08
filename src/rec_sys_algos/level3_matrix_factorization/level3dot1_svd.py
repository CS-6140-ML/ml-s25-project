import os

import pandas as pd

from src.common.cache import cache_results
from src.common.user_item_matrix_components import build_user_item_matrix_components
from src.rec_sys_algos.level3_matrix_factorization import base
from util.paths import DATA_PROCESSED_PATH


@cache_results("svd_model_cache.pkl", force_recompute=False)
def train_svd(sparse_matrix, n_factors=50):
    return base.train_base_svd(sparse_matrix, n_factors)


def matrix_factorization_based_recommendations(user_id, matrix_components, svd_model_components, top_n=5):
    print("Matrix Factorization-based Recommendations using SVD...")
    return base.matrix_factorization_based_recommendations(user_id, matrix_components, svd_model_components, top_n)


if __name__ == "__main__":
    ratings_csv = os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    sample_user_id = user_ids[0]

    # Train SVD
    svd_model_components = train_svd(sparse_matrix, n_factors=50)
    recommendations = matrix_factorization_based_recommendations(sample_user_id, matrix_components,
                                                                 svd_model_components, top_n=5)
    print(f"Matrix Factorization Recommendations for user {sample_user_id}:")
    print(recommendations)
