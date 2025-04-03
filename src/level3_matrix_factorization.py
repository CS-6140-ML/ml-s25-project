import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from common.cache import cache_results
from util.paths import CACHE_DIR, DATA_PROCESSED


def build_user_item_matrix(ratings_df):
    """
    Build a user-item matrix from the ratings dataframe.
    """
    return ratings_df.pivot(index='user_id', columns='business_id', values='rating')


@cache_results("svd_model_cache.pkl", force_recompute=False)
def train_svd(user_item_matrix, n_factors=20):
    """
    Train SVD on the user-item matrix and cache the model and its factors.

    Returns:
        tuple: (svd model, U matrix, Vt matrix)
    """
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    U = svd.fit_transform(user_item_matrix.fillna(0))
    Vt = svd.components_
    return svd, U, Vt


def matrix_factorization_recommendations(user_id, user_item_matrix, svd_model, U, Vt, top_n=5):
    """
    Recommend items using the trained SVD model.
    """
    predicted_ratings = np.dot(U, Vt)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index,
                                        columns=user_item_matrix.columns)

    # Exclude items already rated by the user
    user_actual_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_actual_ratings[user_actual_ratings.isna()].index
    predictions = predicted_ratings_df.loc[user_id, unrated_items]

    recommended_items = predictions.sort_values(ascending=False).head(top_n).index.tolist()
    return recommended_items


if __name__ == "__main__":
    ratings_csv = os.path.join(DATA_PROCESSED, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_csv)
    user_item_matrix = build_user_item_matrix(ratings_df)
    sample_user_id = user_item_matrix.index[0]
    svd_model, U, Vt = train_svd(user_item_matrix, n_factors=20)
    recommendations = matrix_factorization_recommendations(sample_user_id, user_item_matrix, svd_model, U, Vt, top_n=5)
    print(f"Matrix Factorization Recommendations for user {sample_user_id}:")
    print(recommendations)
