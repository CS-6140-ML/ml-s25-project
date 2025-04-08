import os

import pandas as pd

from src.common.user_item_matrix_components import build_user_item_matrix_components
from src.rec_sys_algos.level1_content.level1dot1_tf_idf import build_item_profiles, content_based_recommendations
from src.rec_sys_algos.level2_cf.level2dot1_cf import cf_based_recommendations
from src.rec_sys_algos.level3_matrix_factorization.level3dot1_svd import train_svd, \
    matrix_factorization_based_recommendations
from util.paths import DATA_PROCESSED_PATH


def run_content_based_recommendations(user_id, business_df, reviews_df, ratings_df, top_n=5):
    # Content-Based Recommendations
    print("Generating Content-Based recommendations...")
    profiles = build_item_profiles(business_df, reviews_df)
    cb_recommendations = content_based_recommendations(user_id, ratings_df, profiles, top_n=top_n)
    return cb_recommendations


def run_cf_based_recommendations(user_id, ratings_df, top_n):
    # Collaborative Filtering Recommendations
    print("Generating Collaborative Filtering recommendations...")
    matrix_components = build_user_item_matrix_components(ratings_df)
    cf_recommendations = cf_based_recommendations(user_id, matrix_components, top_n=top_n)
    return cf_recommendations, matrix_components


def run_matrix_factorization_based_recommendations(user_id, top_n=5, n_factors=50):
    # Matrix Factorization Recommendations
    print("Generating Matrix Factorization recommendations...")
    matrix_components = build_user_item_matrix_components(ratings_df)
    sparse_matrix, user_ids, business_ids = matrix_components
    svd_model_components = train_svd(sparse_matrix, nfactors=n_factors)
    mf_recommendations = matrix_factorization_based_recommendations(user_id, matrix_components, svd_model_components,
                                                                    top_n=top_n)
    return mf_recommendations


def hybrid_recommendations(cb_recommendations, cf_recommendations, mf_recommendations, top_n=5,
                           weights=(0.33, 0.33, 0.34)):
    """
    Generate hybrid recommendations by combining content-based, collaborative filtering, 
    and matrix factorization recommendations.
    """
    w1, w2, w3 = weights

    # Combine recommendations with weights
    combined_scores = {}
    for rec_list, weight in zip([cb_recommendations, cf_recommendations, mf_recommendations], [w1, w2, w3]):
        for business_id, score in rec_list:
            if business_id not in combined_scores:
                combined_scores[business_id] = 0
            combined_scores[business_id] += score * weight

    # Sort combined recommendations by score
    recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations


if __name__ == "__main__":
    # Load preprocessed data
    business_csv = os.path.join(DATA_PROCESSED_PATH, "business_processed.csv")
    reviews_csv = os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv")
    ratings_csv = os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")
    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)
    ratings_df = pd.read_csv(ratings_csv)

    # Sample user ID
    sample_user_id = ratings_df['user_id'].iloc[0]

    cb_recommendations = run_content_based_recommendations(sample_user_id, business_df, reviews_df, ratings_df,
                                                           top_n=5)

    cf_recommendations = run_cf_based_recommendations(sample_user_id, ratings_df, top_n=5)

    mf_recommendations = run_matrix_factorization_based_recommendations(sample_user_id, top_n=5, n_factors=50)

    # Generate hybrid recommendations
    print("Generating Hybrid Recommendations...")
    recommendations = hybrid_recommendations(business_df, reviews_df, ratings_df, top_n=5)
    print(f"Hybrid Recommendations for user {sample_user_id}:")
    print(recommendations)
