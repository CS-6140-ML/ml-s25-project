# Import Level 1: Content-Based Filtering functions
import argparse
import os

import pandas as pd

import src.level1_content_based as l1
# Import Level 2: Collaborative Filtering functions
import src.level2_cf as l2
# Import Level 3: Matrix Factorization functions
import src.level3_matrix_factorization as l3
from src.common.user_item_matrix_components import build_user_item_matrix_components
from src.main import run_preprocessing
from util.paths import TEST_DATA_PROCESSED, DATA_PROCESSED
from collections import defaultdict



def hybrid_recommend(user_id, top_n):
    # if weights is None or len(weights) != 3:
    #     print("weights not defined")
    #     return []
    # w1, w2, w3 = weights

    cb_result = run_content_based(user_id, top_n=top_n)  # top_n*2 to allow overlap
    cf_result = run_collaborative(user_id, top_n=top_n)
    svd_result = run_matrix_factorization(user_id, top_n=top_n)

    # Combine all results
    combined_results = cb_result + cf_result + svd_result

    # Deduplicate by business name, keeping the one with highest rating
    unique_businesses = {}
    for business in combined_results:
        name = business['business_id']
        rating = business['stars']
        if name not in unique_businesses or rating > unique_businesses[name]['stars']:
            unique_businesses[name] = business  # Keep the one with higher rating

    # Sort unique businesses by rating (descending)
    sorted_unique = sorted(unique_businesses.values(), key=lambda x: x['stars'], reverse=True)

    # Get only the business names
    ranked_names = [b['business_id'] for b in sorted_unique]

    return ranked_names

def run_collaborative(user_id=None, top_n=5):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_dir, "ratings_processed.csv")

    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    if user_id is None:
        user_id = user_ids[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Collaborative Filtering recommendations...")
    recommendations = l2.user_based_recommendations(user_id, matrix_components, top_n=top_n)

    # Get user's name and business names for better readability
    users_csv = os.path.join(processed_dir, "user_processed.csv")
    business_csv = os.path.join(processed_dir, "business_processed.csv")
    user_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)

    business_data_list = []
    for business_id in recommendations:
        match = business_df[business_df['business_id'] == business_id]
        if not match.empty:
            business_data_list.append(match.iloc[0])  # Append the entire row as a Series
        else:
            business_data_list.append({"business_id": business_id, "name": "Unknown"})

    return business_data_list

def run_content_based(business_id=None, top_n=5):
    # Load preprocessed business metadata and reviews
    business_csv = os.path.join(processed_dir, "business_processed.csv")
    reviews_csv = os.path.join(processed_dir, "reviews_processed.csv")

    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)

    if business_id is None:
        business_id = business_df['business_id'].iloc[0]
        print(f"No business_id provided. Using default: {business_id}")

    print("Building item profiles using Content-Based Filtering...")
    profiles = l1.build_item_profiles(business_df, reviews_df)
    recommendations = l1.recommend_similar_businesses(business_id, profiles, top_n=top_n)

    # Get business names for better readability
    business_csv = os.path.join(processed_dir, "business_processed.csv")
    business_df = pd.read_csv(business_csv)

    # Get business names for recommendations
    business_data_list = []
    for business_id in recommendations:
        match = business_df[business_df['business_id'] == business_id]
        if not match.empty:
            business_data_list.append(match.iloc[0])  # Append the entire row as a Series
        else:
            business_data_list.append({"business_id": business_id, "name": "Unknown"})

    return business_data_list

def run_matrix_factorization(user_id=None, top_n=5, n_factors=20):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_dir, "ratings_processed.csv")

    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    if user_id is None:
        user_id = user_ids[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Matrix Factorization (SVD) recommendations...")
    svd_model_components = l3.train_svd(sparse_matrix, n_factors=n_factors)

    recommendations = l3.matrix_factorization_recommendations(user_id, matrix_components, svd_model_components,
                                                              top_n=top_n)

    # Get user's name and business names for better readability
    users_csv = os.path.join(processed_dir, "user_processed.csv")
    business_csv = os.path.join(processed_dir, "business_processed.csv")
    user_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)

    # Get business names for recommendations
    business_data_list = []
    for business_id in recommendations:
        match = business_df[business_df['business_id'] == business_id]
        if not match.empty:
            business_data_list.append(match.iloc[0])  # Append the entire row as a Series
        else:
            business_data_list.append({"business_id": business_id, "name": "Unknown"})

    return business_data_list


if __name__ == "__main__":

    # args = parser.parse_args()

    # processed_dir = TEST_DATA_PROCESSED if args.testing else DATA_PROCESSED
    processed_dir = DATA_PROCESSED

    # hybrid_recommend(business_id=args.id, top_n=args.top_n)
    hybrid_recommend(1000, top_n=5)

