import argparse
import collections
import os

import pandas as pd

# Import Level 1: Content-Based Filtering functions
import src.rec_sys_algos.level1_content.base as l1
# Import Level 2: Collaborative Filtering functions
import src.rec_sys_algos.level2_cf.level2dot1_cf as l2
# Import Level 3: Matrix Factorization functions
import src.rec_sys_algos.level3_matrix_factorization.level3dot1_svd as l3
# Import Level 3.1: Matrix Factorization with PCA functions
import src.rec_sys_algos.level3_matrix_factorization.level3dot2_svd_with_pca as l3dot1

from src.common.data_preprocessing import preprocess_data
from src.common.user_item_matrix_components import build_user_item_matrix_components
from util.paths import DATA_PROCESSED_PATH, TEST_DATA_PROCESSED_PATH


def run_preprocessing(test_mode=False):
    # Check if processed files exist; if not, run preprocessing
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")
    reviews_csv = os.path.join(processed_data_path, "reviews_processed.csv")

    processed_files = [business_csv, ratings_csv, reviews_csv]
    missing_files = [f for f in processed_files if not os.path.exists(f)]

    if missing_files:
        print("Processed files missing. Running preprocessing steps...")
        preprocess_data(test_mode)
    else:
        print("All processed files found. Skipping preprocessing.")


def run_content_based(user_id=None, top_n=5):
    # Load preprocessed business metadata, reviews, and ratings
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    reviews_csv = os.path.join(processed_data_path, "reviews_processed.csv")
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")

    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)
    ratings_df = pd.read_csv(ratings_csv)

    if user_id is None:
        user_id = ratings_df['user_id'].iloc[0]
        print(f"No user_id provided. Using default: {user_id}")

    # Get the top 3 highest-rated restaurants for the user
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    top_rated = user_ratings.sort_values(by='stars', ascending=False).head(3)

    if top_rated.empty:
        print(f"No ratings found for user {user_id}.")
        return []

    # Build item profiles
    print("Building item profiles using Content-Based Filtering...")
    profiles = l1.build_item_profiles(business_df, reviews_df)

    # Get recommendations for each of the top 3 restaurants
    all_recommendations = []
    for _, row in top_rated.iterrows():
        business_id = row['business_id']
        recommendations = l1.recommend_similar_businesses(business_id, profiles, top_n=top_n)
        all_recommendations.extend(recommendations)

    # Combine recommendations using score-based ranking with count as a tiebreaker
    recommendation_scores = collections.defaultdict(list)
    for business_id, score in all_recommendations:
        recommendation_scores[business_id].append(score)

    # Calculate final scores and counts
    final_scores = [
        (business_id, sum(scores) / len(scores), len(scores))  # (business_id, avg_score, count)
        for business_id, scores in recommendation_scores.items()
    ]

    # Sort by average score (descending), then by count (descending)
    final_scores.sort(key=lambda x: (-x[1], -x[2]))

    # Get user's name and business names for better readability
    users_csv = os.path.join(processed_data_path, "user_processed.csv")
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    user_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)

    # Get user's name
    user_name = user_df[user_df['user_id'] == user_id]['name'].iloc[0] if not user_df[
        user_df['user_id'] == user_id].empty else "Unknown"

    # Get business names for recommendations
    business_names = []
    for rec_business_id, score, _ in final_scores:
        business_name = business_df[business_df['business_id'] == rec_business_id]['name'].iloc[0] if not business_df[
            business_df['business_id'] == rec_business_id].empty else "Unknown"
        business_names.append(f"{business_name} (ID: {rec_business_id}. Score: {score})")

    print(f"Content-Based Filtering Recommendations for business '{business_name}' (ID: {business_id}:")
    for i, name in enumerate(business_names, 1):
        print(f"{i}. {name}")


def run_collaborative(user_id=None, top_n=5):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")

    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    if user_id is None:
        user_id = user_ids[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Collaborative Filtering recommendations...")
    recommendations = l2.user_based_recommendations(user_id, matrix_components, top_n=top_n)

    # Get user's name and business names for better readability
    users_csv = os.path.join(processed_data_path, "user_processed.csv")
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    user_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)

    # Get user's name
    user_name = user_df[user_df['user_id'] == user_id]['name'].iloc[0] if not user_df[
        user_df['user_id'] == user_id].empty else "Unknown"

    # Get business names for recommendations
    business_names = []
    for business_id in recommendations:
        business_name = business_df[business_df['business_id'] == business_id]['name'].iloc[0] if not business_df[
            business_df['business_id'] == business_id].empty else "Unknown"
        business_names.append(f"{business_name}")

    print(f"Collaborative Filtering Recommendations for user '{user_name}':")
    for i, name in enumerate(business_names, 1):
        print(f"{i}. {name}")


def run_matrix_factorization(user_id=None, top_n=5, n_factors=50):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")

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
    users_csv = os.path.join(processed_data_path, "user_processed.csv")
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    user_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)

    # Get user's name
    user_name = user_df[user_df['user_id'] == user_id]['name'].iloc[0] if not user_df[
        user_df['user_id'] == user_id].empty else "Unknown"

    # Get business names for recommendations
    business_names = []
    for business_id in recommendations:
        business_name = business_df[business_df['business_id'] == business_id]['name'].iloc[0] if not business_df[
            business_df['business_id'] == business_id].empty else "Unknown"
        business_names.append(f"{business_name}")

    print(f"Matrix Factorization (SVD) Recommendations for user '{user_name}':")
    for i, name in enumerate(business_names, 1):
        print(f"{i}. {name}")


def run_matrix_factorization_with_pca(user_id=None, top_n=5, n_factors=50, variance_threshold=0.8):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")

    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    if user_id is None:
        user_id = user_ids[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Matrix Factorization with PCA (SVD-PCA) recommendations...")
    svd_model_components = l3dot1.train_svd_pca(sparse_matrix, n_factors=n_factors,
                                                variance_threshold=variance_threshold)

    recommendations = l3dot1.matrix_factorization_with_pca_recommendations(user_id, matrix_components,
                                                                           svd_model_components, top_n=top_n)

    # Get user's name and business names for better readability
    users_csv = os.path.join(processed_data_path, "user_processed.csv")
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    user_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)

    # Get user's name
    user_name = user_df[user_df['user_id'] == user_id]['name'].iloc[0] if not user_df[
        user_df['user_id'] == user_id].empty else "Unknown"

    # Get business names for recommendations
    business_names = []
    for business_id in recommendations:
        business_name = business_df[business_df['business_id'] == business_id]['name'].iloc[0] if not business_df[
            business_df['business_id'] == business_id].empty else "Unknown"
        business_names.append(f"{business_name}")

    print(f"Matrix Factorization (SVD) Recommendations for user '{user_name}':")
    for i, name in enumerate(business_names, 1):
        print(f"{i}. {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Yelp Recommendation System - Main Integration")
    parser.add_argument('--method', type=str, required=True, choices=['content', 'cf', 'svd', 'svd-pca'],
                        help="Select the recommendation method: 'content' for Content-Based, 'cf' for Collaborative Filtering, 'svd' for Matrix Factorization")
    parser.add_argument('--id', type=str, required=False,
                        help="ID of the business (for content-based) or user (for cf/svd). Defaults to the first record if not provided.")
    parser.add_argument('--top_n', type=int, default=5,
                        help="Number of recommendations to return (default is 5).")
    parser.add_argument('--n_factors', type=int, default=50,
                        help="Number of latent factors for SVD in matrix factorization (default is 20).")
    parser.add_argument('--variance_threshold', type=float, default=0.8,
                        help="Variance Threshold limit for Matrix Factorization with PCA method (default is 80% or 0.8).")
    parser.add_argument('--testing', type=bool, default=False,
                        help="Set to True to use test (5% subsample) data.")
    args = parser.parse_args()

    # Store the testing flag in an environment variable for later use
    os.environ['TESTING'] = str(args.testing)

    # Determine the processed directory based on testing flag.
    processed_data_path = TEST_DATA_PROCESSED_PATH if args.testing else DATA_PROCESSED_PATH

    # Run preprocessing before executing any recommendations
    run_preprocessing(args.testing)

    if args.method == "content":
        run_content_based(business_id=args.id, top_n=args.top_n)
    elif args.method == "cf":
        run_collaborative(user_id=args.id, top_n=args.top_n)
    elif args.method == "svd":
        run_matrix_factorization(user_id=args.id, top_n=args.top_n, n_factors=args.n_factors)
    elif args.method == "svd-pca":
        run_matrix_factorization_with_pca(user_id=args.id, top_n=args.top_n, n_factors=args.n_factors,
                                          variance_threshold=args.variance_threshold)
