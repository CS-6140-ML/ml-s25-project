import argparse
import os

import pandas as pd

# Import Level 1: Content-Based Filtering functions
import src.rec_sys_algos.level1_content.level1dot1_tf_idf as l1dot1
import src.rec_sys_algos.level1_content.level1dot2_sentence_transformer as l1dot2
import src.rec_sys_algos.level1_content.level1dot3_lsa as l1dot3
# Import Level 2: Collaborative Filtering functions
import src.rec_sys_algos.level2_cf.level2dot1_cf as l2dot1
# Import Level 3: Matrix Factorization functions
import src.rec_sys_algos.level3_matrix_factorization.level3dot1_svd as l3dot1
import src.rec_sys_algos.level3_matrix_factorization.level3dot2_svd_with_pca as l3dot2
from src.common.data_preprocessing import preprocess_data
from src.common.user_item_matrix_components import build_user_item_matrix_components
from util.paths import DATA_PROCESSED_PATH, TEST_DATA_PROCESSED_PATH


def validate_processed_files(processed_data_path):
    """Ensure all required processed files exist."""
    required_files = [
        os.path.join(processed_data_path, "business_processed.csv"),
        os.path.join(processed_data_path, "ratings_processed.csv"),
        os.path.join(processed_data_path, "reviews_processed.csv"),
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Running preprocessing...")
        preprocess_data(test_mode=(processed_data_path == TEST_DATA_PROCESSED_PATH))


def run_preprocessing(test_mode=False):
    """Run preprocessing if required files are missing."""
    processed_data_path = TEST_DATA_PROCESSED_PATH if test_mode else DATA_PROCESSED_PATH
    validate_processed_files(processed_data_path)


def print_recommendations(user_id, recommendations):
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
    for rec_business_id, score in recommendations:
        business_name = business_df[business_df['business_id'] == rec_business_id]['name'].iloc[0] if not business_df[
            business_df['business_id'] == rec_business_id].empty else "Unknown"
        business_names.append(f"{business_name} (ID: {rec_business_id}. Score: {round(score, 2)})")

    print(f"Recommendations for user '{user_name}':")
    for i, name in enumerate(business_names, 1):
        print(f"{i}. {name}")


def run_content_based(user_id=None, top_n=5, method='tfidf'):
    """Run Content-Based Filtering."""
    business_csv = os.path.join(processed_data_path, "business_processed.csv")
    reviews_csv = os.path.join(processed_data_path, "reviews_processed.csv")
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")

    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)
    ratings_df = pd.read_csv(ratings_csv)

    if user_id is None:
        user_id = ratings_df['user_id'].iloc[0]
        print(f"No user_id provided. Using default: {user_id}")

    l1 = l1dot1
    if method == 'tfidf':
        l1 = l1dot1
    elif method == 'sentence_transformer':
        l1 = l1dot2
    elif method == 'lsa':
        l1 = l1dot3

    # Build item profiles
    print("Building item profiles using Content-Based Filtering...")
    profiles = l1.build_item_profiles(business_df, reviews_df)

    recommendations = l1.content_based_recommendations(user_id, ratings_df, profiles, top_n=top_n)

    print_recommendations(user_id, recommendations)


def run_collaborative(user_id=None, top_n=5, method='cf'):
    """Run Collaborative Filtering."""
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    if user_id is None:
        user_id = user_ids[0]
        print(f"No user_id provided. Using default: {user_id}")

    l2 = l2dot1
    if method == 'cf':
        l2 = l2dot1

    print("Generating Collaborative Filtering recommendations...")
    recommendations = l2.user_based_recommendations(user_id, matrix_components, top_n=top_n)

    print_recommendations(user_id, recommendations)


def run_matrix_factorization(user_id=None, top_n=5, n_factors=50, variance_threshold=0.8, method='svd'):
    """Run Matrix Factorization (SVD)."""
    ratings_csv = os.path.join(processed_data_path, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)

    sparse_matrix, user_ids, business_ids = matrix_components

    if user_id is None:
        user_id = user_ids[0]
        print(f"No user_id provided. Using default: {user_id}")

    l3 = l3dot1
    if method == 'svd':
        l3 = l3dot1
        print("Generating Matrix Factorization (SVD) recommendations...")
        svd_model_components = l3.train_svd(sparse_matrix, n_factors=n_factors)
    elif method == 'svd_with_pca':
        l3 = l3dot2
        print("Generating Matrix Factorization with PCA (SVD-PCA) recommendations...")
        svd_model_components = l3.train_svd(sparse_matrix, n_factors=n_factors, variance_threshold=variance_threshold)

    recommendations = l3.matrix_factorization_recommendations(user_id, matrix_components, svd_model_components,
                                                              top_n=top_n)

    print_recommendations(user_id, recommendations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Yelp Recommendation System - Main Integration")
    parser.add_argument('--method', type=str, required=True,
                        choices=['content_tf_idf', 'content_sentence_transformer', 'content_lsa', 'cf', 'svd',
                                 'svd_with_pca'],
                        help="Select the recommendation method: 'content_tf_idf' for TF-TDF powered Content-Based, \
                        'content_sentence_transformer' for Sentence Transformer powered Content-Based, \
                        'content_lsa' for LSA powered Content-Based, 'cf' for Collaborative Filtering, \
                        'svd' for Matrix Factorization, 'svd_with_pca' for PCA-enhanced Matrix Factorization.")
    parser.add_argument('--id', type=str, required=False,
                        help="ID of the business (for content-based) or user (for cf/svd). Defaults to the first record if not provided.")
    parser.add_argument('--top_n', type=int, default=5,
                        help="Number of recommendations to return (default is 5).")
    parser.add_argument('--n_factors', type=int, default=50,
                        help="Number of latent factors for SVD in matrix factorization (default is 50).")
    parser.add_argument('--variance_threshold', type=float, default=0.8,
                        help="Variance Threshold for PCA in SVD-PCA method (default is 80% or 0.8).")
    parser.add_argument('--testing', action='store_true',
                        help="Set to True to use test (5% subsample) data.")
    args = parser.parse_args()

    # Set the testing flag in the environment
    os.environ['TESTING'] = str(args.testing)

    # Determine the processed data path
    processed_data_path = TEST_DATA_PROCESSED_PATH if args.testing else DATA_PROCESSED_PATH

    # Validate processed files
    run_preprocessing(args.testing)

    # Execute the selected recommendation method
    if args.method == "content_tf_idf":
        run_content_based(business_id=args.id, top_n=args.top_n, method='tf_idf')
    elif args.method == "content_sentence_transformer":
        run_content_based(business_id=args.id, top_n=args.top_n, method='sentence_transformer')
    elif args.method == "content_lsa":
        run_content_based(business_id=args.id, top_n=args.top_n, method='lsa')
    elif args.method == "cf":
        run_collaborative(user_id=args.id, top_n=args.top_n, method='cf')
    elif args.method == "svd":
        run_matrix_factorization(user_id=args.id, top_n=args.top_n, n_factors=args.n_factors, method='svd')
    elif args.method == "svd_with_pca":
        run_matrix_factorization(user_id=args.id, top_n=args.top_n, n_factors=args.n_factors,
                                 variance_threshold=args.variance_threshold, method='svd_with_pca')
