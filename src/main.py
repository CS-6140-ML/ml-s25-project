import argparse
import os
import pandas as pd

# Import Level 1: Content-Based Filtering functions
import level1_content_based as l1
# Import Level 2: Collaborative Filtering functions
import level2_cf as l2
# Import Level 3: Matrix Factorization functions
import level3_matrix_factorization as l3
from util.paths import DATA_PROCESSED, TEST_DATA_PROCESSED


def run_preprocessing():
    # Check if processed files exist; if not, run preprocessing
    business_csv = os.path.join(processed_dir, "business_processed.csv")
    ratings_csv = os.path.join(processed_dir, "ratings_processed.csv")
    reviews_csv = os.path.join(processed_dir, "reviews_processed.csv")

    processed_files = [business_csv, ratings_csv, reviews_csv]
    missing_files = [f for f in processed_files if not os.path.exists(f)]

    if missing_files:
        print("Processed files missing. Running preprocessing steps...")
        from common.data_preprocessing import preprocess_ratings, preprocess_reviews, preprocess_business, \
            preprocess_checkin, preprocess_user

        preprocess_business()
        preprocess_reviews()
        preprocess_ratings()
        preprocess_user()
        preprocess_checkin()
    else:
        print("All processed files found. Skipping preprocessing.")


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
    print(f"Content-Based Recommendations for business '{business_id}': {recommendations}")


def run_collaborative(user_id=None, top_n=5):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_dir, "ratings_processed.csv")

    ratings_df = pd.read_csv(ratings_csv)
    user_item_matrix = l2.build_user_item_matrix(ratings_df)

    if user_id is None:
        user_id = user_item_matrix.index[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Collaborative Filtering recommendations...")
    recommendations = l2.user_based_recommendations(user_id, user_item_matrix, top_n=top_n)
    print(f"Collaborative Filtering Recommendations for user '{user_id}': {recommendations}")


def run_matrix_factorization(user_id=None, top_n=5, n_factors=20):
    # Load preprocessed ratings
    ratings_csv = os.path.join(processed_dir, "ratings_processed.csv")

    ratings_df = pd.read_csv(ratings_csv)
    user_item_matrix = l3.build_user_item_matrix(ratings_df)

    if user_id is None:
        user_id = user_item_matrix.index[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Matrix Factorization (SVD) recommendations...")
    recommendations = l3.matrix_factorization_recommendations(user_id, user_item_matrix, top_n=top_n,
                                                              n_factors=n_factors)
    print(f"Matrix Factorization Recommendations for user '{user_id}': {recommendations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Yelp Recommendation System - Main Integration")
    parser.add_argument('--method', type=str, required=True, choices=['content', 'collab', 'svd'],
                        help="Select the recommendation method: 'content' for Content-Based, 'collab' for Collaborative Filtering, 'svd' for Matrix Factorization")
    parser.add_argument('--id', type=str, required=False,
                        help="ID of the business (for content-based) or user (for collab/svd). Defaults to the first record if not provided.")
    parser.add_argument('--top_n', type=int, default=5,
                        help="Number of recommendations to return (default is 5).")
    parser.add_argument('--n_factors', type=int, default=20,
                        help="Number of latent factors for SVD in matrix factorization (default is 20).")
    parser.add_argument('--testing', type=bool, default=False,
                        help="Set to True to use test (5% subsample) data.")
    args = parser.parse_args()

    # Determine the processed directory based on testing flag.
    processed_dir = TEST_DATA_PROCESSED if args.testing else DATA_PROCESSED

    # Run preprocessing before executing any recommendations
    run_preprocessing()

    if args.method == "content":
        run_content_based(business_id=args.id, top_n=args.top_n)
    elif args.method == "collab":
        run_collaborative(user_id=args.id, top_n=args.top_n)
    elif args.method == "svd":
        run_matrix_factorization(user_id=args.id, top_n=args.top_n, n_factors=args.n_factors)
