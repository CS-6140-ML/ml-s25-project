import argparse
import os

import pandas as pd

# Import Level 1: Content-Based Filtering functions
import level1_content_based as l1
# Import Level 2: Collaborative Filtering functions
import level2_cf as l2
# Import Level 3: Matrix Factorization functions
import level3_matrix_factorization as l3


def run_preprocessing():
    # Check if processed files exist; if not, run preprocessing
    processed_files = ["data/processed/ratings_processed.csv",
                       "data/processed/reviews_processed.csv",
                       "data/processed/business_processed.csv"]
    missing_files = [f for f in processed_files if not os.path.exists(f)]

    if missing_files:
        print("Processed files missing. Running preprocessing steps...")
        from common.data_preprocessing import preprocess_ratings, preprocess_reviews, preprocess_business

        preprocess_ratings()
        preprocess_reviews()
        preprocess_business()
    else:
        print("All processed files found. Skipping preprocessing.")


def run_content_based(business_id=None, top_n=5):
    # Load preprocessed business metadata and reviews
    business_df = pd.read_csv("data/processed/business_processed.csv")
    reviews_df = pd.read_csv("data/processed/reviews_processed.csv")

    if business_id is None:
        business_id = business_df['business_id'].iloc[0]
        print(f"No business_id provided. Using default: {business_id}")

    print("Building item profiles using Content-Based Filtering...")
    profiles = l1.build_item_profiles(business_df, reviews_df)
    recommendations = l1.recommend_similar_businesses(business_id, profiles, top_n=top_n)
    print(f"Content-Based Recommendations for business '{business_id}': {recommendations}")


def run_collaborative(user_id=None, top_n=5):
    # Load preprocessed ratings
    ratings_df = pd.read_csv("data/processed/ratings_processed.csv")
    user_item_matrix = l2.build_user_item_matrix(ratings_df)

    if user_id is None:
        user_id = user_item_matrix.index[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Collaborative Filtering recommendations...")
    recommendations = l2.user_based_recommendations(user_id, user_item_matrix, top_n=top_n)
    print(f"Collaborative Filtering Recommendations for user '{user_id}': {recommendations}")


def run_matrix_factorization(user_id=None, top_n=5, n_factors=20):
    # Load preprocessed ratings
    ratings_df = pd.read_csv("data/processed/ratings_processed.csv")
    user_item_matrix = l3.build_user_item_matrix(ratings_df)

    if user_id is None:
        user_id = user_item_matrix.index[0]
        print(f"No user_id provided. Using default: {user_id}")

    print("Generating Matrix Factorization (SVD) recommendations...")
    recommendations = l3.matrix_factorization_recommendations(user_id, user_item_matrix, top_n=top_n,
                                                              n_factors=n_factors)
    print(f"Matrix Factorization Recommendations for user '{user_id}': {recommendations}")


if __name__ == "__main__":
    # Run preprocessing before executing any recommendations
    run_preprocessing()

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Hybrid Yelp Recommendation System - Main Integration"
    )
    parser.add_argument(
        '--method', type=str, required=True, choices=['content', 'collab', 'svd'],
        help="Select the recommendation method: 'content' for Content-Based, 'collab' for Collaborative Filtering, 'svd' for Matrix Factorization"
    )
    parser.add_argument(
        '--id', type=str, required=False,
        help="ID of the business (for content-based) or user (for collab/svd). Defaults to the first record if not provided."
    )
    parser.add_argument(
        '--top_n', type=int, default=5,
        help="Number of recommendations to return (default is 5)."
    )
    parser.add_argument(
        '--n_factors', type=int, default=20,
        help="Number of latent factors to use for SVD in matrix factorization (default is 20)."
    )

    args = parser.parse_args()

    if args.method == "content":
        run_content_based(business_id=args.id, top_n=args.top_n)
    elif args.method == "collab":
        run_collaborative(user_id=args.id, top_n=args.top_n)
    elif args.method == "svd":
        run_matrix_factorization(user_id=args.id, top_n=args.top_n, n_factors=args.n_factors)
