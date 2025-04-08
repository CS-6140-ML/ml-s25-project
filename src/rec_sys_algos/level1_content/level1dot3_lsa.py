import os

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.cache import cache_results
from src.rec_sys_algos.level1_content import base
from util.paths import DATA_PROCESSED_PATH


@cache_results("lsa_embeddings_cache.pkl", force_recompute=False)
def compute_embeddings(texts, n_components=100):
    """Compute LSA embeddings using TF-IDF and TruncatedSVD."""
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(tfidf_matrix)


@cache_results("item_profiles_lsa_cache.pkl", force_recompute=False)
def build_item_profiles(business_df, reviews_df):
    """Build item profiles using LSA embeddings."""
    aggregated_reviews = base.aggregate_reviews(reviews_df)
    merged_df = pd.merge(business_df, aggregated_reviews, on='business_id', how='left')
    merged_df['review_text'] = merged_df['review_text'].fillna("")
    embeddings = compute_embeddings(merged_df['review_text'].tolist())
    return {row['business_id']: embeddings[idx] for idx, row in merged_df.iterrows()}


def content_based_recommendations(user_id, ratings_df, profiles, top_n=5):
    base.content_based_recommendations(user_id, ratings_df, profiles, top_n)


if __name__ == "__main__":
    business_csv = os.path.join(DATA_PROCESSED_PATH, "business_processed.csv")
    reviews_csv = os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv")
    ratings_csv = os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")
    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)
    ratings_df = pd.read_csv(reviews_csv)

    print("Building item profiles...")
    profiles = build_item_profiles(business_df, reviews_df)

    # Pick a sample business_id from the business dataframe
    sample_user_id = ratings_df['user_id'].iloc[0]
    recommendations = content_based_recommendations(sample_user_id, ratings_df, profiles, top_n=5)
    print(f"LSA powered Content-Based Recommendations for user {sample_user_id}:")
    print(recommendations)
