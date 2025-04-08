import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.common.cache import cache_results
from util.paths import DATA_PROCESSED_PATH
from src.rec_sys_algos.level1_content.base import recommend_similar_businesses


@cache_results("tfidf_embeddings_cache.pkl", force_recompute=False)
def compute_tfidf_embeddings(texts):
    """Compute TF-IDF embeddings for a list of texts."""
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(texts).toarray()


@cache_results("item_profiles_tfidf_cache.pkl", force_recompute=False)
def build_item_profiles_tfidf(business_df, reviews_df):
    """Build item profiles using TF-IDF embeddings."""
    aggregated_reviews = reviews_df.groupby('business_id')['review_text'].apply(
        lambda texts: " ".join(texts)).reset_index()
    merged_df = pd.merge(business_df, aggregated_reviews, on='business_id', how='left')
    merged_df['review_text'] = merged_df['review_text'].fillna("")
    embeddings = compute_tfidf_embeddings(merged_df['review_text'].tolist())
    return {row['business_id']: embeddings[idx] for idx, row in merged_df.iterrows()}


def recommend_similar_businesses_tfidf(business_id, item_profiles, top_n=5):
    """Recommend similar businesses using TF-IDF embeddings."""
    return recommend_similar_businesses(business_id, item_profiles, top_n)


if __name__ == "__main__":
    business_csv = os.path.join(DATA_PROCESSED_PATH, "business_processed.csv")
    reviews_csv = os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv")
    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)

    profiles = build_item_profiles_tfidf(business_df, reviews_df)
    sample_business_id = business_df['business_id'].iloc[0]
    recommendations = recommend_similar_businesses_tfidf(sample_business_id, profiles, top_n=5)
    print(f"TF-IDF Recommendations for business {sample_business_id}: {recommendations}")
