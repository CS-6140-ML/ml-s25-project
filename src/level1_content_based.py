import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from common.cache import cache_results
from common.sentiment_analysis import batch_sentiment_analysis
from common.text_embeddings import compute_embeddings
# Import centralized paths and common modules
from util.paths import CACHE_DIR, DATA_PROCESSED


@cache_results("item_profiles_cache.pkl", force_recompute=False)
def build_item_profiles(business_df, reviews_df):
    """
    Build content-based item profiles by aggregating review texts, computing text embeddings,
    and incorporating average sentiment.

    Returns:
        dict: Mapping from business_id to feature vector.
    """
    # Aggregate review texts per business_id
    aggregated_reviews = reviews_df.groupby('business_id')['review_text'].apply(
        lambda texts: " ".join(texts)).reset_index()

    # Merge aggregated reviews with business metadata
    merged_df = pd.merge(business_df, aggregated_reviews, on='business_id', how='left')
    merged_df['review_text'] = merged_df['review_text'].fillna("")

    # Compute text embeddings (cached)
    embeddings = compute_embeddings(merged_df['review_text'].tolist())

    # Compute a simple average sentiment for each business (average polarity)
    def avg_polarity(business_id):
        texts = reviews_df[reviews_df['business_id'] == business_id]['review_text'].tolist()
        if not texts:
            return 0.0
        sentiments = batch_sentiment_analysis(texts)
        # Extract only polarity values
        polarities = [s[0] for s in sentiments]
        return np.mean(polarities)

    merged_df['avg_sentiment'] = merged_df['business_id'].apply(avg_polarity)

    # Append average sentiment as an extra feature dimension for each business
    item_profiles = {}
    for idx, row in merged_df.iterrows():
        business_id = row['business_id']
        vector = np.append(embeddings[idx], row['avg_sentiment'])
        item_profiles[business_id] = vector
    return item_profiles


def recommend_similar_businesses(business_id, item_profiles, top_n=5):
    """
    Recommend similar businesses based on cosine similarity between item profiles.
    """
    business_ids = list(item_profiles.keys())
    feature_matrix = np.array([item_profiles[b] for b in business_ids])

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(feature_matrix)

    try:
        idx = business_ids.index(business_id)
    except ValueError:
        print("Business ID not found in profiles.")
        return []

    sim_scores = sim_matrix[idx]
    sim_scores[idx] = 0  # Exclude self from recommendations

    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    recommended_ids = [business_ids[i] for i in top_indices]
    return recommended_ids


if __name__ == "__main__":
    # Load processed data using centralized paths
    business_csv = os.path.join(DATA_PROCESSED, "business_processed.csv")
    reviews_csv = os.path.join(DATA_PROCESSED, "reviews_processed.csv")
    business_df = pd.read_csv(business_csv)
    reviews_df = pd.read_csv(reviews_csv)

    print("Building item profiles...")
    profiles = build_item_profiles(business_df, reviews_df)

    # Pick a sample business_id from the business dataframe
    sample_business_id = business_df['business_id'].iloc[0]
    recommendations = recommend_similar_businesses(sample_business_id, profiles, top_n=5)
    print(f"Content-Based Recommendations for business {sample_business_id}:")
    print(recommendations)
