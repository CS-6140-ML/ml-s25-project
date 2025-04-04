import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.common.cache import cache_results
from src.common.sentiment_analysis import batch_sentiment_analysis
from src.common.text_embeddings import compute_embeddings
from util.paths import DATA_PROCESSED


@cache_results("item_profiles_cache.pkl", force_recompute=False)
def build_item_profiles(business_df, reviews_df):
    """
    Build content-based item profiles by aggregating review texts, computing text embeddings,
    and incorporating average sentiment.

    Returns:
        dict: Mapping from business_id to feature vector.
    """
    # Aggregate review texts per business_id (this could be cached separately)
    aggregated_reviews = aggregate_business_reviews(reviews_df)

    # Merge aggregated reviews with business metadata
    merged_df = pd.merge(business_df, aggregated_reviews, on='business_id', how='left')
    merged_df['review_text'] = merged_df['review_text'].fillna("")

    # Compute text embeddings
    embeddings = compute_embeddings(merged_df['review_text'].tolist())

    # Calculate sentiment scores (cached)
    business_sentiments = calculate_business_sentiments(reviews_df)

    # Merge average sentiment with the merged_df
    merged_df = pd.merge(merged_df, business_sentiments, on='business_id', how='left')
    merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0.0)

    # Append average sentiment as an extra feature dimension for each business
    item_profiles = {}
    for idx, row in merged_df.iterrows():
        business_id = row['business_id']
        vector = np.append(embeddings[idx], row['avg_sentiment'])
        item_profiles[business_id] = vector
    return item_profiles


@cache_results("aggregated_reviews_cache.pkl", force_recompute=False)
def aggregate_business_reviews(reviews_df):
    """Cache the aggregation of review texts per business."""
    return reviews_df.groupby('business_id')['review_text'].apply(
        lambda texts: " ".join(texts)).reset_index()


@cache_results("business_sentiments_cache.pkl", force_recompute=False)
def calculate_business_sentiments(reviews_df):
    """Cache sentiment calculations for reviews."""
    tic = time.time()
    sentiments = batch_sentiment_analysis(reviews_df['review_text'].tolist())

    # Extract polarities using vectorized operations instead of apply
    polarities = [sentiment[0] for sentiment in sentiments]

    # Create a temporary DataFrame for group-by operation
    sentiment_df = pd.DataFrame({
        'business_id': reviews_df['business_id'],
        'polarity': polarities
    })

    # Compute average sentiment using optimized group-by
    avg_sentiments = sentiment_df.groupby('business_id')['polarity'].mean().reset_index()
    avg_sentiments.rename(columns={'polarity': 'avg_sentiment'}, inplace=True)

    toc = time.time()
    minutes = (toc - tic) // 60
    seconds = (toc - tic) % 60
    print(f"Sentiment calculation took {minutes} minutes {seconds} seconds.")
    return avg_sentiments


def recommend_similar_businesses(business_id, item_profiles, top_n=5):
    """
    Recommend similar businesses based on cosine similarity between item profiles.
    Memory-efficient implementation that only computes similarities for the target business.
    """
    business_ids = list(item_profiles.keys())

    try:
        idx = business_ids.index(business_id)
    except ValueError:
        print("Business ID not found in profiles.")
        return []

    # Get the feature vector for the target business
    target_vector = item_profiles[business_id].reshape(1, -1)

    # Create array of all other business vectors
    other_business_ids = [bid for bid in business_ids if bid != business_id]
    other_vectors = np.array([item_profiles[bid] for bid in other_business_ids])

    # Compute similarity only between target and all others (not all-to-all)
    sim_scores = cosine_similarity(target_vector, other_vectors)[0]

    # Get indices of top similar businesses
    top_indices = np.argsort(sim_scores)[::-1][:top_n]

    # Return top business IDs
    return [other_business_ids[i] for i in top_indices]


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
