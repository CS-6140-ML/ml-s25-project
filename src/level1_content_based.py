import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from common.sentiment_analysis import batch_sentiment_analysis
# Import common utilities
from common.text_embeddings import compute_embeddings


def build_item_profiles(business_df, reviews_df):
    """
    Build content-based item profiles by aggregating review texts and computing sentiment features.

    Parameters:
        business_df (pd.DataFrame): Business metadata (assumed to include 'business_id').
        reviews_df (pd.DataFrame): Reviews data (assumed to include 'business_id' and 'review_text').

    Returns:
        dict: Mapping from business_id to its feature vector.
    """
    # Aggregate review texts per business_id
    aggregated_reviews = reviews_df.groupby('business_id')['review_text'].apply(
        lambda texts: " ".join(texts)).reset_index()

    # Merge aggregated reviews with business metadata
    merged_df = pd.merge(business_df, aggregated_reviews, on='business_id', how='left')
    merged_df['review_text'] = merged_df['review_text'].fillna("")

    # Compute text embeddings for each business using the aggregated review text
    embeddings = compute_embeddings(merged_df['review_text'].tolist())

    # Compute a simple average sentiment for each business (average polarity)
    # For each business, process its reviews individually then take the average polarity.
    def avg_polarity(business_id):
        texts = reviews_df[reviews_df['business_id'] == business_id]['review_text'].tolist()
        if not texts:
            return 0.0
        sentiments = batch_sentiment_analysis(texts)
        # sentiments is a list of (polarity, subjectivity) tuples; we take the polarity
        polarities = [s[0] for s in sentiments]
        return np.mean(polarities)

    merged_df['avg_sentiment'] = merged_df['business_id'].apply(avg_polarity)

    # For simplicity, append the average sentiment as an extra feature dimension.
    item_profiles = {}
    for idx, row in merged_df.iterrows():
        business_id = row['business_id']
        # Append sentiment to the embedding vector
        vector = np.append(embeddings[idx], row['avg_sentiment'])
        item_profiles[business_id] = vector
    return item_profiles


def recommend_similar_businesses(business_id, item_profiles, top_n=5):
    """
    Recommend similar businesses based on cosine similarity between item profiles.

    Parameters:
        business_id: The reference business id.
        item_profiles: Dict mapping business_id to feature vector.
        top_n: Number of recommendations to return.

    Returns:
        list: Business IDs for recommended items.
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
    # Exclude self by setting similarity to zero
    sim_scores[idx] = 0

    # Get indices for top_n similar items
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    recommended_ids = [business_ids[i] for i in top_indices]
    return recommended_ids


if __name__ == "__main__":
    # Load preprocessed data (ensure these CSVs exist in data/processed)
    business_df = pd.read_csv("data/processed/business_metadata.csv")
    reviews_df = pd.read_csv("data/processed/reviews_processed.csv")

    print("Building item profiles...")
    profiles = build_item_profiles(business_df, reviews_df)

    # Pick a sample business_id from the business dataframe
    sample_business_id = business_df['business_id'].iloc[0]
    print(f"Recommendations for business {sample_business_id}:")
    recommendations = recommend_similar_businesses(sample_business_id, profiles, top_n=5)
    print(recommendations)
