import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.common.cache import cache_results
from src.common.user_item_matrix_components import build_user_item_matrix_components
from util.paths import DATA_PROCESSED_PATH


def load_data():
    """Load and prepare datasets for clustering."""
    print("Loading data from:", DATA_PROCESSED_PATH)
    # Load all necessary data files
    business_df = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "business_processed.csv"))
    print("Business data loaded, shape:", business_df.shape)
    reviews_df = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv"))
    print("Reviews data loaded, shape:", reviews_df.shape)
    ratings_df = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv"))
    print("Ratings data loaded, shape:", ratings_df.shape)
    
    # Print column names to verify structure
    print("\nBusiness columns:", business_df.columns.tolist())
    print("Ratings columns:", ratings_df.columns.tolist())
    return business_df, reviews_df, ratings_df


def extract_features(business_df, ratings_df):
    """
    Extract and engineer features for clustering.
    Combines business characteristics with user interaction patterns.
    """
    # Aggregate business metrics
    business_features = business_df[["business_id", "stars", "review_count"]].copy()
    
    # Calculate average rating and review count per business
    business_aggs = ratings_df.groupby("business_id").agg({
        "rating": ["mean", "count", "std"]
    }).reset_index()
    
    # Flatten multi-level column names
    business_aggs.columns = ["business_id", "avg_user_rating", "user_rating_count", "rating_std"]
    
    # Merge business features with aggregated metrics
    merged_features = pd.merge(business_features, business_aggs, on="business_id", how="left")
    
    # Fill missing values with means for numeric columns only
    numeric_columns = merged_features.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        merged_features[col] = merged_features[col].fillna(merged_features[col].mean())
    
    print("Feature columns before preparation:", merged_features.columns.tolist())
    print("Sample of numeric values:\n", merged_features[numeric_columns].head())
    
    return merged_features


@cache_results("kmeans_features_cache.pkl", force_recompute=False)
def prepare_features(merged_features):
    """
    Prepare and normalize features for clustering.
    Uses StandardScaler to normalize all features to same scale.
    """
    # Select only numeric columns for clustering
    numeric_columns = merged_features.select_dtypes(include=[np.number]).columns.tolist()
    # Remove business_id if it's somehow numeric
    if 'business_id' in numeric_columns:
        numeric_columns.remove('business_id')
    
    print("Using numeric features for clustering:", numeric_columns)
    features = merged_features[numeric_columns]
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    print("Feature matrix shape:", normalized_features.shape)
    return normalized_features, numeric_columns


@cache_results("kmeans_model_cache.pkl", force_recompute=False)
def train_kmeans(normalized_features, n_clusters=8):
    
    # Initialize and train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(normalized_features)
    return kmeans


def get_cluster_for_items(user_ratings, merged_features, kmeans, normalized_features):
    
    # Get items rated 4 or above by the user
    high_rated_items = user_ratings[user_ratings["rating"] >= 4]["business_id"].tolist()
    
    if not high_rated_items:
        # If no high ratings, consider all rated items
        high_rated_items = user_ratings["business_id"].tolist()
    
    # Find indices of these items in our feature matrix
    item_indices = merged_features[merged_features["business_id"].isin(high_rated_items)].index
    
    if len(item_indices) == 0:
        return None
    
    # Get clusters for these items
    item_clusters = kmeans.predict(normalized_features[item_indices])
    
    # Return most common cluster
    return np.bincount(item_clusters).argmax()


def cluster_based_recommendations(user_id, matrix_components, merged_features, 
                                normalized_features, kmeans, top_n=5):
    """
    Generate recommendations based on clustering.
    Finds user's preferred cluster and recommends highly rated items from that cluster.
    """
    sparse_matrix, user_ids, business_ids = matrix_components
    
    try:
        # Get user's ratings
        user_idx = user_ids.index(user_id)
        user_ratings = pd.DataFrame({
            "business_id": business_ids,
            "rating": sparse_matrix[user_idx].toarray().flatten()
        })
        user_ratings = user_ratings[user_ratings["rating"] > 0]
    except ValueError:
        print("User {} not found in the matrix.".format(user_id))
        return []
    
    # Find user's preferred cluster
    preferred_cluster = get_cluster_for_items(user_ratings, merged_features, 
                                            kmeans, normalized_features)
    
    if preferred_cluster is None:
        return []
    
    # Get all items in the preferred cluster
    cluster_labels = kmeans.labels_
    cluster_items = merged_features[cluster_labels == preferred_cluster]
    
    # Filter out items user has already rated
    rated_items = set(user_ratings["business_id"])
    new_items = cluster_items[~cluster_items["business_id"].isin(rated_items)]
    
    # Sort by predicted quality (using business rating and review count)
    new_items = new_items.copy()  # Create a copy to avoid the warning
    new_items.loc[:, "score"] = new_items["stars"] * np.log1p(new_items["review_count"])
    recommendations = new_items.nlargest(top_n, "score")
    
    # Return recommendations with scores
    return [(row["business_id"], row["score"]) 
            for _, row in recommendations.iterrows()]


if __name__ == "__main__":
    try:
        print("Starting KNN clustering recommendation system...")
        print("Current working directory:", os.getcwd())
        
        # Load data
        business_df, reviews_df, ratings_df = load_data()
        print("Data loaded successfully")
        
        # Build user-item matrix components
        matrix_components = build_user_item_matrix_components(ratings_df)
        print("User-item matrix built")
        
        # Extract and prepare features
        print("Extracting features...")
        merged_features = extract_features(business_df, ratings_df)
        print("Features extracted")
        
        normalized_features, feature_columns = prepare_features(merged_features)
        print("Features prepared: {}".format(", ".join(feature_columns)))
        
        # Train KMeans model
        kmeans = train_kmeans(normalized_features)
        print("KMeans model trained with {} clusters".format(kmeans.n_clusters))
        
        # Generate sample recommendations
        sparse_matrix, user_ids, business_ids = matrix_components
        sample_user_id = user_ids[0]
        
        recommendations = cluster_based_recommendations(
            sample_user_id, matrix_components, merged_features, 
            normalized_features, kmeans
        )
        
        print("\nCluster-based Recommendations for user {}:".format(sample_user_id))
        for business_id, score in recommendations:
            print("Business: {}, Score: {:.2f}".format(business_id, score))
            
    except Exception as e:
        import traceback
        print("An error occurred:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()
