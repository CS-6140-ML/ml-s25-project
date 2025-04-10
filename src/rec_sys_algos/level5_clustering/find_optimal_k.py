import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.common.cache import cache_results
from src.common.user_item_matrix_components import build_user_item_matrix_components
from util.paths import DATA_PROCESSED_PATH, CACHE_PATH

def load_data():
    """Load and prepare datasets for clustering."""
    # Load all necessary data files
    business_df = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "business_processed.csv"))
    reviews_df = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv"))
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

@cache_results("elbow_method_results.pkl", force_recompute=True)
def find_optimal_k_elbow(normalized_features, max_k=15):
    """
    Use the Elbow Method to find the optimal number of clusters.
    Args:
        normalized_features: Normalized feature matrix
        max_k: Maximum number of clusters to try
    Returns:
        List of inertias for each k value
    """
    print("\nCalculating inertias for k=1 to", max_k)
    inertias = []
    for k in range(1, max_k + 1):
        print(f"Training KMeans with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(normalized_features)
        inertias.append(kmeans.inertia_)
        print(f"k={k}, inertia={kmeans.inertia_:.2f}")
    
    return inertias

def plot_elbow_curve(inertias, max_k):
    """
    Plot the elbow curve to visualize the optimal number of clusters.
    """
    plt.figure(figsize=(10, 6))
    k_values = range(1, max_k + 1)
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k in Business Clustering')
    plt.grid(True)
    
    # Calculate the percentage decrease in inertia
    inertia_decreases = []
    for i in range(1, len(inertias)):
        decrease = ((inertias[i-1] - inertias[i]) / inertias[i-1]) * 100
        inertia_decreases.append(decrease)
        plt.annotate(f'{decrease:.1f}%', 
                    xy=(k_values[i], inertias[i]),
                    xytext=(10, 10),
                    textcoords='offset points')
    
    # Save the plot
    plot_path = os.path.join(CACHE_PATH, 'elbow_curve.png')
    plt.savefig(plot_path)
    print(f"\nElbow curve plot saved to: {plot_path}")
    
    # Find the elbow point (where the rate of decrease slows significantly)
    significant_decrease = 10  # Consider 10% as a significant decrease
    optimal_k = 1
    for i, decrease in enumerate(inertia_decreases, 2):
        if decrease < significant_decrease:
            optimal_k = i
            break
    
    print(f"\nBased on the elbow method (looking for decrease < {significant_decrease}%):")
    print(f"Suggested optimal k = {optimal_k}")
    return optimal_k

if __name__ == "__main__":
    # Load and prepare data
    print("Loading data...")
    business_df, reviews_df, ratings_df = load_data()
    
    # Extract and prepare features
    print("\nExtracting features...")
    merged_features = extract_features(business_df, ratings_df)
    normalized_features, feature_columns = prepare_features(merged_features)
    
    # Find optimal k using elbow method
    max_k = 15
    inertias = find_optimal_k_elbow(normalized_features, max_k)
    
    # Plot and analyze results
    optimal_k = plot_elbow_curve(inertias, max_k)
