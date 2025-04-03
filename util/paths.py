import os

# Calculate the base directory of the project.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define common paths
DATA_RAW_JSON = os.path.join(BASE_DIR, "data", "raw", "json")
DATA_RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")
    print(f"Raw JSON data path: {DATA_RAW_JSON}")
    print(f"Raw CSV data path: {DATA_RAW_CSV}")
    print(f"Processed data path: {DATA_PROCESSED}")
    print(f"Cache directory: {CACHE_DIR}")
