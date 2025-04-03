import os

# Calculate the base path of the project.
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define common paths
DATA_PATH = os.path.join(BASE_PATH, 'data')

DATA_RAW_JSON_PATH = os.path.join(DATA_PATH, "raw", "json")
DATA_RAW_CSV_PATH = os.path.join(DATA_PATH, "raw", "csv")

DATA_PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
TEST_DATA_PROCESSED_PATH = os.path.join(DATA_PROCESSED_PATH, "test")

CACHE_PATH = os.path.join(BASE_PATH, "data", "cache")
TEST_CACHE_PATH = os.path.join(CACHE_PATH, "test")

EVALUATION_PATH = os.path.join(DATA_PATH, "evaluation")
TEST_EVALUATION_PATH = os.path.join(EVALUATION_PATH, "test")

if __name__ == "__main__":
    print(f"Base Path: {BASE_PATH}")
    print(f"Data Path: {DATA_PATH}")

    print(f"Raw JSON Data Path: {DATA_RAW_JSON_PATH}")
    print(f"Raw CSV Data Path: {DATA_RAW_CSV_PATH}")

    print(f"Processed Data Path: {DATA_PROCESSED_PATH}")
    print(f"Test Processed Data Path: {TEST_DATA_PROCESSED_PATH}")

    print(f"Cache Path: {CACHE_PATH}")
    print(f"Test Cache Path: {TEST_CACHE_PATH}")

    print(f"Evaluation Path: {EVALUATION_PATH}")
    print(f"Test Evaluation Path: {TEST_EVALUATION_PATH}")
