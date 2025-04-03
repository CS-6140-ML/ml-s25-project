import csv
import json
import os

import pandas as pd

from util.paths import DATA_RAW_JSON_PATH, DATA_RAW_CSV_PATH, DATA_PROCESSED_PATH, TEST_DATA_PROCESSED_PATH


##############################################
# Utility: List Manipulation & Cleaning
##############################################
def parse_list_field(value, separator=","):
    """
    Converts a comma-separated string into a list of trimmed items.
    If the input is already a list, it returns it as-is.
    If the input is None or not a string/list, it returns an empty list.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    elif isinstance(value, list):
        return value
    else:
        return []


##############################################
# Conversion Functions (JSON -> CSV)
##############################################
def convert_business_json_to_csv(
        json_path=os.path.join(DATA_RAW_JSON_PATH, "yelp_academic_dataset_business.json"),
        output_path=os.path.join(DATA_RAW_CSV_PATH, "business.csv"),
        chunk_size=10000
):
    columns_to_keep = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "categories"
    ]
    output_fields = columns_to_keep + ["categories_list"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(json_path, "r", encoding="utf8") as fin, \
            open(output_path, "w", newline="", encoding="utf8") as fout:

        writer = csv.DictWriter(fout, fieldnames=output_fields)
        writer.writeheader()

        count = 0
        chunk = []
        for line in fin:
            record = json.loads(line)
            # Skip records missing critical field 'business_id'
            if not record.get("business_id"):
                continue
            filtered = {field: record.get(field, None) for field in columns_to_keep}
            filtered["categories_list"] = parse_list_field(record.get("categories", ""))
            chunk.append(filtered)
            count += 1
            if count % chunk_size == 0:
                writer.writerows(chunk)
                print(f"Processed {count} business records...")
                chunk = []
        if chunk:
            writer.writerows(chunk)
            print(f"Processed a total of {count} business records.")


def convert_review_json_to_csv(
        json_path=os.path.join(DATA_RAW_JSON_PATH, "yelp_academic_dataset_review.json"),
        reviews_output=os.path.join(DATA_RAW_CSV_PATH, "reviews.csv"),
        ratings_output=os.path.join(DATA_RAW_CSV_PATH, "ratings.csv"),
        chunk_size=10000
):
    reviews_columns = ["review_id", "user_id", "business_id", "review_text"]
    ratings_columns = ["user_id", "business_id", "rating"]

    os.makedirs(os.path.dirname(reviews_output), exist_ok=True)
    os.makedirs(os.path.dirname(ratings_output), exist_ok=True)

    with open(json_path, "r", encoding="utf8") as fin, \
            open(reviews_output, "w", newline="", encoding="utf8") as fout_reviews, \
            open(ratings_output, "w", newline="", encoding="utf8") as fout_ratings:

        reviews_writer = csv.DictWriter(fout_reviews, fieldnames=reviews_columns)
        ratings_writer = csv.DictWriter(fout_ratings, fieldnames=ratings_columns)

        reviews_writer.writeheader()
        ratings_writer.writeheader()

        count = 0
        reviews_chunk = []
        ratings_chunk = []
        for line in fin:
            record = json.loads(line)
            if not (record.get("review_id") and record.get("user_id") and record.get("business_id")):
                continue
            review_record = {
                "review_id": record.get("review_id"),
                "user_id": record.get("user_id"),
                "business_id": record.get("business_id"),
                "review_text": record.get("text", "")
            }
            rating_record = {
                "user_id": record.get("user_id"),
                "business_id": record.get("business_id"),
                "rating": record.get("stars")
            }
            reviews_chunk.append(review_record)
            ratings_chunk.append(rating_record)
            count += 1
            if count % chunk_size == 0:
                reviews_writer.writerows(reviews_chunk)
                ratings_writer.writerows(ratings_chunk)
                print(f"Processed {count} review records...")
                reviews_chunk = []
                ratings_chunk = []
        if reviews_chunk:
            reviews_writer.writerows(reviews_chunk)
            ratings_writer.writerows(ratings_chunk)
            print(f"Processed a total of {count} review records.")


def convert_user_json_to_csv(
        json_path=os.path.join(DATA_RAW_JSON_PATH, "yelp_academic_dataset_user.json"),
        output_path=os.path.join(DATA_RAW_CSV_PATH, "user.csv"),
        chunk_size=10000
):
    columns_to_keep = [
        "user_id",
        "name",
        "review_count",
        "average_stars",
        "friends"
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(json_path, "r", encoding="utf8") as fin, \
            open(output_path, "w", newline="", encoding="utf8") as fout:

        writer = csv.DictWriter(fout, fieldnames=columns_to_keep)
        writer.writeheader()

        count = 0
        chunk = []
        for line in fin:
            record = json.loads(line)
            if not record.get("user_id"):
                continue
            filtered = {field: record.get(field, None) for field in columns_to_keep}
            chunk.append(filtered)
            count += 1
            if count % chunk_size == 0:
                writer.writerows(chunk)
                print(f"Processed {count} user records...")
                chunk = []
        if chunk:
            writer.writerows(chunk)
            print(f"Processed a total of {count} user records.")


def convert_checkin_json_to_csv(
        json_path=os.path.join(DATA_RAW_JSON_PATH, "yelp_academic_dataset_checkin.json"),
        output_path=os.path.join(DATA_RAW_CSV_PATH, "checkin.csv"),
        chunk_size=10000
):
    output_fields = ["business_id", "date", "date_list"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(json_path, "r", encoding="utf8") as fin, \
            open(output_path, "w", newline="", encoding="utf8") as fout:

        writer = csv.DictWriter(fout, fieldnames=output_fields)
        writer.writeheader()

        count = 0
        chunk = []
        for line in fin:
            record = json.loads(line)
            if not record.get("business_id"):
                continue
            filtered = {
                "business_id": record.get("business_id"),
                "date": record.get("date", None),
                "date_list": parse_list_field(record.get("date", ""))
            }
            chunk.append(filtered)
            count += 1
            if count % chunk_size == 0:
                writer.writerows(chunk)
                print(f"Processed {count} checkin records...")
                chunk = []
        if chunk:
            writer.writerows(chunk)
            print(f"Processed a total of {count} checkin records.")


##############################################
# Cleaning Functions (Post-Conversion)
##############################################
def clean_ratings(df):
    df = df.dropna(subset=["user_id", "business_id", "rating"])
    df["rating"] = df["rating"].astype(float)
    return df


def preprocess_ratings(input_csv=os.path.join(DATA_RAW_CSV_PATH, "ratings.csv"),
                       output_csv=os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_ratings(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed ratings saved to {output_csv}")
    return df


def clean_reviews(df):
    df = df.dropna(subset=["review_text"])
    return df


def preprocess_reviews(input_csv=os.path.join(DATA_RAW_CSV_PATH, "reviews.csv"),
                       output_csv=os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_reviews(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed reviews saved to {output_csv}")
    return df


def clean_business(df):
    df = df.dropna(subset=["business_id", "name"])
    return df


def preprocess_business(input_csv=os.path.join(DATA_RAW_CSV_PATH, "business.csv"),
                        output_csv=os.path.join(DATA_PROCESSED_PATH, "business_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_business(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed business data saved to {output_csv}")
    return df


def clean_user(df):
    df = df.dropna(subset=["user_id", "name"])
    return df


def preprocess_user(input_csv=os.path.join(DATA_RAW_CSV_PATH, "user.csv"),
                    output_csv=os.path.join(DATA_PROCESSED_PATH, "user_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_user(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed user data saved to {output_csv}")
    return df


def clean_checkin(df):
    df = df.dropna(subset=["business_id", "date"])
    return df


def preprocess_checkin(input_csv=os.path.join(DATA_RAW_CSV_PATH, "checkin.csv"),
                       output_csv=os.path.join(DATA_PROCESSED_PATH, "checkin_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_checkin(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed checkin data saved to {output_csv}")
    return df


##############################################
# Subsampling Function for Testing
##############################################
def subsample_processed_data(percent=5):
    """
    Subsample processed data to only include a percentage of users.
    It filters ratings, reviews, business, user, and checkin data accordingly,
    and writes the subsampled data to the TEST_DATA_PROCESSED directory using the same file names.
    """
    import shutil

    # Remove any existing test directory
    if os.path.exists(TEST_DATA_PROCESSED_PATH):
        shutil.rmtree(TEST_DATA_PROCESSED_PATH)
    os.makedirs(TEST_DATA_PROCESSED_PATH, exist_ok=True)

    # Subsample ratings
    ratings_file = os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")
    ratings_df = pd.read_csv(ratings_file)
    unique_users = ratings_df["user_id"].unique()
    total_users = len(unique_users)
    subset_n = max(1, int(total_users * percent / 100))
    subset_users = unique_users[:subset_n]
    print(f"Total users: {total_users}. Subsampling {subset_n} users ({percent}%).")
    ratings_sub = ratings_df[ratings_df["user_id"].isin(subset_users)]
    ratings_sub_file = os.path.join(TEST_DATA_PROCESSED_PATH, "ratings_processed.csv")
    ratings_sub.to_csv(ratings_sub_file, index=False)

    # Subsample reviews
    reviews_file = os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv")
    reviews_df = pd.read_csv(reviews_file)
    reviews_sub = reviews_df[reviews_df["user_id"].isin(subset_users)]
    reviews_sub_file = os.path.join(TEST_DATA_PROCESSED_PATH, "reviews_processed.csv")
    reviews_sub.to_csv(reviews_sub_file, index=False)

    # Determine businesses from these users
    businesses_from_ratings = set(ratings_sub["business_id"].unique())
    businesses_from_reviews = set(reviews_sub["business_id"].unique())
    subsampled_businesses = businesses_from_ratings.union(businesses_from_reviews)
    print(f"Subsampled businesses: {len(subsampled_businesses)}")

    # Subsample business data
    business_file = os.path.join(DATA_PROCESSED_PATH, "business_processed.csv")
    business_df = pd.read_csv(business_file)
    business_sub = business_df[business_df["business_id"].isin(subsampled_businesses)]
    business_sub_file = os.path.join(TEST_DATA_PROCESSED_PATH, "business_processed.csv")
    business_sub.to_csv(business_sub_file, index=False)

    # Subsample user data
    user_file = os.path.join(DATA_PROCESSED_PATH, "user_processed.csv")
    user_df = pd.read_csv(user_file)
    user_sub = user_df[user_df["user_id"].isin(subset_users)]
    user_sub_file = os.path.join(TEST_DATA_PROCESSED_PATH, "user_processed.csv")
    user_sub.to_csv(user_sub_file, index=False)

    # Subsample checkin data if exists
    checkin_file = os.path.join(DATA_PROCESSED_PATH, "checkin_processed.csv")
    checkin_df = pd.read_csv(checkin_file)
    checkin_sub = checkin_df[checkin_df["business_id"].isin(subsampled_businesses)]
    checkin_sub_file = os.path.join(TEST_DATA_PROCESSED_PATH, "checkin_processed.csv")
    checkin_sub.to_csv(checkin_sub_file, index=False)

    print("Subsampled data files created in TEST_DATA_PROCESSED:")
    print(f" - Ratings: {ratings_sub_file}")
    print(f" - Reviews: {reviews_sub_file}")
    print(f" - Business: {business_sub_file}")
    print(f" - User: {user_sub_file}")
    print(f" - Checkins: {checkin_sub_file}")


##############################################
# Main Execution with Testing Flag
##############################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument('--testing', type=bool, default=False,
                        help="Set to True to create test (5% subsample) files in TEST_DATA_PROCESSED folder")
    args = parser.parse_args()

    # Check and convert JSON files to CSV in data/raw/csv if needed
    if not os.path.exists(os.path.join(DATA_RAW_CSV_PATH, "business.csv")):
        print("Converting business JSON to CSV...")
        convert_business_json_to_csv()
    else:
        print("Business CSV already exists. Skipping conversion.")

    if not (os.path.exists(os.path.join(DATA_RAW_CSV_PATH, "reviews.csv")) and os.path.exists(
            os.path.join(DATA_RAW_CSV_PATH, "ratings.csv"))):
        print("Converting review JSON to CSV...")
        convert_review_json_to_csv()
    else:
        print("Review and Ratings CSV already exist. Skipping conversion.")

    if not os.path.exists(os.path.join(DATA_RAW_CSV_PATH, "user.csv")):
        print("Converting user JSON to CSV...")
        convert_user_json_to_csv()
    else:
        print("User CSV already exists. Skipping conversion.")

    if not os.path.exists(os.path.join(DATA_RAW_CSV_PATH, "checkin.csv")):
        print("Converting checkin JSON to CSV...")
        convert_checkin_json_to_csv()
    else:
        print("Checkin CSV already exists. Skipping conversion.")

    # Check and clean/process CSV files into data/processed if needed
    if not os.path.exists(os.path.join(DATA_PROCESSED_PATH, "business_processed.csv")):
        print("Cleaning and processing business data...")
        preprocess_business()
    else:
        print("Processed business data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(DATA_PROCESSED_PATH, "reviews_processed.csv")):
        print("Cleaning and processing reviews data...")
        preprocess_reviews()
    else:
        print("Processed reviews data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(DATA_PROCESSED_PATH, "ratings_processed.csv")):
        print("Cleaning and processing ratings data...")
        preprocess_ratings()
    else:
        print("Processed ratings data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(DATA_PROCESSED_PATH, "user_processed.csv")):
        print("Cleaning and processing user data...")
        preprocess_user()
    else:
        print("Processed user data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(DATA_PROCESSED_PATH, "checkin_processed.csv")):
        print("Cleaning and processing checkin data...")
        preprocess_checkin()
    else:
        print("Processed checkin data already exists. Skipping cleaning.")

    if args.testing:
        print("Subsampling processed data to 5% for testing...")
        subsample_processed_data(percent=5)
