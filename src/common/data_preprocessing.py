import csv
import json
import os
import pandas as pd
from util.paths import DATA_RAW_JSON, DATA_RAW_CSV, DATA_PROCESSED, TEST_DATA_PROCESSED


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
        json_path=os.path.join(DATA_RAW_JSON, "yelp_academic_dataset_business.json"),
        output_path=os.path.join(DATA_RAW_CSV, "business.csv"),
        chunk_size=10000
):
    """
    Streams through the Yelp business JSON file and converts it to a CSV with selected columns.
    Also converts the 'categories' field into a list as 'categories_list'.
    Skips records missing 'business_id'.
    """
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
        json_path=os.path.join(DATA_RAW_JSON, "yelp_academic_dataset_review.json"),
        reviews_output=os.path.join(DATA_RAW_CSV, "reviews.csv"),
        ratings_output=os.path.join(DATA_RAW_CSV, "ratings.csv"),
        chunk_size=10000
):
    """
    Streams through the Yelp review JSON file and converts it into two CSVs:
      - reviews.csv: Contains review_id, user_id, business_id, review_text.
      - ratings.csv: Contains user_id, business_id, rating.
    Skips records missing 'review_id', 'user_id', or 'business_id'.
    """
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
        json_path=os.path.join(DATA_RAW_JSON, "yelp_academic_dataset_user.json"),
        output_path=os.path.join(DATA_RAW_CSV, "user.csv"),
        chunk_size=10000
):
    """
    Streams through the Yelp user JSON file and converts it to a CSV.
    Skips records missing 'user_id'.
    """
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
        json_path=os.path.join(DATA_RAW_JSON, "yelp_academic_dataset_checkin.json"),
        output_path=os.path.join(DATA_RAW_CSV, "checkin.csv"),
        chunk_size=10000
):
    """
    Streams through the Yelp checkin JSON file and converts it to a CSV.
    Converts the 'date' field (which may be a comma-separated string) into a list called 'date_list'.
    Skips records missing 'business_id'.
    """
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
    """
    Drop rows with missing critical values and convert ratings to float.
    """
    df = df.dropna(subset=["user_id", "business_id", "rating"])
    df["rating"] = df["rating"].astype(float)
    return df


def preprocess_ratings(input_csv=os.path.join(DATA_RAW_CSV, "ratings.csv"),
                       output_csv=os.path.join(DATA_PROCESSED, "ratings_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_ratings(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed ratings saved to {output_csv}")
    return df


def clean_reviews(df):
    """
    Drop rows with missing review text.
    """
    df = df.dropna(subset=["review_text"])
    return df


def preprocess_reviews(input_csv=os.path.join(DATA_RAW_CSV, "reviews.csv"),
                       output_csv=os.path.join(DATA_PROCESSED, "reviews_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_reviews(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed reviews saved to {output_csv}")
    return df


def clean_business(df):
    """
    Drop rows missing critical fields like business_id or name.
    """
    df = df.dropna(subset=["business_id", "name"])
    return df


def preprocess_business(input_csv=os.path.join(DATA_RAW_CSV, "business.csv"),
                        output_csv=os.path.join(DATA_PROCESSED, "business_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_business(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed business data saved to {output_csv}")
    return df


def clean_user(df):
    """
    Drop rows missing critical fields like user_id or name.
    """
    df = df.dropna(subset=["user_id", "name"])
    return df


def preprocess_user(input_csv=os.path.join(DATA_RAW_CSV, "user.csv"),
                    output_csv=os.path.join(DATA_PROCESSED, "user_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_user(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed user data saved to {output_csv}")
    return df


def clean_checkin(df):
    """
    Drop rows missing critical fields like business_id or date.
    """
    df = df.dropna(subset=["business_id", "date"])
    return df


def preprocess_checkin(input_csv=os.path.join(DATA_RAW_CSV, "checkin.csv"),
                       output_csv=os.path.join(DATA_PROCESSED, "checkin_processed.csv")):
    df = pd.read_csv(input_csv)
    df = clean_checkin(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed checkin data saved to {output_csv}")
    return df


##############################################
# Main Execution with Testing Flag
##############################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument('--testing', type=bool, default=False,
                        help="Set to True to create test files in data/processed/test/")
    args = parser.parse_args()

    # Determine output directory based on testing flag
    processed_dir = TEST_DATA_PROCESSED if args.testing else DATA_PROCESSED

    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Check and convert JSON files to CSV in data/raw/csv if needed
    if not os.path.exists(os.path.join(DATA_RAW_CSV, "business.csv")):
        print("Converting business JSON to CSV...")
        convert_business_json_to_csv()
    else:
        print("Business CSV already exists. Skipping conversion.")

    if not (os.path.exists(os.path.join(DATA_RAW_CSV, "reviews.csv")) and os.path.exists(
            os.path.join(DATA_RAW_CSV, "ratings.csv"))):
        print("Converting review JSON to CSV...")
        convert_review_json_to_csv()
    else:
        print("Review and Ratings CSV already exist. Skipping conversion.")

    if not os.path.exists(os.path.join(DATA_RAW_CSV, "user.csv")):
        print("Converting user JSON to CSV...")
        convert_user_json_to_csv()
    else:
        print("User CSV already exists. Skipping conversion.")

    if not os.path.exists(os.path.join(DATA_RAW_CSV, "checkin.csv")):
        print("Converting checkin JSON to CSV...")
        convert_checkin_json_to_csv()
    else:
        print("Checkin CSV already exists. Skipping conversion.")

    # Check and clean/process CSV files into data/processed if needed
    if not os.path.exists(os.path.join(processed_dir, "business_processed.csv")):
        print("Cleaning and processing business data...")
        preprocess_business(input_csv=os.path.join(DATA_RAW_CSV, "business.csv"),
                            output_csv=os.path.join(processed_dir, "business_processed.csv"))
    else:
        print("Processed business data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(processed_dir, "reviews_processed.csv")):
        print("Cleaning and processing reviews data...")
        preprocess_reviews(input_csv=os.path.join(DATA_RAW_CSV, "reviews.csv"),
                           output_csv=os.path.join(processed_dir, "reviews_processed.csv"))
    else:
        print("Processed reviews data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(processed_dir, "ratings_processed.csv")):
        print("Cleaning and processing ratings data...")
        preprocess_ratings(input_csv=os.path.join(DATA_RAW_CSV, "ratings.csv"),
                           output_csv=os.path.join(processed_dir, "ratings_processed.csv"))
    else:
        print("Processed ratings data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(processed_dir, "user_processed.csv")):
        print("Cleaning and processing user data...")
        preprocess_user(input_csv=os.path.join(DATA_RAW_CSV, "user.csv"),
                        output_csv=os.path.join(processed_dir, "user_processed.csv"))
    else:
        print("Processed user data already exists. Skipping cleaning.")

    if not os.path.exists(os.path.join(processed_dir, "checkin_processed.csv")):
        print("Cleaning and processing checkin data...")
        preprocess_checkin(input_csv=os.path.join(DATA_RAW_CSV, "checkin.csv"),
                           output_csv=os.path.join(processed_dir, "checkin_processed.csv"))
    else:
        print("Processed checkin data already exists. Skipping cleaning.")
