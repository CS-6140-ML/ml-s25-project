import os

import pandas as pd


def convert_business_json_to_csv(
        json_path="data/raw/json/yelp_academic_dataset_business.json",
        output_path="data/raw/csv/business.csv"
):
    """
    Reads the Yelp business JSON and converts it to a CSV with selected columns.
    """
    df = pd.read_json(json_path, lines=True)

    # Keep relevant columns (adjust as needed)
    columns_to_keep = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "categories"
    ]
    df = df[columns_to_keep]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Converted {json_path} to {output_path} with {len(df)} rows.")


def convert_review_json_to_csv(
        json_path="data/raw/json/yelp_academic_dataset_review.json",
        reviews_output="data/raw/csv/reviews.csv",
        ratings_output="data/raw/csv/ratings.csv"
):
    """
    Reads the Yelp review JSON and splits it into two CSVs:
    - reviews.csv: For text-based processing (includes review details).
    - ratings.csv: For rating-based collaborative filtering.
    """
    df = pd.read_json(json_path, lines=True)

    # Prepare reviews CSV: Rename 'text' to 'review_text'
    reviews_df = df[["review_id", "user_id", "business_id", "text"]].copy()
    reviews_df.rename(columns={"text": "review_text"}, inplace=True)
    os.makedirs(os.path.dirname(reviews_output), exist_ok=True)
    reviews_df.to_csv(reviews_output, index=False)
    print(f"Converted {json_path} to {reviews_output} with {len(reviews_df)} rows.")

    # Prepare ratings CSV: Rename 'stars' to 'rating'
    ratings_df = df[["user_id", "business_id", "stars"]].copy()
    ratings_df.rename(columns={"stars": "rating"}, inplace=True)
    os.makedirs(os.path.dirname(ratings_output), exist_ok=True)
    ratings_df.to_csv(ratings_output, index=False)
    print(f"Converted {json_path} to {ratings_output} with {len(ratings_df)} rows.")


def convert_user_json_to_csv(
        json_path="data/raw/json/yelp_academic_dataset_user.json",
        output_path="data/raw/csv/user.csv"
):
    """
    Reads the Yelp user JSON and converts it to a CSV.
    """
    df = pd.read_json(json_path, lines=True)

    columns_to_keep = [
        "user_id",
        "name",
        "review_count",
        "average_stars",
        "friends"
    ]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Converted {json_path} to {output_path} with {len(df)} rows.")


if __name__ == "__main__":
    convert_business_json_to_csv()
    convert_review_json_to_csv()
    convert_user_json_to_csv()
