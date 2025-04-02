import os

import pandas as pd


def load_csv(filename, data_dir='data/raw/csv'):
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath)


def clean_ratings(df):
    # Drop missing values and ensure correct datatype for ratings
    df = df.dropna()
    df['rating'] = df['rating'].astype(float)
    return df


def preprocess_ratings(data_dir='data/raw/csv', output_dir='data/processed'):
    df = load_csv('ratings.csv', data_dir)
    df = clean_ratings(df)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ratings_processed.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed ratings saved to {output_file}")
    return df


def clean_reviews(df):
    # Drop rows with missing review texts
    df = df.dropna(subset=['review_text'])
    return df


def preprocess_reviews(data_dir='data/raw/csv', output_dir='data/processed'):
    df = load_csv('reviews.csv', data_dir)
    df = clean_reviews(df)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reviews_processed.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed reviews saved to {output_file}")
    return df


def clean_business(df):
    # Ensure key fields are present
    df = df.dropna(subset=['business_id', 'name'])
    return df


def preprocess_business(data_dir='data/raw/csv', output_dir='data/processed'):
    df = load_csv('business.csv', data_dir)
    df = clean_business(df)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'business_processed.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed business data saved to {output_file}")
    return df


if __name__ == "__main__":
    preprocess_ratings()
    preprocess_reviews()
    preprocess_business()
