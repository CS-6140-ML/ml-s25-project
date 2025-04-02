import pandas as pd


def preprocess_business_metadata(input_file='data/processed/business_processed.csv',
                                 output_file='data/processed/business_metadata.csv'):
    df = pd.read_csv(input_file)
    # Example: split categories if stored as a comma-separated string
    if 'categories' in df.columns:
        df['categories_list'] = df['categories'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    df.to_csv(output_file, index=False)
    print(f"Business metadata processed and saved to {output_file}")
    return df


if __name__ == "__main__":
    preprocess_business_metadata()
