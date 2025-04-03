import pandas as pd
from scipy.sparse import coo_matrix


def build_user_item_matrix_components(ratings_df):
    user_ids = ratings_df['user_id'].unique()
    business_ids = ratings_df['business_id'].unique()
    user_to_index = {user: idx for idx, user in enumerate(user_ids)}
    business_to_index = {biz: idx for idx, biz in enumerate(business_ids)}

    rows = ratings_df['user_id'].map(user_to_index).values
    cols = ratings_df['business_id'].map(business_to_index).values
    data = ratings_df['rating'].values

    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(business_ids))).tocsr()
    return sparse_matrix, user_ids.tolist(), business_ids.tolist()


if __name__ == "__main__":
    from util.paths import DATA_PROCESSED_PATH

    ratings_csv = DATA_PROCESSED_PATH + "/ratings_processed.csv"
    ratings_df = pd.read_csv(ratings_csv)
    matrix_components = build_user_item_matrix_components(ratings_df)
    sample_user_id = matrix_components[1][0]
    print(f"User IDs: {matrix_components[1]}")
    print(f"Business IDs: {matrix_components[2]}")
    print(f"Sparse Matrix Shape: {matrix_components[0].shape}")
    print(f"Sample User ID: {sample_user_id}")
