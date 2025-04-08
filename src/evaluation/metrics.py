import math

from sklearn.metrics import root_mean_squared_error


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    """
    return root_mean_squared_error(y_true, y_pred)


def precision_at_k(recommended, relevant, k):
    """
    Compute precision at k.

    Args:
        recommended (list): List of recommended item IDs.
        relevant (set): Set of relevant item IDs.
        k (int): Number of top recommendations to consider.

    Returns:
        float: Precision at k.
    """
    recommended_k = recommended[:k]
    num_relevant = sum([1 for item in recommended_k if item in relevant])
    return num_relevant / k


def recall_at_k(recommended, relevant, k):
    """
    Compute recall at k.
    """
    recommended_k = recommended[:k]
    num_relevant = sum([1 for item in recommended_k if item in relevant])
    if len(relevant) == 0:
        return 0.0
    return num_relevant / len(relevant)


def f1_at_k(recommended, relevant, k):
    """
    Compute F1 score at k.
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended, relevant, k):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        recommended (list): List of recommended items (binary relevance assumed).
        relevant (set): Set of relevant item IDs.
        k (int): Number of top recommendations to consider.

    Returns:
        float: NDCG value at k.
    """
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        rel = 1 if item in relevant else 0
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # Compute Ideal DCG (IDCG)
    ideal_rels = [1] * min(len(relevant), k)
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


if __name__ == "__main__":
    # Test evaluation functions with dummy data
    y_true = [3, 4, 5, 2]
    y_pred = [2.5, 3.5, 5, 2]
    print("RMSE:", rmse(y_true, y_pred))

    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}
    k = 3
    print("Precision@3:", precision_at_k(recommended, relevant, k))
    print("Recall@3:", recall_at_k(recommended, relevant, k))
    print("F1@3:", f1_at_k(recommended, relevant, k))
    print("NDCG@3:", ndcg_at_k(recommended, relevant, k))
