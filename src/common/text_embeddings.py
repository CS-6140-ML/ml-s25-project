from sentence_transformers import SentenceTransformer

# Load a pre-trained model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_embeddings(texts):
    """
    Compute embeddings for a list of texts.

    Args:
        texts (list of str): List of texts.

    Returns:
        numpy.ndarray: Array of embeddings.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    sample_texts = [
        "This restaurant has amazing food!",
        "I didn't like the service."
    ]
    embeddings = compute_embeddings(sample_texts)
    print("Computed embeddings:\n", embeddings)
