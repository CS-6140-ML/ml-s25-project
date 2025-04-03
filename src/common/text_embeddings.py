import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from cache import cache_results
from util.paths import CACHE_DIR

# Set device to GPU if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


@cache_results(os.path.join(CACHE_DIR, "embeddings_cache.pkl"), force_recompute=False)
def compute_embeddings(texts):
    """
    Compute and cache embeddings for a list of texts.
    Args:
        texts (list): List of text strings.
    Returns:
        numpy.ndarray: Array of embeddings.
    """
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    sample_texts = [
        "This restaurant has amazing food!",
        "I didn't like the service."
    ]
    embeddings = compute_embeddings(sample_texts)
    print("Computed embeddings shape:", np.array(embeddings).shape)
