# CS 6140 - ML - Spring 2025 Project

## Overview

This project is part of `CS 6140 - Machine Learning` for Spring 2025 and aims to build a Recommendation System.

## Project Structure

```
s25-project/
├── data/
│   ├── raw/                  # Original Yelp data (ratings.csv, reviews.csv, business.csv)
│   └── processed/            # Cleaned/processed files
├── notebooks/                # Jupyter notebooks for exploration and visualization
├── src/
│   ├── __init__.py
│   ├── common/               
│   │   ├── data_preprocessing.py    # Data import and cleaning routines
│   │   ├── metadata_preprocessing.py  # Preprocessing business metadata
│   │   ├── text_embeddings.py         # Compute text embeddings from Yelp reviews
│   │   ├── sentiment_analysis.py      # Extract sentiment features from Yelp reviews
│   │   └── evaluation.py              # Evaluation metrics (RMSE, Precision@K, Recall@K, F1@K, NDCG@K)
│   ├── level1_content_based.py      # L1: Content-Based Filtering (using metadata, text embeddings, and sentiment features)
│   ├── level2_cf.py                 # L2: Collaborative Filtering (memory-based and model-based approaches)
│   ├── level3_matrix_factorization.py   # L3: Matrix Factorization (e.g., SVD/PCA for latent factors)
│   ├── level4_hybrid.py             # L4: Hybrid Recommender (combining CF and content-based signals)
│   ├── level5_clustered.py          # L5: Clustered Recommendation (using clustering techniques)
│   ├── level6_graph_based.py        # L6: Graph-Based Collaborative Filtering (optional advanced approach)
│   └── main.py                      # Main file to run experiments and integrate all levels
├── .gitignore
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview, setup instructions, and roadmap
```

## How to Run

1. Clone the repository:
   `git clone [REPO]`
2. Navigate to the project directory:
   `cd ml-s25-project`
3. Run the main pipeline:
   `python main.py`

## Future Work

- ...
