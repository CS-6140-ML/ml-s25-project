# CS 6140 - ML - Spring 2025 Project

## Overview

This project is part of `CS 6140 - Machine Learning` for Spring 2025 and aims to build a Recommendation System.

## Project Structure

```
ml-s25-project/
├── data/
│   ├── cache/
│   │   ├── test/
│   │   │   └── ...
│   │   └── ...
│   ├── processed/
│   │   ├── test/
│   │   │   └── ...
│   │   └── ...
│   └── raw/
│   │   ├── csv/
│   │   │   └── ...
│   │   └── json/
│   │       └── ...
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── data_preprocessing.py
│   │   ├── metadata_preprocessing.py
│   │   ├── text_embeddings.py
│   │   ├── sentiment_analysis.py
│   │   └── evaluation.py
│   ├── level1_content_based.py
│   ├── level2_cf.py
│   ├── level3_matrix_factorization.py
│   ├── level3dot1_matrix_factorization_with_pca.py
│   ├── level4_hybrid.py [TBD]
│   ├── level5_clustered.py [TBD]
│   ├── level6_graph_based.py [TBD]
│   └── main.py
├── util/
│   ├── __init__.py
│   └── paths.py
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository:
    - `git clone [REPO]`
2. Navigate to the project directory:
    - `cd ml-s25-project`
3. Install the required dependencies:
    - `pip install -r requirements.txt`
4. Prepare your data:
    - Place your Yelp data files in the `data/raw/json` directory
5. Run the recommendation system:
    - Content-based Filtering
        - TF-IDF / Sentence Transformer / LSA
            - `python -m src.main --method [METHOD] --id [USER_ID] --top_n 5 --testing`
            - `[METHOD] = content_tf_idf, content_sentence_transformer, content_lsa`
    - Collaborative Filtering
        - `python -m src.main --method cf --id [USER_ID] --top_n 5 --testing`
    - Matrix Factorization
        - SVD
            - `python -m src.main --method svd --id [USER_ID] --top_n 5 --testing`
        - SVD with PCA
            - `python -m src.main --method svd_with_pca --id [USER_ID] --top_n 5 --variance_threshold 0.8 --testing`
    - Common Parameters
        - `--method`: Method to use for recommendation (content, cf, svd, hybrid, clustered)
            - Mandatory
        - `--id`: ID of the user to get recommendations for
            - Optional, default 1st ID in the dataset
        - `--top_n`: Number of recommendations to return
            - Optional, default 5
        - `--variance_threshold`: Threshold for Variance limit used for PCA
            - Optional, default 0.8
        - `--testing`: Whether to run in testing mode
            - Optional, default to False if the tag is not present

## Future Work

- ...
