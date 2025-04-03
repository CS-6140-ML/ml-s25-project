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
    - Content-based filtering
        - `python -m src.main.py --method content --id [BUSINESS_ID] --top_n 5 --testing True`
        - Replace `[BUSINESS_ID]` with the ID of the business you want to get recommendations for.
    - Collaborative filtering
        - `python -m src.main.py --method cf --id [USER_ID] --top_n 5 --testing True`
        - Replace `[USER_ID]` with the ID of the user you want to get recommendations for.
    - Matrix factorization
        - `python -m src.main.py --method svd --id [USER_ID] --top_n 5 --testing True`
        - Replace `[USER_ID]` with the ID of the user you want to get recommendations for.
    - Common Parameters
        - `--method`: Method to use for recommendation (content, cf, svd, hybrid, clustered)
            - Mandatory
        - `--id`: ID of the business/user to get recommendations for
            - Optional, default 1st ID in the dataset
            - For content-based filtering, use business ID.
            - For other methods, use user ID.
        - `--top_n`: Number of recommendations to return
            - Optional, default 5
        - `--testing`: Whether to run in testing mode (True/False)
            - Optional, default False

## Future Work

- ...
