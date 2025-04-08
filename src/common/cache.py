import os
import pickle
import hashlib
from functools import wraps
import pandas as pd
from util.paths import CACHE_PATH, TEST_CACHE_PATH


def compute_stable_hash(obj):
    """
    Compute a stable hash for the given object.
    For pandas DataFrames, serialize to CSV and hash the content.
    """
    if isinstance(obj, pd.DataFrame):
        obj = obj.to_csv(index=False)  # Serialize DataFrame to CSV
    elif isinstance(obj, (list, dict, tuple, set)):
        obj = str(sorted(obj))  # Sort and convert to string for consistent hashing
    else:
        obj = str(obj)  # Convert to string for other types

    return hashlib.md5(obj.encode()).hexdigest()


def cache_results(cache_filename, force_recompute=False):
    """
    Decorator to cache the output of a function to disk.
    Uses a stable hash for arguments to generate a unique cache key.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if in test mode
            test_mode = os.environ.get("TESTING", "False") == "True"
            base_cache_dir = CACHE_PATH if not test_mode else TEST_CACHE_PATH

            final_cache_path = os.path.join(base_cache_dir, cache_filename)

            # Create a unique cache key using stable hashes of arguments
            args_hash = '_'.join(compute_stable_hash(arg) for arg in args)
            kwargs_hash = '_'.join(f"{k}:{compute_stable_hash(v)}" for k, v in sorted(kwargs.items()))
            cache_key = f"{func.__name__}_{args_hash}_{kwargs_hash}"

            if os.path.exists(final_cache_path) and not force_recompute:
                try:
                    with open(final_cache_path, 'rb') as f:
                        cache_dict = pickle.load(f)
                        if cache_key in cache_dict:
                            print(f"Loading cached results for {func.__name__}")
                            return cache_dict[cache_key]
                except (pickle.PickleError, EOFError):
                    print(f"Cache file corrupted, recomputing")

            print(f"Computing results for {func.__name__}")
            result = func(*args, **kwargs)
            os.makedirs(os.path.dirname(final_cache_path), exist_ok=True)

            cache_dict = {}
            if os.path.exists(final_cache_path):
                try:
                    with open(final_cache_path, 'rb') as f:
                        cache_dict = pickle.load(f)
                except (pickle.PickleError, EOFError):
                    # If cache file is corrupted, start with empty dict
                    pass

            cache_dict[cache_key] = result
            print(f"Saving results to {final_cache_path}")
            with open(final_cache_path, 'wb') as f:
                pickle.dump(cache_dict, f)

            return result

        return wrapper

    return decorator
