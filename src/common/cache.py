import os
import pickle
from functools import wraps


def cache_results(cache_path, force_recompute=False):
    """
    Decorator to cache the output of a function to disk.

    Args:
        cache_path (str): File path where the results will be stored.
        force_recompute (bool): If True, ignore cache and recompute.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_path) and not force_recompute:
                print(f"Loading cached results from {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper

    return decorator
