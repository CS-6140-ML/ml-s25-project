import os
import pickle
from functools import wraps
from util.paths import CACHE_DIR, TEST_CACHE_DIR


def cache_results(cache_filename, force_recompute=False):
    """
    Decorator to cache the output of a function to disk.
    If the environment variable TESTING is set to "True", it uses the test cache directory.
    Accepts only a filename; the full path is constructed automatically.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if in test mode
            test_mode = os.environ.get("TESTING", "False") == "True"
            base_cache_dir = CACHE_DIR if not test_mode else TEST_CACHE_DIR

            final_cache_path = os.path.join(base_cache_dir, cache_filename)

            if os.path.exists(final_cache_path) and not force_recompute:
                print(f"Loading cached results from {final_cache_path}")
                with open(final_cache_path, "rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            os.makedirs(os.path.dirname(final_cache_path), exist_ok=True)
            with open(final_cache_path, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper

    return decorator
