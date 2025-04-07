import os
import pickle
from functools import wraps

from util.paths import CACHE_PATH, TEST_CACHE_PATH


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
            base_cache_dir = CACHE_PATH if not test_mode else TEST_CACHE_PATH

            final_cache_path = os.path.join(base_cache_dir, cache_filename)

            # Create a unique cache key based on function name and parameters
            cache_key = f"{func.__name__}_{hash(str(args))}__{hash(str(sorted(kwargs.items())))}"

            if os.path.exists(final_cache_path) and not force_recompute:
                with open(final_cache_path, 'rb') as f:
                    cache_dict = pickle.load(f)
                    if cache_key in cache_dict:
                        print(f"Loading cached results for {func.__name__}")
                        return cache_dict[cache_key]

            print(f"Computing results for {func.__name__}")
            result = func(*args, **kwargs)
            os.makedirs(os.path.dirname(final_cache_path), exist_ok=True)

            cache_dict = {}
            if os.path.exists(final_cache_path):
                with open(final_cache_path, 'rb') as f:
                    cache_dict = pickle.load(f)

            cache_dict[cache_key] = result
            with open(final_cache_path, 'wb') as f:
                pickle.dump(cache_dict, f)

            return result

        return wrapper

    return decorator
