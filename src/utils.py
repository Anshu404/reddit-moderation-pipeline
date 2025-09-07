import time
import pandas as pd
import functools
from typing import Callable, Any

def log_section(title: str):
    """
    Prints a formatted section header to the console for better readability of logs.

    Args:
        title (str): The title of the section to be printed.
    """
    header = f" {title} "
    print("\n" + header.center(90, "="))
    print()

def safe_print_df(df: pd.DataFrame, max_rows: int = 10):
    """
    Prints the head of a pandas DataFrame using a controlled display context
    to ensure consistent and readable output.

    Args:
        df (pd.DataFrame): The DataFrame to print.
        max_rows (int): The maximum number of rows to display.
    """
    with pd.option_context("display.max_rows", max_rows, "display.width", 140):
        print(df.head(max_rows))

def memory_mb(df: pd.DataFrame) -> float:
    """
    Calculates the memory usage of a pandas DataFrame in megabytes (MB).

    Args:
        df (pd.DataFrame): The DataFrame to measure.

    Returns:
        float: The memory usage in MB.
    """
    return df.memory_usage(deep=True).sum() / (1024 ** 2)

def timeit(msg: str) -> Callable:
    """
    A decorator that logs the execution time of a function.

    Args:
        msg (str): A message to print describing the timed operation.

    Returns:
        Callable: The decorated function.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            print(f"[Start] {msg}...")
            result = fn(*args, **kwargs)
            duration = time.time() - start_time
            print(f"[Done ] {msg} in {duration:.2f}s")
            return result
        return wrapper
    return decorator
