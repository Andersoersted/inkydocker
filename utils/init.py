"""
Shared initialization code for the application.
This module contains setup code that needs to run across multiple entry points.
"""
import multiprocessing


def setup_multiprocessing():
    """
    Set multiprocessing start method to 'spawn' to fix CUDA issues.
    This needs to be done before any multiprocessing operations.

    Safe to call multiple times - will not raise an error if already set.
    """
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set, ignore
        pass
