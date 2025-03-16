"""
Preload script for gunicorn
"""
import multiprocessing

# Set multiprocessing start method to 'spawn' to fix CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

print("Preload complete")