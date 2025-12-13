"""
Preload script for gunicorn
"""
from utils.init import setup_multiprocessing

# Set multiprocessing start method to 'spawn' to fix CUDA issues
setup_multiprocessing()

print("Preload complete")