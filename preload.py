"""
Preload script for gunicorn to ensure gevent monkey patching happens
before any other imports.
"""
from gevent import monkey
# Patch everything before any other imports
monkey.patch_all()

# Now it's safe to import other modules
import multiprocessing

# Set multiprocessing start method to 'spawn' to fix CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

print("Preload complete: gevent monkey patching applied before other imports")