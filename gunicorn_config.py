import multiprocessing

# Set multiprocessing start method to 'spawn' to fix CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

# Gunicorn config
bind = "0.0.0.0:5001"
workers = 2
timeout = 300
worker_class = "gthread"
threads = 4
max_requests = 100
preload_app = True

# Log settings
accesslog = "-"
errorlog = "-"
loglevel = "info"