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
worker_class = "gevent"
worker_connections = 1000
max_requests = 100
max_requests_jitter = 20
preload_app = True  # Load application code before forking workers

# Log settings
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Initialization hook
def on_starting(server):
    print("Gunicorn starting with multiprocessing start method:", multiprocessing.get_start_method())