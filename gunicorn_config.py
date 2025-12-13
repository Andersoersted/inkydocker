from utils.init import setup_multiprocessing

# Set multiprocessing start method to 'spawn' to fix CUDA issues
setup_multiprocessing()

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