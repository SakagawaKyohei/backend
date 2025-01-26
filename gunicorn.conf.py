# Gunicorn configuration file
import multiprocessing

# Bind to 0.0.0.0:$PORT
bind = "0.0.0.0:$PORT"

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Thread per worker
threads = 2

# Timeout
timeout = 120

# Access log - writes to stdout by default
accesslog = "-"

# Error log - writes to stderr by default
errorlog = "-"

# Log level
loglevel = "info" 