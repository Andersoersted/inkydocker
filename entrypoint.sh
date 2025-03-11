#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

# Check available memory
echo "Checking system memory..."
MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_TOTAL_GB=$(echo "scale=2; $MEM_TOTAL/1024/1024" | bc)
echo "Total memory: ${MEM_TOTAL_GB}GB"

# Set environment variable for memory-aware operations
export SYSTEM_MEMORY_GB=$MEM_TOTAL_GB

# Suppress PyTorch warnings
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

# Configure memory limits based on available memory
if [ $(echo "$MEM_TOTAL_GB < 4" | bc) -eq 1 ]; then
  echo "WARNING: Low memory system detected (${MEM_TOTAL_GB}GB). Some large models may not work properly."
  # Set environment variable to force CPU usage for large models
  export FORCE_CPU_FOR_LARGE_MODELS=1
  # Reduce Celery worker memory limit
  export CELERY_MAX_MEMORY=250000
elif [ $(echo "$MEM_TOTAL_GB < 8" | bc) -eq 1 ]; then
  echo "Medium memory system detected (${MEM_TOTAL_GB}GB). Some very large models may be limited to CPU."
  export FORCE_CPU_FOR_LARGE_MODELS=0
  export CELERY_MAX_MEMORY=500000
else
  echo "High memory system detected (${MEM_TOTAL_GB}GB). All models should work properly."
  export FORCE_CPU_FOR_LARGE_MODELS=0
  export CELERY_MAX_MEMORY=1000000
fi

# Start Redis locally
echo "Starting Redis server..."
redis-server --daemonize yes
sleep 2
echo "Checking Redis connection..."
redis-cli ping
if [ $? -ne 0 ]; then
  echo "Redis is not responding. Waiting a bit longer..."
  sleep 5
  redis-cli ping
  if [ $? -ne 0 ]; then
    echo "Redis still not responding. Please check Redis configuration."
  else
    echo "Redis is now running."
  fi
else
  echo "Redis is running."
fi

# Check if the database file exists
if [ ! -f /app/data/mydb.sqlite ]; then
  echo "Creating database tables for the first time..."
  python -c "from app import app; from models import db; app.app_context().push(); db.create_all()"
  echo "Database tables created successfully."
else
  echo "Database already exists, skipping table creation."
fi

echo "Running database migrations..."
cd /app && python -m flask db upgrade
echo "Database migrations applied successfully."

# Clear any stale lock files
echo "Clearing any stale lock files..."
rm -f /tmp/*.lock

# Set up core dump pattern for debugging segmentation faults
echo "Setting up core dump pattern for debugging..."
echo "/tmp/core.%e.%p.%t" > /proc/sys/kernel/core_pattern
ulimit -c unlimited

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
