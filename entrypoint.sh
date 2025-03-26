#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

# Suppress PyTorch warnings
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning"

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

# Check if the database exists and run migrations
echo "Running database migrations..."
python -c "from app import app; from flask_migrate import upgrade as flask_migrate_upgrade; app.app_context().push(); flask_migrate_upgrade()"
echo "Database migrations completed successfully."

# Clear any stale lock files
echo "Clearing any stale lock files..."
rm -f /tmp/*.lock

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
