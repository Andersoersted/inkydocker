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

# Since the app is in development and we're frequently making schema changes,
# we'll just create a fresh database with the latest schema instead of migrations
echo "Creating database tables from the latest models..."
python -c "from app import app; from models import db; app.app_context().push(); db.drop_all(); db.create_all()"
echo "Database tables created successfully with the latest schema."

# Clear any stale lock files
echo "Clearing any stale lock files..."
rm -f /tmp/*.lock

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
