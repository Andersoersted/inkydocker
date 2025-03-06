#!/bin/sh
# entrypoint.sh - Auto-create the database tables then launch the app via Supervisor.

echo "Ensuring /app/data folder exists..."
mkdir -p /app/data

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

echo "Starting Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
