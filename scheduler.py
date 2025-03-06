#!/usr/bin/env python3
"""
Dedicated scheduler process for InkyDocker.

This file runs the APScheduler in a dedicated process to avoid starting
multiple schedulers in Celery worker processes. This solves the issue where
each Celery worker was starting its own scheduler, leading to duplicate jobs
and potential race conditions.

The scheduler is responsible for:
1. Running scheduled image sends based on the schedule in the database
2. Periodically checking device online status
"""

import os
import sys
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize scheduler at the module level so it can be imported by other modules
# Configure with optimized settings for better performance
scheduler = BackgroundScheduler(
    job_defaults={
        'coalesce': True,  # Combine multiple pending executions of the same job into one
        'max_instances': 1,  # Only allow one instance of each job to run at a time
        'misfire_grace_time': 3600  # Allow misfires up to 1 hour
    },
    executor='threadpool',  # Use threadpool executor for better performance
    timezone='UTC'  # Use UTC for consistent timezone handling
)

def create_app():
    """Create a minimal Flask app for the scheduler context"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    return app

def start_scheduler(app):
    """
    Start the APScheduler with the Flask app context.
    This function should only be called once in this dedicated process.
    """
    with app.app_context():
        # Import the necessary functions
        from tasks import fetch_device_metrics, send_scheduled_image
        from models import ScheduleEvent
        
        # Schedule the fetch_device_metrics task to run every 60 seconds (reduced from 5 seconds to prevent log flooding)
        scheduler.add_job(
            fetch_device_metrics,
            'interval',
            seconds=60,
            id='fetch_device_metrics'
        )
        
        # Load all scheduled events from the database and schedule them
        try:
            events = ScheduleEvent.query.filter_by(sent=False).all()
            for event in events:
                try:
                    import datetime
                    from datetime import timezone
                    dt = datetime.datetime.fromisoformat(event.datetime_str)
                    
                    # Ensure both datetimes are timezone-aware for comparison
                    # If dt has a timezone, use it as is
                    # If dt doesn't have a timezone, assume it's in UTC
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    
                    # Get current time in UTC for comparison
                    now = datetime.datetime.now(timezone.utc)
                    
                    if dt > now:
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt,
                            args=[event.id],
                            id=f'event_{event.id}'
                        )
                        logger.info(f"Scheduled event {event.id} for {dt}")
                except Exception as e:
                    logger.error(f"Error scheduling event {event.id}: {e}")
        except Exception as e:
            logger.error(f"Error loading scheduled events: {e}")
            logger.info("Continuing without loading scheduled events")
        
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduler started successfully")
        
        # Keep the process running
        try:
            import time
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler shutting down...")
            scheduler.shutdown()
            logger.info("Scheduler shut down successfully")

if __name__ == "__main__":
    app = create_app()
    start_scheduler(app)