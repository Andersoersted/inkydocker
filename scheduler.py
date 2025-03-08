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
import multiprocessing

# Set multiprocessing start method to 'spawn' to fix CUDA issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

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
    timezone='Europe/Copenhagen'  # Use the same timezone as the Docker container
)

def create_app():
    """Create a minimal Flask app for the scheduler context"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    return app

# Function to load and schedule events from the database
def load_scheduled_events(app):
    """
    Load all scheduled events from the database and schedule them.
    This function is called periodically to pick up new events.
    """
    with app.app_context():
        from tasks import send_scheduled_image
        from models import ScheduleEvent
        import datetime
        import pytz
        
        try:
            # Clear any existing scheduled events to avoid duplicates
            for job in scheduler.get_jobs():
                if job.id.startswith('event_'):
                    job.remove()
            
            # Load all scheduled events from the database
            events = ScheduleEvent.query.filter_by(sent=False).all()
            logger.info(f"Loading {len(events)} scheduled events from database")
            
            # Get the Copenhagen timezone
            copenhagen_tz = pytz.timezone('Europe/Copenhagen')
            
            # Get current time in Copenhagen timezone for comparison
            now = datetime.datetime.now(copenhagen_tz)
            
            scheduled_count = 0
            past_count = 0
            
            for event in events:
                try:
                    # Parse the datetime string
                    dt = datetime.datetime.fromisoformat(event.datetime_str)
                    
                    # Ensure the datetime is in Copenhagen timezone
                    if dt.tzinfo is None:
                        dt = copenhagen_tz.localize(dt)
                    else:
                        dt = dt.astimezone(copenhagen_tz)
                    
                    if dt > now:
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt,
                            args=[event.id],
                            id=f'event_{event.id}'
                        )
                        scheduled_count += 1
                    else:
                        past_count += 1
                except Exception as e:
                    logger.error(f"Error scheduling event {event.id}: {e}")
            
            logger.info(f"Scheduled {scheduled_count} events, {past_count} events were in the past")
            
        except Exception as e:
            logger.error(f"Error loading scheduled events: {e}")

def start_scheduler(app):
    """
    Start the APScheduler with the Flask app context.
    This function should only be called once in this dedicated process.
    """
    with app.app_context():
        # Import the necessary functions
        from tasks import fetch_device_metrics
        
        # Schedule the fetch_device_metrics task to run every 60 seconds
        scheduler.add_job(
            fetch_device_metrics,
            'interval',
            seconds=60,
            id='fetch_device_metrics'
        )
        
        # Schedule a job to check for new events every 60 seconds
        scheduler.add_job(
            lambda: load_scheduled_events(app),
            'interval',
            seconds=60,
            id='check_for_new_events'
        )
        
        # Initial load of scheduled events
        load_scheduled_events(app)
        
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