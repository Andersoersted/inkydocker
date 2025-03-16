#!/usr/bin/env python3
"""
Dedicated scheduler process for InkyDocker.

The scheduler is responsible for:
1. Running scheduled image sends based on the schedule in the database
2. Periodically checking device online status
"""

import sys
import logging
import multiprocessing
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.distributed.reduce_op.*")

# Set multiprocessing start method to 'spawn'
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from config import Config

# Configure logging with reduced verbosity
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler(
    job_defaults={
        'coalesce': True,
        'max_instances': 1,
        'misfire_grace_time': 3600
    },
    timezone='Europe/Copenhagen'
)

def create_app():
    """Create a minimal Flask app for the scheduler context"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    from models import db
    db.init_app(app)
    
    return app

def load_scheduled_events(app):
    """Load all scheduled events from the database and schedule them."""
    with app.app_context():
        from tasks import send_scheduled_image
        from models import ScheduleEvent
        import datetime
        import pytz
        
        try:
            # Clear existing scheduled events
            for job in scheduler.get_jobs():
                if job.id.startswith('event_'):
                    job.remove()
            
            # Load events from database
            events = ScheduleEvent.query.filter_by(sent=False).all()
            logger.info(f"Loading {len(events)} scheduled events from database")
            
            copenhagen_tz = pytz.timezone('Europe/Copenhagen')
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
    """Start the APScheduler with the Flask app context."""
    with app.app_context():
        from tasks import fetch_device_metrics
        
        # Schedule device metrics check
        scheduler.add_job(
            fetch_device_metrics,
            'interval',
            seconds=60,
            id='fetch_device_metrics'
        )
        
        # Schedule event check
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

if __name__ == "__main__":
    app = create_app()
    start_scheduler(app)