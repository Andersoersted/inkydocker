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
        'misfire_grace_time': 3600  # Allow misfires up to 1 hour
    }
    # Timezone removed to treat schedule times as naive local times
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
        from models import ScheduleEvent, db
        import datetime
        import os
        
        try:
            # Clear existing scheduled events
            for job in scheduler.get_jobs():
                if job.id.startswith('event_'):
                    job.remove()
            
            # Load events from database
            events = ScheduleEvent.query.filter_by(sent=False).all()
            logger.info(f"Loading {len(events)} scheduled events from database")
            
            # Get current naive local time
            now = datetime.datetime.now()
            
            # Define a cutoff time (10 minutes in the past) using naive time
            cutoff_time = now - datetime.timedelta(minutes=10)
            
            scheduled_count = 0
            past_count = 0
            skipped_count = 0
            
            for event in events:
                try:
                    # Parse the datetime string as naive local time
                    # Assuming datetime_str is stored in ISO format without timezone info
                    # or that any timezone info should be ignored.
                    dt_str = event.datetime_str
                    if '+' in dt_str: # Remove timezone offset if present
                        dt_str = dt_str.split('+')[0]
                    if 'Z' in dt_str: # Remove UTC indicator if present
                        dt_str = dt_str.replace('Z', '')
                        
                    try:
                        dt = datetime.datetime.fromisoformat(dt_str)
                    except ValueError:
                        # Fallback for potentially different formats, though ISO is expected
                        from dateutil import parser
                        logger.warning(f"Could not parse '{event.datetime_str}' with fromisoformat, trying dateutil.parser")
                        dt = parser.parse(dt_str) # dateutil parser returns naive datetime if no tz info

                    # Check if the event is in the future (using naive comparison)
                    if dt > now:
                        # Schedule the event using the naive datetime
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt, # APScheduler interprets naive datetime as local time
                            args=[event.id],
                            id=f'event_{event.id}',
                            misfire_grace_time=3600
                        )
                        scheduled_count += 1
                        logger.debug(f"Scheduled event {event.id} for future: {dt}")
                    else:
                        # For past events, check if they're older than the cutoff time
                        if dt < cutoff_time:
                            # For events older than cutoff, mark as sent without executing
                            logger.info(f"Event {event.id} is older than 10-minute cutoff ({dt}), marking as sent without executing")
                            event.sent = True
                            db.session.commit()
                            skipped_count += 1
                        else:
                            # Only process recent past events (within 10-minute cutoff)
                            logger.info(f"Event {event.id} is recent past (<10 minutes old, {dt}), executing immediately")
                            try:
                                # Execute the event directly
                                send_scheduled_image(event.id)
                                
                                # Verify if the event was actually processed and marked as sent
                                refreshed_event = ScheduleEvent.query.get(event.id)
                                if refreshed_event and not refreshed_event.sent:
                                    logger.warning(f"Event {event.id} was executed but not marked as sent, marking now")
                                    refreshed_event.sent = True
                                    db.session.commit()
                                    
                            except Exception as exec_error:
                                logger.error(f"Error executing past event {event.id}: {exec_error}")
                                
                            past_count += 1
                except Exception as e:
                    logger.error(f"Error processing event {event.id} ('{event.datetime_str}'): {e}")
            
            logger.info(f"Finished loading events: Scheduled={scheduled_count}, Recent Past Executed={past_count}, Old Skipped={skipped_count}")
            
        except Exception as e:
            logger.error(f"Critical error loading scheduled events: {e}")

def start_scheduler(app):
    """Start the APScheduler with the Flask app context."""
    with app.app_context():
        from tasks import fetch_device_metrics, cleanup_expired_notifications
        
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
        
        # Schedule cleanup of expired notifications
        scheduler.add_job(
            cleanup_expired_notifications,
            'interval',
            hours=12,  # Run twice a day
            id='cleanup_expired_notifications'
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