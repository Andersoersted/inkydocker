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
        from models import ScheduleEvent, db
        import datetime
        import pytz
        import os
        
        try:
            # Clear existing scheduled events
            for job in scheduler.get_jobs():
                if job.id.startswith('event_'):
                    job.remove()
            
            # Load events from database
            events = ScheduleEvent.query.filter_by(sent=False).all()
            logger.info(f"Loading {len(events)} scheduled events from database")
            
            # ENHANCEMENT: Also get recent events that might have been marked as sent incorrectly
            recent_sent_events = ScheduleEvent.query.filter(
                ScheduleEvent.sent==True,
                ScheduleEvent.datetime_str >= (datetime.datetime.now() - datetime.timedelta(hours=6)).isoformat()
            ).all()
            logger.info(f"Found {len(recent_sent_events)} recent events marked as sent")
            
            # Write to a separate log file for easier debugging
            with open('/tmp/scheduler_log.txt', 'a') as f:
                f.write(f"\n{'-'*80}\n{datetime.datetime.now()}: Loading scheduled events\n")
                f.write(f"Found {len(events)} unsent events\n")
                f.write(f"Found {len(recent_sent_events)} recent events marked as sent\n")
                
                # Additional logs for timezone debugging
                f.write(f"Current time: {datetime.datetime.now()}\n")
                f.write(f"Current time (Copenhagen): {datetime.datetime.now(copenhagen_tz)}\n")
            
            copenhagen_tz = pytz.timezone('Europe/Copenhagen')
            now = datetime.datetime.now(copenhagen_tz)
            
            # Define a cutoff time (10 minutes in the past)
            cutoff_time = now - datetime.timedelta(minutes=10)
            
            scheduled_count = 0
            past_count = 0
            skipped_count = 0
            
            for event in events:
                try:
                    # Parse the datetime string with enhanced error handling
                    try:
                        dt = datetime.datetime.fromisoformat(event.datetime_str)
                    except ValueError as e:
                        # Try additional parsing methods if the standard one fails
                        with open('/tmp/scheduler_log.txt', 'a') as f:
                            f.write(f"Error parsing datetime '{event.datetime_str}': {e}\n")
                            f.write(f"Attempting alternative parsing methods...\n")
                        
                        # Try removing timezone info and treating as Copenhagen time
                        if '+' in event.datetime_str:
                            dt_str = event.datetime_str.split('+')[0]
                            dt = datetime.datetime.fromisoformat(dt_str)
                            dt = copenhagen_tz.localize(dt)
                        else:
                            # Last resort - try parsing with dateutil
                            from dateutil import parser
                            dt = parser.parse(event.datetime_str)
                            if dt.tzinfo is None:
                                dt = copenhagen_tz.localize(dt)
                    
                    # Ensure the datetime is in Copenhagen timezone
                    if dt.tzinfo is None:
                        dt = copenhagen_tz.localize(dt)
                    else:
                        dt = dt.astimezone(copenhagen_tz)
                    
                    # ENHANCEMENT: More detailed logging of event time information
                    with open('/tmp/scheduler_log.txt', 'a') as f:
                        f.write(f"Event ID: {event.id}, Filename: {event.filename}, Device: {event.device}\n")
                        f.write(f"  Original datetime string: {event.datetime_str}\n")
                        f.write(f"  Parsed datetime: {dt}\n")
                        f.write(f"  Current time: {now}\n")
                        f.write(f"  Time difference: {dt - now}\n")
                    
                    # Check if the event is in the future
                    time_diff = dt - now
                    time_diff_seconds = time_diff.total_seconds()
                    
                    # ENHANCEMENT: More detailed event time comparison
                    if time_diff_seconds > 0:
                        with open('/tmp/scheduler_log.txt', 'a') as f:
                            f.write(f"  Event is {time_diff_seconds:.1f} seconds in the future\n")
                        # Schedule the event
                        scheduler.add_job(
                            send_scheduled_image,
                            'date',
                            run_date=dt,
                            args=[event.id],
                            id=f'event_{event.id}',
                            misfire_grace_time=3600  # Allow misfires up to 1 hour
                        )
                        scheduled_count += 1
                        with open('/tmp/scheduler_log.txt', 'a') as f:
                            f.write(f"  -> Scheduled for future: {dt}\n")
                    else:
                        # For past events, check if they're older than the cutoff time
                        if dt < cutoff_time:
                            # For events older than cutoff, mark as sent without executing
                            logger.info(f"Event {event.id} is older than 10-minute cutoff ({dt}), marking as sent without executing")
                            with open('/tmp/scheduler_log.txt', 'a') as f:
                                f.write(f"  -> Older than 10-minute cutoff ({cutoff_time}), skipping and marking as sent\n")
                            
                            # Mark as sent
                            event.sent = True
                            db.session.commit()
                            skipped_count += 1
                        else:
                            # Only process recent past events (within 10-minute cutoff)
                            logger.info(f"Event {event.id} is recent past (<10 minutes old, {dt}), executing")
                            with open('/tmp/scheduled_image_log.txt', 'a') as f:
                                f.write(f"\n{'-'*80}\n{datetime.datetime.now()}: Executing recent past event\n")
                                f.write(f"Event ID: {event.id}, Time: {dt}\n")
                                f.write(f"Time difference from now: {now - dt}\n")
                            
                            with open('/tmp/scheduler_log.txt', 'a') as f:
                                f.write(f"  -> Recent past event (<10 minutes old), executing immediately\n")
                            
                            try:
                                # Execute the event directly with better error handling
                                send_scheduled_image(event.id)
                                
                                # Verify if the event was actually processed
                                refreshed_event = ScheduleEvent.query.get(event.id)
                                if refreshed_event and not refreshed_event.sent:
                                    logger.warning(f"Event {event.id} was executed but not marked as sent, marking now")
                                    refreshed_event.sent = True
                                    db.session.commit()
                                    
                                with open('/tmp/scheduler_log.txt', 'a') as f:
                                    f.write(f"  -> Event executed successfully\n")
                                    
                            except Exception as exec_error:
                                logger.error(f"Error executing past event {event.id}: {exec_error}")
                                with open('/tmp/scheduler_log.txt', 'a') as f:
                                    f.write(f"  -> ERROR executing event: {str(exec_error)}\n")
                                    
                            past_count += 1
                except Exception as e:
                    logger.error(f"Error scheduling event {event.id}: {e}")
                    with open('/tmp/scheduler_log.txt', 'a') as f:
                        f.write(f"  -> ERROR: {str(e)}\n")
            
            logger.info(f"Scheduled {scheduled_count} events, {past_count} events were in the recent past (<10 minutes old), {skipped_count} older events were skipped")
            with open('/tmp/scheduler_log.txt', 'a') as f:
                f.write(f"Scheduled {scheduled_count} events, {past_count} events were in the recent past (<10 minutes old), {skipped_count} older events were skipped\n")
            
        except Exception as e:
            logger.error(f"Error loading scheduled events: {e}")
            with open('/tmp/scheduler_log.txt', 'a') as f:
                f.write(f"ERROR loading scheduled events: {str(e)}\n")

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