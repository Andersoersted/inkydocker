import os
import sys
import datetime
import pytz
from datetime import timedelta

# Add the current directory to the path for imports
sys.path.append(os.getcwd())

# Create a minimal Flask app for the context
from flask import Flask
app = Flask(__name__)
from config import Config
app.config.from_object(Config)

# Initialize database
from models import db, ScheduleEvent
db.init_app(app)

# Function to check scheduled events
def check_events():
    with app.app_context():
        # Get current time in Copenhagen timezone
        copenhagen_tz = pytz.timezone('Europe/Copenhagen')
        now = datetime.datetime.now(copenhagen_tz)
        print(f"Current time (Copenhagen): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Get all events
        events = ScheduleEvent.query.all()
        print(f"Found {len(events)} total events")
        
        # Unsent events
        unsent_events = ScheduleEvent.query.filter_by(sent=False).all()
        print(f"Found {len(unsent_events)} unsent events:")
        
        for event in unsent_events:
            try:
                # Parse the datetime string
                dt = datetime.datetime.fromisoformat(event.datetime_str)
                
                # Ensure the datetime is in Copenhagen timezone
                if dt.tzinfo is None:
                    dt = copenhagen_tz.localize(dt)
                else:
                    dt = dt.astimezone(copenhagen_tz)
                
                time_diff = dt - now
                print(f"Event ID: {event.id}, Filename: {event.filename}")
                print(f"  Scheduled for: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"  Time difference: {time_diff}")
                
                # Check if this event should have been executed
                if dt < now:
                    print(f"  WARNING: This event should have executed already!")
                    print(f"  Original datetime string: {event.datetime_str}")
                
                # Check if this event is coming up soon
                elif dt < now + timedelta(minutes=15):
                    print(f"  INFO: This event will execute soon")
            except Exception as e:
                print(f"  ERROR parsing event {event.id}: {e}")
                print(f"  Original datetime string: {event.datetime_str}")
        
        # Check scheduler configuration
        from scheduler import scheduler
        print("\nScheduler jobs:")
        for job in scheduler.get_jobs():
            print(f"Job ID: {job.id}, Next run: {job.next_run_time}")

if __name__ == "__main__":
    check_events()