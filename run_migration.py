#!/usr/bin/env python
"""
Migration script for the schedule recurrence update.
This will add the necessary columns to the database.
"""
import os
import sys
import json
from datetime import datetime
import pytz

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import db
from models import ScheduleEvent

def migrate_events():
    """Migrate existing events to use the new recurrence details format."""
    print("Starting migration of scheduling events...")
    
    # Check if the columns exist
    inspector = db.inspect(db.engine)
    columns = inspector.get_columns('schedule_events')
    column_names = [col['name'] for col in columns]
    
    needs_recurrence_details = 'recurrence_details' not in column_names
    needs_created_at = 'created_at' not in column_names
    
    if needs_recurrence_details or needs_created_at:
        print("Adding missing columns to the database...")
        
        # Add columns using raw SQL to avoid alembic dependency
        if needs_recurrence_details:
            db.engine.execute('ALTER TABLE schedule_events ADD COLUMN recurrence_details TEXT')
            print("Added recurrence_details column")
            
        if needs_created_at:
            db.engine.execute('ALTER TABLE schedule_events ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            print("Added created_at column")
        
    # Update existing events
    events = ScheduleEvent.query.all()
    now = datetime.now(pytz.timezone('Europe/Copenhagen'))
    count = 0
    
    print(f"Found {len(events)} events to process")
    
    for ev in events:
        if ev.recurrence.lower() != "none" and not getattr(ev, 'recurrence_details', None):
            # Create new recurrence details with start_from set to now
            # This ensures recurring events only appear from today onwards
            details = {
                "start_from": now.isoformat(),  # Force to today's date to prevent past events
            }
            
            # Add pattern-specific details
            if ev.recurrence.lower() == "daily":
                details["interval"] = 1
            elif ev.recurrence.lower() == "weekly":
                details["interval"] = 1
                # Default to the day of week from the start date
                try:
                    start_dt = datetime.fromisoformat(ev.datetime_str)
                    details["weekdays"] = [start_dt.weekday()]
                except:
                    details["weekdays"] = [0]  # Monday as default
            elif ev.recurrence.lower() == "monthly":
                details["interval"] = 1
                details["type"] = "day"
            
            # Save details as JSON
            ev.recurrence_details = json.dumps(details)
            count += 1
    
    # Commit all changes
    db.session.commit()
    
    print(f"Successfully migrated {count} events")
    return count

if __name__ == "__main__":
    count = migrate_events()
    print(f"Migration completed. {count} events updated.")