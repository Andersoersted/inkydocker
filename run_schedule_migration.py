"""
Migration script to run inside Docker to add missing columns.
This is a simplified version of run_migration.py that only adds
the missing columns to the database.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_migration():
    """Run the migration to add missing columns to the schedule_events table."""
    from app import db
    from models import ScheduleEvent, UserConfig
    
    logger.info("Starting database schema migration...")
    
    # Check for missing columns in various tables
    inspector = db.inspect(db.engine)
    
    # 1. Check schedule_events table
    columns = inspector.get_columns('schedule_events')
    schedule_columns = [col['name'] for col in columns]
    
    needs_recurrence_details = 'recurrence_details' not in schedule_columns
    needs_created_at = 'created_at' not in schedule_columns
    
    if needs_recurrence_details or needs_created_at:
        logger.info("Adding missing columns to schedule_events table...")
        
        # Add columns using raw SQL to avoid alembic dependency
        if needs_recurrence_details:
            db.engine.execute('ALTER TABLE schedule_events ADD COLUMN recurrence_details TEXT')
            logger.info("Added recurrence_details column")
            
        if needs_created_at:
            db.engine.execute('ALTER TABLE schedule_events ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            logger.info("Added created_at column")
    else:
        logger.info("All required columns already exist in schedule_events table")
    
    # 2. Check user_config table
    columns = inspector.get_columns('user_config')
    user_config_columns = [col['name'] for col in columns]
    
    needs_zero_shot_enabled = 'zero_shot_enabled' not in user_config_columns
    
    if needs_zero_shot_enabled:
        logger.info("Adding missing columns to user_config table...")
        
        # Add zero_shot_enabled column if missing
        db.engine.execute('ALTER TABLE user_config ADD COLUMN zero_shot_enabled BOOLEAN DEFAULT 1')
        logger.info("Added zero_shot_enabled column")
        
        # Set default value for existing rows
        db.engine.execute("UPDATE user_config SET zero_shot_enabled = 1")
    else:
        logger.info("All required columns already exist in user_config table")
    
    logger.info("Migration completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = run_migration()
        if success:
            print("Migration completed successfully!")
        else:
            print("Migration failed. Check the logs for details.")
            sys.exit(1)
    except Exception as e:
        print(f"Migration failed with error: {str(e)}")
        sys.exit(1)