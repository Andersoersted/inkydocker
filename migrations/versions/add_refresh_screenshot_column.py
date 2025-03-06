"""Add refresh_screenshot column to schedule_events table

Revision ID: add_refresh_screenshot_column
Revises: 
Create Date: 2025-03-06 13:50:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_refresh_screenshot_column'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Check if the column already exists
    from sqlalchemy import inspect
    from sqlalchemy.engine.reflection import Inspector
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [column['name'] for column in inspector.get_columns('schedule_events')]
    
    # Only add the column if it doesn't already exist
    if 'refresh_screenshot' not in columns:
        # For SQLite, we need to use a different approach since it doesn't support ALTER COLUMN
        # Add refresh_screenshot column with default value False and make it non-nullable from the start
        op.add_column('schedule_events', sa.Column('refresh_screenshot', sa.Boolean(), nullable=False, server_default='0'))
        print("Added refresh_screenshot column to schedule_events table")
    else:
        print("refresh_screenshot column already exists in schedule_events table")


def downgrade():
    # Check if the column exists before trying to drop it
    from sqlalchemy import inspect
    from sqlalchemy.engine.reflection import Inspector
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [column['name'] for column in inspector.get_columns('schedule_events')]
    
    # Only drop the column if it exists
    if 'refresh_screenshot' in columns:
        # Remove the refresh_screenshot column
        op.drop_column('schedule_events', 'refresh_screenshot')
        print("Removed refresh_screenshot column from schedule_events table")
    else:
        print("refresh_screenshot column does not exist in schedule_events table")