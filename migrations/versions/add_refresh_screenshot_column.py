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
    # Add refresh_screenshot column with default value False
    op.add_column('schedule_events', sa.Column('refresh_screenshot', sa.Boolean(), nullable=True, server_default='0'))
    
    # Update existing rows to have refresh_screenshot=False
    op.execute("UPDATE schedule_events SET refresh_screenshot = 0 WHERE refresh_screenshot IS NULL")
    
    # Make the column non-nullable after setting default values
    op.alter_column('schedule_events', 'refresh_screenshot', nullable=False, server_default='0')


def downgrade():
    # Remove the refresh_screenshot column
    op.drop_column('schedule_events', 'refresh_screenshot')