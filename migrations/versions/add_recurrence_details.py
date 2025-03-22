"""Add recurrence_details and created_at columns

Revision ID: e1c9f69dc8c2
Revises: update_min_tags_default
Create Date: 2025-03-22 14:49:00

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


# revision identifiers, used by Alembic.
revision = 'e1c9f69dc8c2'
down_revision = 'update_min_tags_default'  # Replace with an actual revision ID if needed
branch_labels = None
depends_on = None


def upgrade():
    # Add recurrence_details column to store JSON data
    op.add_column('schedule_events', sa.Column('recurrence_details', sa.Text(), nullable=True))
    
    # Add created_at column with default value of current time
    op.add_column('schedule_events', sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False))
    
    # Optionally, you could add a migration helper to populate recurrence_details for existing records
    # but we've added a route to do this in schedule_routes.py for more control


def downgrade():
    # Remove the columns if needed
    op.drop_column('schedule_events', 'created_at')
    op.drop_column('schedule_events', 'recurrence_details')