"""Add updated_at column to crop_info table

Revision ID: 7bb4c3d5e3a1
Revises: add_ram_model_columns
Create Date: 2025-03-21 22:25:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


# revision identifiers, used by Alembic.
revision = '7bb4c3d5e3a1'
down_revision = 'add_ram_model_columns'  # Adjust this if your latest migration has a different name
branch_labels = None
depends_on = None


def upgrade():
    # Add the updated_at column with a default value of the current timestamp
    op.add_column('crop_info', sa.Column('updated_at', sa.DateTime, default=datetime.utcnow))
    
    # Set all existing rows to have current timestamp
    op.execute("UPDATE crop_info SET updated_at = CURRENT_TIMESTAMP")
    

def downgrade():
    # Simply remove the column when downgrading
    op.drop_column('crop_info', 'updated_at')