"""Fix updated_at column in crop_info table

Revision ID: fix_crop_updated_at
Revises: 7bb4c3d5e3a1
Create Date: 2025-03-22 08:36:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = 'fix_crop_updated_at'
down_revision = '7bb4c3d5e3a1'  # This should be the previous migration
branch_labels = None
depends_on = None


def upgrade():
    # Get inspector to check if column exists
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [column['name'] for column in inspector.get_columns('crop_info')]
    
    # Only add the column if it doesn't exist
    if 'updated_at' not in columns:
        op.add_column('crop_info', sa.Column('updated_at', sa.DateTime, nullable=True))
        op.execute("UPDATE crop_info SET updated_at = CURRENT_TIMESTAMP")


def downgrade():
    # Don't drop the column on downgrade, to prevent data loss
    pass