"""add device_address column to crop tables

Revision ID: add_device_specific_crop_info
Revises: fix_crop_updated_at_column
Create Date: 2025-03-23 10:37:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_device_specific_crop_info'
down_revision = 'fix_crop_updated_at_column'
branch_labels = None
depends_on = None


def upgrade():
    # Add device_address column to crop_info table
    op.add_column('crop_info', sa.Column('device_address', sa.String(length=255), nullable=True))
    
    # Add device_address column to screenshot_crop_info table
    op.add_column('screenshot_crop_info', sa.Column('device_address', sa.String(length=255), nullable=True))
    
    # Set default value for existing records
    op.execute("UPDATE crop_info SET device_address = 'default_device' WHERE device_address IS NULL")
    op.execute("UPDATE screenshot_crop_info SET device_address = 'default_device' WHERE device_address IS NULL")
    
    # Add composite index on filename and device_address columns for faster lookups
    op.create_index('ix_crop_info_filename_device', 'crop_info', ['filename', 'device_address'], unique=False)
    op.create_index('ix_screenshot_crop_info_filename_device', 'screenshot_crop_info', ['filename', 'device_address'], unique=False)
    
    # Now update the primary key constraint to be (filename, device_address) instead of just filename
    # First drop the primary key constraint if it exists
    try:
        op.drop_constraint('crop_info_pkey', 'crop_info', type_='primary')
    except Exception:
        # It might not exist or have a different name, continue anyway
        pass
    
    try:
        op.drop_constraint('screenshot_crop_info_pkey', 'screenshot_crop_info', type_='primary')
    except Exception:
        # It might not exist or have a different name, continue anyway
        pass
    
    # Create new primary key constraints
    op.create_primary_key('crop_info_pkey', 'crop_info', ['filename', 'device_address'])
    op.create_primary_key('screenshot_crop_info_pkey', 'screenshot_crop_info', ['filename', 'device_address'])


def downgrade():
    # Drop the composite primary keys
    op.drop_constraint('crop_info_pkey', 'crop_info', type_='primary')
    op.drop_constraint('screenshot_crop_info_pkey', 'screenshot_crop_info', type_='primary')
    
    # Recreate the original primary keys on just filename
    op.create_primary_key('crop_info_pkey', 'crop_info', ['filename'])
    op.create_primary_key('screenshot_crop_info_pkey', 'screenshot_crop_info', ['filename'])
    
    # Drop the indexes
    op.drop_index('ix_crop_info_filename_device', table_name='crop_info')
    op.drop_index('ix_screenshot_crop_info_filename_device', table_name='screenshot_crop_info')
    
    # Drop the device_address columns
    op.drop_column('crop_info', 'device_address')
    op.drop_column('screenshot_crop_info', 'device_address')