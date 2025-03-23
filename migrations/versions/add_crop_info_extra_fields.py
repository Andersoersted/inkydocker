"""add crop info extra fields

Revision ID: add_crop_info_extra_fields
Revises: add_zero_shot_model_columns
Create Date: 2025-03-23 08:52:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_crop_info_extra_fields'
down_revision = 'add_zero_shot_model_columns'  # I'm using a more stable revision here that should exist
branch_labels = None
depends_on = None


def upgrade():
    # First check if these columns already exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('screenshot_crop_info')]
    
    # Only add columns if they don't already exist
    if 'device_address' not in columns:
        op.add_column('screenshot_crop_info', sa.Column('device_address', sa.String(length=256), nullable=True))
    
    if 'natural_width' not in columns:
        op.add_column('screenshot_crop_info', sa.Column('natural_width', sa.Integer(), nullable=True))
    
    if 'natural_height' not in columns:
        op.add_column('screenshot_crop_info', sa.Column('natural_height', sa.Integer(), nullable=True))
    
    if 'created_at' not in columns:
        op.add_column('screenshot_crop_info', sa.Column('created_at', sa.DateTime(), 
                      server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True))
    
    if 'updated_at' not in columns:
        op.add_column('screenshot_crop_info', sa.Column('updated_at', sa.DateTime(), 
                      server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True))


def downgrade():
    # Remove the columns if they exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('screenshot_crop_info')]
    
    if 'updated_at' in columns:
        op.drop_column('screenshot_crop_info', 'updated_at')
    
    if 'created_at' in columns:
        op.drop_column('screenshot_crop_info', 'created_at')
    
    if 'natural_height' in columns:
        op.drop_column('screenshot_crop_info', 'natural_height')
    
    if 'natural_width' in columns:
        op.drop_column('screenshot_crop_info', 'natural_width')
    
    if 'device_address' in columns:
        op.drop_column('screenshot_crop_info', 'device_address')