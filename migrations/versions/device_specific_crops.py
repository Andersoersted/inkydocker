"""device_specific_crops

Revision ID: device_specific_crops
Revises: None
Create Date: 2025-03-23 12:17:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = 'device_specific_crops'
down_revision = None  # This migration doesn't depend on any previous migration
branch_labels = None
depends_on = None


def upgrade():
    """Create tables with device-specific crop functionality from scratch."""
    from sqlalchemy.engine.reflection import Inspector
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    
    # Check if the tables already exist
    tables = inspector.get_table_names()
    
    # Create crop_info table if it doesn't exist
    if 'crop_info' not in tables:
        op.create_table(
            'crop_info',
            sa.Column('filename', sa.String(256), nullable=False),
            sa.Column('device_address', sa.String(256), nullable=False, default='default_device'),
            sa.Column('x', sa.Float, default=0),
            sa.Column('y', sa.Float, default=0),
            sa.Column('width', sa.Float, nullable=True),
            sa.Column('height', sa.Float, nullable=True),
            sa.Column('resolution', sa.String(32), nullable=True),
            sa.Column('updated_at', sa.DateTime, nullable=True),
            sa.PrimaryKeyConstraint('filename', 'device_address', name='crop_info_pkey')
        )
        
        # Create an index for faster lookups
        op.create_index('ix_crop_info_filename_device', 'crop_info', ['filename', 'device_address'], unique=False)
    
    # Create screenshot_crop_info table if it doesn't exist
    if 'screenshot_crop_info' not in tables:
        op.create_table(
            'screenshot_crop_info',
            sa.Column('filename', sa.String(256), nullable=False),
            sa.Column('device_address', sa.String(256), nullable=False, default='default_device'),
            sa.Column('x', sa.Float, default=0),
            sa.Column('y', sa.Float, default=0),
            sa.Column('width', sa.Float, nullable=True),
            sa.Column('height', sa.Float, nullable=True),
            sa.Column('resolution', sa.String(32), nullable=True),
            sa.Column('natural_width', sa.Integer, nullable=True),
            sa.Column('natural_height', sa.Integer, nullable=True),
            sa.Column('created_at', sa.DateTime, server_default=sa.func.current_timestamp()),
            sa.Column('updated_at', sa.DateTime, server_default=sa.func.current_timestamp(), 
                      onupdate=sa.func.current_timestamp()),
            sa.PrimaryKeyConstraint('filename', 'device_address', name='screenshot_crop_info_pkey')
        )
        
        # Create an index for faster lookups
        op.create_index('ix_screenshot_crop_info_filename_device', 'screenshot_crop_info', 
                         ['filename', 'device_address'], unique=False)
    else:
        # If the table already exists, check if we need to add the device_address column
        columns = [c['name'] for c in inspector.get_columns('screenshot_crop_info')]
        if 'device_address' not in columns:
            op.add_column('screenshot_crop_info', 
                          sa.Column('device_address', sa.String(256), nullable=True))
            op.execute("UPDATE screenshot_crop_info SET device_address = 'default_device' WHERE device_address IS NULL")
            
            # Try to update the primary key if needed
            try:
                pk_constraint = inspector.get_pk_constraint('screenshot_crop_info')
                constraint_name = pk_constraint.get('name', 'screenshot_crop_info_pkey')
                op.drop_constraint(constraint_name, 'screenshot_crop_info', type_='primary')
                op.create_primary_key('screenshot_crop_info_pkey', 'screenshot_crop_info', 
                                     ['filename', 'device_address'])
            except Exception as e:
                print(f"Error updating screenshot_crop_info primary key: {e}")
                
            # Add the index if it doesn't exist
            indexes = inspector.get_indexes('screenshot_crop_info')
            if not any(idx['name'] == 'ix_screenshot_crop_info_filename_device' for idx in indexes):
                op.create_index('ix_screenshot_crop_info_filename_device', 'screenshot_crop_info', 
                               ['filename', 'device_address'], unique=False)

    if 'crop_info' in tables:
        # If the crop_info table already exists, check if we need to add the device_address column
        columns = [c['name'] for c in inspector.get_columns('crop_info')]
        if 'device_address' not in columns:
            op.add_column('crop_info', 
                         sa.Column('device_address', sa.String(256), nullable=True))
            op.execute("UPDATE crop_info SET device_address = 'default_device' WHERE device_address IS NULL")
            
            # Try to update the primary key if needed
            try:
                pk_constraint = inspector.get_pk_constraint('crop_info')
                constraint_name = pk_constraint.get('name', 'crop_info_pkey')
                op.drop_constraint(constraint_name, 'crop_info', type_='primary')
                op.create_primary_key('crop_info_pkey', 'crop_info', 
                                     ['filename', 'device_address'])
            except Exception as e:
                print(f"Error updating crop_info primary key: {e}")
                
            # Add the index if it doesn't exist
            indexes = inspector.get_indexes('crop_info')
            if not any(idx['name'] == 'ix_crop_info_filename_device' for idx in indexes):
                op.create_index('ix_crop_info_filename_device', 'crop_info', 
                               ['filename', 'device_address'], unique=False)


def downgrade():
    """Don't implement downgrade to avoid data loss."""
    pass