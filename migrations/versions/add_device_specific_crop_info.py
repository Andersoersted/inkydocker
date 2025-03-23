"""add device_address column to crop tables

Revision ID: add_device_specific_crop_info
Revises: fix_crop_updated_at
Create Date: 2025-03-23 10:37:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_device_specific_crop_info'
down_revision = 'fix_crop_updated_at'
branch_labels = None
depends_on = None


def upgrade():
    from sqlalchemy.engine.reflection import Inspector
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    
    # Check if the tables exist before trying to modify them
    tables = inspector.get_table_names()
    
    # Helper function to check if a column exists in a table
    def column_exists(table, column):
        if table not in tables:
            return False
        columns = [c['name'] for c in inspector.get_columns(table)]
        return column in columns

    # Handle crop_info table
    if 'crop_info' in tables:
        if not column_exists('crop_info', 'device_address'):
            # Add device_address column to crop_info table
            op.add_column('crop_info', sa.Column('device_address', sa.String(length=256), nullable=True))
            
            # Set default value for existing records
            op.execute("UPDATE crop_info SET device_address = 'default_device' WHERE device_address IS NULL")
        
        # Add composite index if it doesn't exist
        # Note: creating an index that already exists will fail, so we need to check first
        indexes = inspector.get_indexes('crop_info')
        if not any(idx['name'] == 'ix_crop_info_filename_device' for idx in indexes):
            op.create_index('ix_crop_info_filename_device', 'crop_info', ['filename', 'device_address'], unique=False)
        
        # Update primary key constraint if necessary
        pk_constraint = inspector.get_pk_constraint('crop_info')
        
        # Check if we need to update the primary key
        if pk_constraint and 'device_address' not in pk_constraint.get('constrained_columns', []):
            try:
                constraint_name = pk_constraint.get('name', 'crop_info_pkey')
                op.drop_constraint(constraint_name, 'crop_info', type_='primary')
                op.create_primary_key('crop_info_pkey', 'crop_info', ['filename', 'device_address'])
            except Exception as e:
                # Log error and continue
                print(f"Error updating crop_info primary key: {e}")
    
    # Handle screenshot_crop_info table
    if 'screenshot_crop_info' in tables:
        if not column_exists('screenshot_crop_info', 'device_address'):
            # Add device_address column to screenshot_crop_info table
            op.add_column('screenshot_crop_info', sa.Column('device_address', sa.String(length=256), nullable=True))
            
            # Set default value for existing records
            op.execute("UPDATE screenshot_crop_info SET device_address = 'default_device' WHERE device_address IS NULL")
        
        # Add composite index if it doesn't exist
        indexes = inspector.get_indexes('screenshot_crop_info')
        if not any(idx['name'] == 'ix_screenshot_crop_info_filename_device' for idx in indexes):
            op.create_index('ix_screenshot_crop_info_filename_device', 'screenshot_crop_info', ['filename', 'device_address'], unique=False)
        
        # Update primary key constraint if necessary
        pk_constraint = inspector.get_pk_constraint('screenshot_crop_info')
        
        # Check if we need to update the primary key
        if pk_constraint and 'device_address' not in pk_constraint.get('constrained_columns', []):
            try:
                constraint_name = pk_constraint.get('name', 'screenshot_crop_info_pkey')
                op.drop_constraint(constraint_name, 'screenshot_crop_info', type_='primary')
                op.create_primary_key('screenshot_crop_info_pkey', 'screenshot_crop_info', ['filename', 'device_address'])
            except Exception as e:
                # Log error and continue
                print(f"Error updating screenshot_crop_info primary key: {e}")


def downgrade():
    from sqlalchemy.engine.reflection import Inspector
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    
    # Check if the tables exist before trying to modify them
    tables = inspector.get_table_names()
    
    # Helper function to check if a column exists in a table
    def column_exists(table, column):
        if table not in tables:
            return False
        columns = [c['name'] for c in inspector.get_columns(table)]
        return column in columns
    
    # Handle crop_info table
    if 'crop_info' in tables and column_exists('crop_info', 'device_address'):
        # Try to drop the composite primary key
        try:
            pk_constraint = inspector.get_pk_constraint('crop_info')
            constraint_name = pk_constraint.get('name', 'crop_info_pkey')
            op.drop_constraint(constraint_name, 'crop_info', type_='primary')
            
            # Recreate the original primary key on just filename
            op.create_primary_key('crop_info_pkey', 'crop_info', ['filename'])
        except Exception as e:
            print(f"Error reverting crop_info primary key: {e}")
        
        # Try to drop the index
        try:
            indexes = inspector.get_indexes('crop_info')
            if any(idx['name'] == 'ix_crop_info_filename_device' for idx in indexes):
                op.drop_index('ix_crop_info_filename_device', table_name='crop_info')
        except Exception as e:
            print(f"Error dropping crop_info index: {e}")
        
        # Drop the device_address column
        op.drop_column('crop_info', 'device_address')
    
    # Handle screenshot_crop_info table
    if 'screenshot_crop_info' in tables and column_exists('screenshot_crop_info', 'device_address'):
        # Try to drop the composite primary key
        try:
            pk_constraint = inspector.get_pk_constraint('screenshot_crop_info')
            constraint_name = pk_constraint.get('name', 'screenshot_crop_info_pkey')
            op.drop_constraint(constraint_name, 'screenshot_crop_info', type_='primary')
            
            # Recreate the original primary key on just filename
            op.create_primary_key('screenshot_crop_info_pkey', 'screenshot_crop_info', ['filename'])
        except Exception as e:
            print(f"Error reverting screenshot_crop_info primary key: {e}")
        
        # Try to drop the index
        try:
            indexes = inspector.get_indexes('screenshot_crop_info')
            if any(idx['name'] == 'ix_screenshot_crop_info_filename_device' for idx in indexes):
                op.drop_index('ix_screenshot_crop_info_filename_device', table_name='screenshot_crop_info')
        except Exception as e:
            print(f"Error dropping screenshot_crop_info index: {e}")
        
        # Drop the device_address column
        op.drop_column('screenshot_crop_info', 'device_address')