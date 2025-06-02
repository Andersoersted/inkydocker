"""Add Immich tables

Revision ID: add_immich_tables
Revises: fix_crop_updated_at
Create Date: 2025-04-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_immich_tables'
down_revision = 'fix_crop_updated_at'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'immich_config',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('address', sa.String(length=256), nullable=False),
        sa.Column('api_key', sa.String(length=256), nullable=False),
        sa.Column('active', sa.Boolean, nullable=True, server_default='1')
    )

    op.create_table(
        'immich_gallery',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('gallery_id', sa.String(length=64), nullable=False, unique=True),
        sa.Column('name', sa.String(length=256), nullable=False),
        sa.Column('selected', sa.Boolean, nullable=True, server_default='0')
    )


def downgrade():
    op.drop_table('immich_gallery')
    op.drop_table('immich_config')
