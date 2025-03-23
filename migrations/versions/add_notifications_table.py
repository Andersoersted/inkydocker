"""add notifications table

Revision ID: 912e743f41d8
Revises: (enter the ID of the latest migration here)
Create Date: 2023-07-24 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timedelta


# revision identifiers, used by Alembic.
revision = '912e743f41d8'
down_revision = None  # Set this to the ID of the latest migration
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('notifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('message', sa.String(length=512), nullable=False),
        sa.Column('type', sa.String(length=20), nullable=False, server_default='info'),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('notifications')