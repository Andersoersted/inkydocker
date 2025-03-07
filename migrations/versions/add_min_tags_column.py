"""Add min_tags column to UserConfig

Revision ID: add_min_tags_column
Revises: add_refresh_screenshot_column
Create Date: 2025-03-07 07:52:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_min_tags_column'
down_revision = 'add_refresh_screenshot_column'
branch_labels = None
depends_on = None


def upgrade():
    # Add min_tags column to user_config table with default value of 3
    op.add_column('user_config', sa.Column('min_tags', sa.Integer(), nullable=True, server_default='3'))


def downgrade():
    # Remove min_tags column from user_config table
    op.drop_column('user_config', 'min_tags')