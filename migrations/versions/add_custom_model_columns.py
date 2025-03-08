"""Add custom model columns

Revision ID: add_custom_model_columns
Revises: update_min_tags_default
Create Date: 2025-03-08 09:57:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_custom_model_columns'
down_revision = 'update_min_tags_default'
branch_labels = None
depends_on = None


def upgrade():
    # Add custom_model and custom_model_enabled columns to user_config table
    op.add_column('user_config', sa.Column('custom_model', sa.String(length=256), nullable=True))
    op.add_column('user_config', sa.Column('custom_model_enabled', sa.Boolean(), nullable=True, server_default='0'))


def downgrade():
    # Remove the columns
    op.drop_column('user_config', 'custom_model_enabled')
    op.drop_column('user_config', 'custom_model')