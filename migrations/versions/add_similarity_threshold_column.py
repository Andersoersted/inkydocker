"""Add similarity threshold column

Revision ID: add_similarity_threshold_column
Revises: add_custom_model_columns
Create Date: 2025-03-08 10:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_similarity_threshold_column'
down_revision = 'add_custom_model_columns'
branch_labels = None
depends_on = None


def upgrade():
    # Add similarity_threshold column to user_config table
    op.add_column('user_config', sa.Column('similarity_threshold', sa.String(length=20), nullable=True, server_default='medium'))


def downgrade():
    # Remove the column
    op.drop_column('user_config', 'similarity_threshold')