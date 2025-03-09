"""add RAM model columns

Revision ID: add_ram_model_columns
Revises: add_similarity_threshold_column
Create Date: 2025-03-09 09:01:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_ram_model_columns'
down_revision = 'add_similarity_threshold_column'
branch_labels = None
depends_on = None


def upgrade():
    # Add RAM model columns to user_config table
    op.add_column('user_config', sa.Column('ram_enabled', sa.Boolean(), nullable=True, server_default='1'))
    op.add_column('user_config', sa.Column('ram_model', sa.String(64), nullable=True, server_default='ram_medium'))
    op.add_column('user_config', sa.Column('ram_min_confidence', sa.Float(), nullable=True, server_default='0.3'))


def downgrade():
    # Remove RAM model columns from user_config table
    op.drop_column('user_config', 'ram_enabled')
    op.drop_column('user_config', 'ram_model')
    op.drop_column('user_config', 'ram_min_confidence')