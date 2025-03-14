"""add zero shot model columns

Revision ID: add_zero_shot_model_columns
Revises: add_ram_model_columns
Create Date: 2025-03-11 20:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_zero_shot_model_columns'
down_revision = 'add_ram_model_columns'
branch_labels = None
depends_on = None


def upgrade():
    # Add zero-shot model columns to user_config table
    op.add_column('user_config', sa.Column('zero_shot_enabled', sa.Boolean(), nullable=True, default=True))
    op.add_column('user_config', sa.Column('zero_shot_model', sa.String(64), nullable=True, default='base'))
    op.add_column('user_config', sa.Column('zero_shot_min_confidence', sa.Float(), nullable=True, default=0.3))
    
    # Set default values for existing rows
    op.execute("UPDATE user_config SET zero_shot_enabled = 1")
    op.execute("UPDATE user_config SET zero_shot_model = 'base'")
    op.execute("UPDATE user_config SET zero_shot_min_confidence = 0.3")


def downgrade():
    # Remove zero-shot model columns from user_config table
    op.drop_column('user_config', 'zero_shot_enabled')
    op.drop_column('user_config', 'zero_shot_model')
    op.drop_column('user_config', 'zero_shot_min_confidence')