"""Update min_tags default value to 5

Revision ID: update_min_tags_default
Revises: add_min_tags_column
Create Date: 2025-03-08 08:51:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text


# revision identifiers, used by Alembic.
revision = 'update_min_tags_default'
down_revision = 'add_min_tags_column'
branch_labels = None
depends_on = None


def upgrade():
    # Update existing records to use 5 as the default value for min_tags
    op.execute(text("UPDATE user_config SET min_tags = 5 WHERE min_tags = 3 OR min_tags IS NULL"))
    
    # Update the default value for new records
    # Note: This doesn't change the default in the database schema itself,
    # as SQLAlchemy handles defaults at the application level.
    # The actual schema default is changed in the models.py file.


def downgrade():
    # Revert to the original default value of 3
    op.execute(text("UPDATE user_config SET min_tags = 3 WHERE min_tags = 5"))