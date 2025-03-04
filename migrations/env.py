from __future__ import with_statement
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Get the Alembic config and set up logging
config = context.config
fileConfig(config.config_file_name)

# Determine the project root and ensure the data folder exists.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ensure an empty database file exists so that autogenerate has something to compare.
db_path = os.path.join(data_dir, 'mydb.sqlite')
if not os.path.exists(db_path):
    open(db_path, 'a').close()

# Import all your models so that they are registered with SQLAlchemy's metadata.
from models import db, Device, ImageDB, CropInfo, SendLog, ScheduleEvent, UserConfig, DeviceMetrics
target_metadata = db.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()