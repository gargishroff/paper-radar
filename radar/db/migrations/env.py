from logging.config import fileConfig

from alembic import context

from radar.db import make_engine
from radar.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    # Tests pass an explicit URL through the Alembic config; otherwise use DATABASE_URL.
    engine = make_engine(config.get_main_option("sqlalchemy.url") or None)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
