"""Run Alembic migrations programmatically."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from src.settings.app_settings import settings


def _get_alembic_config() -> Config:
    project_root = Path(__file__).resolve().parents[2]
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    db_path = Path(settings.database.path).resolve()
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return alembic_cfg


def run_migrations() -> None:
    """Apply all pending Alembic migrations."""
    command.upgrade(_get_alembic_config(), "head")
