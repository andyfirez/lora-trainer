"""Strip version suffix from lora_name in stored training config YAML."""

import re
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LORA_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def _strip_lora_version_suffix(name: str) -> str:
    return _LORA_VERSION_SUFFIX_RE.sub("", name)


def _normalize_training_config_yaml(config_yaml: str) -> str:
    data = yaml.safe_load(config_yaml) or {}
    lora_name = data.get("lora_name")
    if not isinstance(lora_name, str):
        return config_yaml
    base_name = _strip_lora_version_suffix(lora_name)
    if lora_name == base_name:
        return config_yaml
    data["lora_name"] = base_name
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def upgrade() -> None:
    connection = op.get_bind()
    for table in ("job_configs", "job_config_versions", "jobs"):
        rows = connection.execute(sa.text(f"SELECT id, config_yaml FROM {table}")).fetchall()
        for row_id, config_yaml in rows:
            if not config_yaml:
                continue
            normalized = _normalize_training_config_yaml(config_yaml)
            if normalized != config_yaml:
                connection.execute(
                    sa.text(f"UPDATE {table} SET config_yaml = :config_yaml WHERE id = :id"),
                    {"config_yaml": normalized, "id": row_id},
                )


def downgrade() -> None:
    pass
