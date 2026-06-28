"""Tagging job configuration."""

from enum import StrEnum

import yaml
from pydantic import BaseModel, Field


class TaggingMode(StrEnum):
    IF_EMPTY = "if_empty"
    OVERWRITE = "overwrite"
    APPEND = "append"


WD14_MODEL_REPOS: dict[str, str] = {
    "wd-v1-4-convnextv2-tagger-v2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-vit-tagger-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-moat-tagger-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
}


class TaggingConfig(BaseModel):
    dataset_id: int
    image_dir: str
    mode: TaggingMode = TaggingMode.IF_EMPTY
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    model: str = "wd-v1-4-convnextv2-tagger-v2"
    caption_extension: str = ".txt"
    strip_rating: bool = True
    filenames: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "TaggingConfig":
        data = yaml.safe_load(yaml_text) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False)

    def resolve_model_repo(self) -> str:
        if self.model in WD14_MODEL_REPOS:
            return WD14_MODEL_REPOS[self.model]
        if self.model.startswith("SmilingWolf/"):
            return self.model
        raise ValueError(f"Unsupported tagging model: {self.model}")
