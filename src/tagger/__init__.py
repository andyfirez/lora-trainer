"""WD14 autotagging — pure ML orchestration layer.

Tagging intentionally runs in-process (not via the runnable subprocess queue):
see ``src.services.tagging.manager`` for the execution model and rationale.
"""

from src.tagger.config import TaggingConfig, TaggingMode, WD14_MODEL_REPOS
from src.tagger.runner import TaggingProgress, TaggingResult, tag_dataset
from src.tagger.wd14 import WD14Tagger

__all__ = [
    "TaggingConfig",
    "TaggingMode",
    "TaggingProgress",
    "TaggingResult",
    "WD14_MODEL_REPOS",
    "WD14Tagger",
    "tag_dataset",
]
