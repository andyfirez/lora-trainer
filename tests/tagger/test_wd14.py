from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.tagger.config import TaggingConfig
from src.tagger.wd14 import WD14Tagger


def test_tagging_config_resolve_model_repo() -> None:
    config = TaggingConfig(dataset_id=1, image_dir="/tmp", model="wd-v1-4-convnextv2-tagger-v2")
    assert config.resolve_model_repo() == "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"


def test_wd14_predict_applies_threshold() -> None:
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[0.9, 0.2, 0.8]], dtype=np.float32)]

    tagger = WD14Tagger.__new__(WD14Tagger)
    tagger._session = mock_session
    tagger._tag_names = ["solo", "rating:safe", "1girl"]
    tagger._general_indexes = [0, 1, 2]
    tagger._height = 4
    tagger._width = 4
    tagger._input_name = "input"
    tagger._output_name = "output"

    with patch("src.tagger.wd14.Image.open") as mock_image_open:
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.resize.return_value = mock_image
        mock_image_open.return_value.__enter__.return_value = mock_image
        with patch("src.tagger.wd14.np.asarray", return_value=np.zeros((4, 4, 3), dtype=np.float32)):
            tags = tagger.predict(Path("demo.png"), threshold=0.35, strip_rating=True)

    assert tags == ["solo", "1girl"]
