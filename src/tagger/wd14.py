"""WD14 ONNX tagger for booru-style captions."""

import csv
from pathlib import Path

import huggingface_hub
import numpy as np
import onnxruntime
from PIL import Image


class WD14Tagger:
    def __init__(self, model_repo: str, *, use_cuda: bool = True) -> None:
        model_path = huggingface_hub.hf_hub_download(model_repo, "model.onnx")
        providers = onnxruntime.get_available_providers()
        if use_cuda and "CUDAExecutionProvider" in providers:
            provider: str | tuple[str, dict[str, str]] = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
        self._session = onnxruntime.InferenceSession(model_path, providers=[provider])

        label_path = huggingface_hub.hf_hub_download(model_repo, "selected_tags.csv")
        self._tag_names: list[str] = []
        self._general_indexes: list[int] = []
        with open(label_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                self._tag_names.append(row["name"])
                if row["category"] == "0":
                    self._general_indexes.append(index)

        input_shape = self._session.get_inputs()[0].shape
        self._height = int(input_shape[1])
        self._width = int(input_shape[2])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(
        self,
        image_path: Path,
        *,
        threshold: float = 0.35,
        strip_rating: bool = True,
    ) -> list[str]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self._width, self._height))
            array = np.asarray(image)[:, :, ::-1].astype(np.float32)
            array = np.expand_dims(array, 0)

        probs = self._session.run([self._output_name], {self._input_name: array})[0][0].astype(float)
        labels = [
            (self._tag_names[index], probs[index])
            for index in self._general_indexes
            if probs[index] > threshold
        ]
        labels.sort(key=lambda item: item[1], reverse=True)
        tags = [name.replace("_", " ") for name, _ in labels]
        if strip_rating:
            tags = [tag for tag in tags if not tag.startswith("rating:")]
        return tags
