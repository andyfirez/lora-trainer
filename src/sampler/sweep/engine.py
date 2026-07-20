"""Unified sweep sampling engine."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from src.sampler.config import SamplingConfig
from src.sampler.sdxl.service import SDXLLoRASampler
from src.sampler.sweep.combinations import build_combinations
from src.sampler.sweep.grid_compositor import compose_grid
from src.sampler.sweep.grid_planner import plan_grids
from src.sampler.sweep.manifest import (
    GRIDS_SUBDIR,
    IMAGES_SUBDIR,
    ManifestGridAxis,
    ManifestGridEntry,
    ManifestImageEntry,
    SweepManifest,
    cell_image_path,
    grid_image_path,
    write_manifest,
)
from src.sampler.sweep.models import parse_trigger_words
from src.trainer.concept_training_metadata import ConceptTrainingMetadata
from src.trainer.config import SampleScheduler, TrainConfig
from src.trainer.sdxl.caption import apply_trigger_words_to_prompt

ProgressStatusCallback = Callable[[str | None], None]
ProgressCallback = Callable[[int, int], None]


def sort_pipeline_groups(
    groups: dict[tuple[str, str | None], list],
) -> list[tuple[tuple[str, str | None], list]]:
    """Order pipeline loads by base model first, then LoRA path, to minimize model switches."""
    return sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or ""))


class SweepEngine:
    def __init__(
        self,
        sampling_config: SamplingConfig,
        *,
        base_train_config: TrainConfig,
        output_dir: Path,
        job_id: int | None = None,
        progress_status_callback: ProgressStatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        log: logging.Logger | None = None,
        concept_metadata: dict[int, ConceptTrainingMetadata] | None = None,
        compose_grids: bool = True,
    ) -> None:
        self._sampling_config = sampling_config
        self._base_train_config = base_train_config
        self._output_dir = output_dir
        self._job_id = job_id
        self._progress_status_callback = progress_status_callback
        self._progress_callback = progress_callback
        self._log = log or logging.getLogger(__name__)
        self._concept_metadata = concept_metadata or {}
        self._compose_grids = compose_grids
        self._sampler = SDXLLoRASampler(
            base_train_config,
            lora_paths=[],
            output_dir=output_dir,
            progress_status_callback=progress_status_callback,
            progress_callback=progress_callback,
            log=self._log,
            concept_metadata=concept_metadata,
        )

    def run(self) -> SweepManifest:
        parameters = self._sampling_config.parameters
        combinations = build_combinations(parameters)
        if not combinations:
            raise ValueError("No sample prompts configured")

        self._log.info(
            "Sweep engine: %d image(s) to generate, output -> %s",
            len(combinations),
            self._output_dir,
        )

        images_dir = self._output_dir / IMAGES_SUBDIR
        grids_dir = self._output_dir / GRIDS_SUBDIR
        images_dir.mkdir(parents=True, exist_ok=True)
        if self._compose_grids:
            grids_dir.mkdir(parents=True, exist_ok=True)

        total_steps = len(combinations) * int(parameters.steps.first_value() or 30)
        self._set_progress(0, total_steps)
        completed_images = 0

        groups: dict[tuple[str, str | None], list] = defaultdict(list)
        for combo in combinations:
            base_model = str(combo.params.get("base_model_name") or self._base_train_config.base_model_name)
            lora_path = combo.params.get("lora_path")
            lora_key = str(lora_path) if lora_path else None
            groups[(base_model, lora_key)].append(combo)

        sorted_groups = sort_pipeline_groups(groups)
        self._log.info(
            "Sweep load plan: %d pipeline group(s) (%d with LoRA, %d base-only)",
            len(sorted_groups),
            sum(1 for (_, k) in sorted_groups if k[1]),
            sum(1 for (_, k) in sorted_groups if not k[1]),
        )
        for index, ((base_model, lora_key), group_combos) in enumerate(sorted_groups, start=1):
            lora_label = Path(lora_key).name if lora_key else "(none)"
            self._log.info(
                "  Group %d: base=%s, lora=%s, cells=%d",
                index,
                base_model,
                lora_label,
                len(group_combos),
            )

        for group_index, ((base_model, lora_key), group_combos) in enumerate(sorted_groups, start=1):
            first = group_combos[0]
            lora_path = Path(lora_key) if lora_key else None
            status = f"Sampling {lora_path.name if lora_path else 'base model'}"
            self._log.info(
                "Pipeline group %d/%d: base=%s, lora=%s, cells=%d",
                group_index,
                len(sorted_groups),
                base_model,
                lora_path if lora_path else "(none)",
                len(group_combos),
            )
            self._set_status(status)
            stack, lora_config, merge_unet = self._sampler.load_stack_for_combo(
                base_model=base_model,
                lora_path=lora_path,
                combo_params=first.params,
            )
            try:
                for combo in group_combos:
                    self._generate_cell(
                        combo=combo,
                        stack=stack,
                        lora_config=lora_config,
                        merge_unet=merge_unet,
                        images_dir=images_dir,
                        completed_images=completed_images,
                        total_steps=total_steps,
                    )
                    completed_images += 1
            finally:
                del stack
                torch.cuda.empty_cache()

        self._log.info("Composing grid images...")
        manifest = self._build_manifest(combinations, images_dir, grids_dir)
        write_manifest(self._output_dir, manifest)
        self._log.info("Sweep complete: %d image(s), %d grid(s)", manifest.total_images, len(manifest.grids))
        self._set_status(None)
        return manifest

    def _generate_cell(
        self,
        *,
        combo: Any,
        stack: Any,
        lora_config: TrainConfig,
        merge_unet: bool,
        images_dir: Path,
        completed_images: int,
        total_steps: int,
    ) -> None:
        params = combo.params
        raw_prompt = str(params.get("prompt") or "")
        trigger = str(params.get("lora_trigger") or "")
        prompt = apply_trigger_words_to_prompt(raw_prompt, parse_trigger_words(trigger))
        sampling_config = self._build_runtime_config(params)
        lora_weight = float(params.get("lora_weight") or 1.0)
        filename = cell_image_path(self._output_dir, combo.index).name
        self._set_status(f"Cell {combo.index + 1}: {prompt[:60]}")
        self._sampler.generate_single_cell(
            stack=stack,
            lora_config=lora_config,
            sampling_config=sampling_config,
            merge_unet=merge_unet,
            prompt=prompt,
            lora_weight=lora_weight,
            output_dir=images_dir,
            output_filename=filename,
            completed_images=completed_images,
            total_steps=total_steps,
        )

    def _build_runtime_config(self, params: dict[str, Any]) -> TrainConfig:
        updates: dict[str, Any] = {
            "sample_prompts": [str(params.get("prompt") or "")],
            "sample_negative_prompt": str(params.get("negative_prompt") or ""),
            "sample_steps": int(params.get("steps") or 30),
            "sample_cfg_scale": float(params.get("cfg_scale") or 7.5),
            "sample_width": params.get("width"),
            "sample_height": params.get("height"),
        }
        scheduler = params.get("scheduler")
        if scheduler is not None:
            updates["sample_scheduler"] = SampleScheduler(str(scheduler))
        seed = params.get("seed")
        if seed is not None:
            updates["seed"] = int(seed)
        base_model = params.get("base_model_name")
        if base_model is not None:
            updates["base_model_name"] = str(base_model)
        return self._base_train_config.model_copy(update=updates)

    def _build_manifest(self, combinations: list, images_dir: Path, grids_dir: Path) -> SweepManifest:
        image_entries = [
            ManifestImageEntry(
                index=combo.index,
                file=str(cell_image_path(self._output_dir, combo.index).relative_to(self._output_dir)),
                params=combo.params,
            )
            for combo in combinations
        ]
        grid_entries: list[ManifestGridEntry] = []
        if self._compose_grids:
            plans = plan_grids(self._sampling_config.parameters, self._sampling_config.grid)
            for plan in plans:
                cell_paths: list[list[Path | None]] = []
                for row in plan.cells:
                    cell_paths.append(
                        [
                            images_dir / f"cell_{idx:04d}.png" if idx is not None else None
                            for idx in row
                        ]
                    )
                grid_file = grid_image_path(self._output_dir, plan.index, plan.title)
                compose_grid(
                    cell_paths,
                    x_axis=plan.x_axis,
                    y_axis=plan.y_axis,
                    x_values=plan.x_values,
                    y_values=plan.y_values,
                    title=plan.title,
                    output_path=grid_file,
                )
                grid_entries.append(
                    ManifestGridEntry(
                        index=plan.index,
                        file=str(grid_file.relative_to(self._output_dir)),
                        slice=plan.slice_params,
                        x=ManifestGridAxis(param=plan.x_axis, values=plan.x_values),
                        y=ManifestGridAxis(param=plan.y_axis, values=plan.y_values),
                        cells=plan.cells,
                        title=plan.title,
                    )
                )
        return SweepManifest(
            job_id=self._job_id,
            total_images=len(combinations),
            images=image_entries,
            grids=grid_entries,
        )

    def _set_status(self, status: str | None) -> None:
        if self._progress_status_callback is not None:
            self._progress_status_callback(status)

    def _set_progress(self, step: int, total: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(step, total)
