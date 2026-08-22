"""Unified sweep sampling engine."""

from __future__ import annotations

import logging
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from src.sampler.config import SamplingConfig
from src.sampler.progress_callbacks import (
    ProgressCallback,
    ProgressCallbackMixin,
    ProgressStatusCallback,
)
from src.sampler.sdxl.cell_generator import generate_sampling_cell
from src.sampler.output_paths import flat_grid_filename, flat_sample_filename
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
from src.trainer.config import SampleScheduler
from src.trainer.inference_config import SDXLInferenceConfig
from src.trainer.sdxl.caption import apply_trigger_words_to_prompt
from src.trainer.sdxl.latent_sampling.preview import LIVE_PREVIEW_FILENAME
from src.trainer.sdxl.pipeline_loader import SDXLPipelineLoader
from src.trainer.sdxl.sampling import PromptEmbedCache


def _pipeline_group_key(params: dict[str, Any], default_base: str) -> tuple[str, str | None]:
    base_model = str(params.get("base_model_name") or default_base)
    stack = params.get("lora_stack")
    if isinstance(stack, list) and len(stack) > 1:
        parts = [
            f"{item.get('path')}:{item.get('weight', 1.0)}"
            for item in stack
            if isinstance(item, dict) and item.get("path")
        ]
        if parts:
            return (base_model, "|".join(parts))
    lora_path = params.get("lora_path")
    return (base_model, str(lora_path) if lora_path else None)


def sort_pipeline_groups(
    groups: dict[tuple[str, str | None], list],
) -> list[tuple[tuple[str, str | None], list]]:
    """Order pipeline loads by base model first, then LoRA path, to minimize model switches."""
    return sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or ""))


class SweepEngine(ProgressCallbackMixin):
    def __init__(
        self,
        sampling_config: SamplingConfig,
        *,
        base_inference_config: SDXLInferenceConfig,
        output_dir: Path,
        sampling_id: int | None = None,
        progress_status_callback: ProgressStatusCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        log: logging.Logger | None = None,
        concept_metadata: dict[int, ConceptTrainingMetadata] | None = None,
        compose_grids: bool = True,
        flat_output: bool = True,
    ) -> None:
        self._sampling_config = sampling_config
        self._base_inference_config = base_inference_config
        self._output_dir = output_dir
        self._sampling_id = sampling_id
        self._progress_status_callback = progress_status_callback
        self._progress_callback = progress_callback
        self._log = log or logging.getLogger(__name__)
        self._concept_metadata = concept_metadata or {}
        self._compose_grids = compose_grids
        self._flat_output = flat_output
        self._pipeline_loader = SDXLPipelineLoader(base_inference_config, log=self._log)
        self._prompt_embed_cache = PromptEmbedCache()

    def run(self) -> SweepManifest | None:
        parameters = self._sampling_config.parameters
        combinations = build_combinations(parameters)
        if not combinations:
            raise ValueError("No sample prompts configured")

        self._log.info(
            "Sweep engine: %d image(s) to generate, output -> %s",
            len(combinations),
            self._output_dir,
        )

        if self._flat_output:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            images_dir = self._output_dir
            grids_dir = None
        else:
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
            groups[_pipeline_group_key(combo.params, self._base_inference_config.base_model_name)].append(
                combo
            )

        sorted_groups = sort_pipeline_groups(groups)
        self._log.info(
            "Sweep load plan: %d pipeline group(s) (%d with LoRA, %d base-only)",
            len(sorted_groups),
            sum(1 for (key, _) in sorted_groups if key[1]),
            sum(1 for (key, _) in sorted_groups if not key[1]),
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
            raw_path = first.params.get("lora_path")
            lora_path = Path(str(raw_path)) if raw_path else None
            lora_stack = first.params.get("lora_stack")
            stack_items = lora_stack if isinstance(lora_stack, list) else None
            status = f"Sampling {lora_path.name if lora_path else 'base model'}"
            self._log.info(
                "Pipeline group %d/%d: base=%s, lora=%s, cells=%d",
                group_index,
                len(sorted_groups),
                base_model,
                lora_key or "(none)",
                len(group_combos),
            )
            self._set_status(status)
            stack, lora_config, merge_unet = self._pipeline_loader.load_stack_for_combo(
                base_model=base_model,
                lora_path=lora_path,
                combo_params=first.params,
                lora_stack=stack_items,
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

        if self._flat_output:
            if self._compose_grids:
                self._log.info("Composing grid images...")
                grid_count = self._compose_flat_grids(combinations, images_dir)
                self._log.info(
                    "Sweep complete: %d image(s), %d grid(s)",
                    len(combinations),
                    grid_count,
                )
            else:
                self._log.info("Sweep complete: %d image(s)", len(combinations))
            self._set_status(None)
            return None

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
        lora_config: SDXLInferenceConfig,
        merge_unet: bool,
        images_dir: Path,
        completed_images: int,
        total_steps: int,
    ) -> None:
        params = combo.params
        raw_prompt = str(params.get("prompt") or "")
        trigger = str(params.get("lora_trigger") or "")
        prompt = apply_trigger_words_to_prompt(raw_prompt, parse_trigger_words(trigger))
        if params.get("seed") is None:
            params["seed"] = secrets.randbelow(2**31)
        sampling_config = self._build_runtime_config(params)
        lora_weight = float(params.get("lora_weight") or 1.0)
        filename = (
            flat_sample_filename(self._sampling_id, params.get("seed"), combo.index)
            if self._flat_output
            else cell_image_path(self._output_dir, combo.index).name
        )
        self._set_status(f"Cell {combo.index + 1}: {prompt[:60]}")
        lora_stack = params.get("lora_stack")
        generate_sampling_cell(
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
            concept_metadata=self._concept_metadata,
            prompt_embed_cache=self._prompt_embed_cache,
            log=self._log,
            on_progress=self._set_progress,
            lora_path=str(params["lora_path"]) if params.get("lora_path") else None,
            lora_stack=lora_stack if isinstance(lora_stack, list) else None,
            preview_path=self._output_dir / LIVE_PREVIEW_FILENAME,
        )

    def _build_runtime_config(self, params: dict[str, Any]) -> SDXLInferenceConfig:
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
        return self._base_inference_config.model_copy(update=updates)

    def _compose_flat_grids(self, combinations: list, images_dir: Path) -> int:
        index_to_filename = {
            combo.index: flat_sample_filename(
                self._sampling_id,
                combo.params.get("seed"),
                combo.index,
            )
            for combo in combinations
        }
        plans = plan_grids(self._sampling_config.parameters, self._sampling_config.grid)
        for plan in plans:
            cell_paths: list[list[Path | None]] = []
            for row in plan.cells:
                cell_paths.append(
                    [
                        images_dir / index_to_filename[idx] if idx is not None else None
                        for idx in row
                    ]
                )
            grid_file = self._output_dir / flat_grid_filename(
                self._sampling_id,
                plan.index,
                plan.title,
            )
            compose_grid(
                cell_paths,
                x_axis=plan.x_axis,
                y_axis=plan.y_axis,
                x_values=plan.x_values,
                y_values=plan.y_values,
                title=plan.title,
                output_path=grid_file,
            )
        return len(plans)

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
            sampling_id=self._sampling_id,
            total_images=len(combinations),
            images=image_entries,
            grids=grid_entries,
        )
