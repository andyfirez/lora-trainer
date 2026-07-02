# Fix M: aligned inference add_time_ids (validation-first)

Отчёт о реализации inference-side fix для гипотезы **M** (июль 2026).

> **Итог валидации (июль 2026): M отклонена как root cause ep4+ шума.** Ручной прогон дал результат, идентичный до-fix (no-op). Подробности и пересмотр — `11-revised-ep4-noise-jul2026.md`. Fix оставлен как низкоприоритетный косметический для in-app preview при сэмплинге ровно в train-bucket'е.

## Проблема

- **Train:** per-image `add_time_ids` из bucketing — `(source_h, source_w, crop_top, crop_left, target_h, target_w)`, crop часто ≠ 0.
- **Inference (до fix):** всегда `(H, W, 0, 0, H, W)` в `latent_sampling/session.py`.
- Симптом: ep4+ шум при lr3e-4 constant; Kohya стабилен при любых LR.

## Реализация (код)

### 1. `resolve_reference_add_time_ids`

Файл: [`src/trainer/concept_training_metadata.py`](../src/trainer/concept_training_metadata.py)

Для заданных `width`×`height` (sample resolution) находит все train-изображения с matching bucket, возвращает **median** `(source_h, source_w, crop_top, crop_left, target_h, target_w)`. Если match нет → `None` → fallback `(H,W,0,0,H,W)`.

### 2. `SDXLSamplingSession.create`

Файл: [`src/trainer/sdxl/latent_sampling/session.py`](../src/trainer/sdxl/latent_sampling/session.py)

Новый параметр `reference_add_time_ids: tuple[float, ...] | None = None`.

### 3. Built-in preview

Файл: [`src/trainer/sdxl/trainer.py`](../src/trainer/sdxl/trainer.py) — `_run_sampling`

Использует `self._concept_metadata` + `config.concepts` dataset IDs. Лог: `Sampling eN: using aligned add_time_ids ...`.

### 4. Standalone sampler

- [`src/sampler/sdxl/service.py`](../src/sampler/sdxl/service.py) — `concept_metadata` в конструкторе; `_reference_dataset_ids()` для sampling jobs без `concepts` в config.
- [`src/sampler/job_runner.py`](../src/sampler/job_runner.py) — при `source_job_id` → training job resolve `concept_metadata` через DB (crop records).

### Тесты

- `tests/trainer/test_concept_training_metadata.py` — median, no match, multi-dataset.
- `tests/trainer/sdxl/test_sampling_session.py` — reference override.
- **132 passed** в `tests/trainer` + `tests/sampler`.

## Ручная валидация — ВЫПОЛНЕНА (результат: no-op)

**Статус: выполнена, M отклонена.**

Прогон: standalone sampling job 25 (`source_job_id=23`, Bloom lr3e-4 constant), ep1–10, `sample_width=832`, `sample_height=1216`, фиксированные prompt/seed.

**Результат:** изображения полностью идентичны прогону до fix M.

**Причина no-op:** датасет обучался с `target_resolution=1024`, поэтому max сторона любого bucket'а = 1024. Bucket'а `832×1216` не существует → `resolve_reference_add_time_ids` вернул `None` для всех эпох → fallback на старый хардкод `(1216,832,0,0,1216,832)`. В логах job 25 строка `using aligned add_time_ids` отсутствует.

**Почему это не просто «неверное разрешение теста»:** цель — генерировать выше train-разрешения. Для high-res txt2img корректное inference conditioning — `(H,W,0,0,H,W)`, а не train crop coords. Плюс ep4+ шум виден и в reForge, где Kohya-LoRA на тех же данных стабильна → root cause в весах checkpoint'ов, не в inference. См. `11-revised-ep4-noise-jul2026.md`.

### Критерии (для истории)

| Результат | Интерпретация |
|-----------|---------------|
| Шум уходит, изображение связное | M подтверждена — inference mismatch был root cause ep4+ |
| Шум сохраняется / no-op | **M отклонена (фактический исход)** — приоритет D (train loop audit) + O (checkpoint weights) |

### Без retrain

Fix влияет только на **generation conditioning**; существующие checkpoint'ы ep4+ можно пересэмплировать с новым кодом.

## Что не покрыто

- **reForge** — внешний инструмент; fix только built-in preview + standalone sampler.
- **Train-side bucketing** (fit+pad) — отдельное решение после результата валидации.

## Связанные документы

- `09-lr-constant-post-run-jul2026.md` — симптом ep4+ до fix M
- `03-hypotheses.md` — статус M
- `04-pipelines.md` — train vs infer diff
