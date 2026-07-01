# Bucketing + add_time_ids collate fix (июль 2026)

Отчёт о реализации aspect-ratio bucketing (offline bake) и последующем прогоне `Winx_Bloom_CFTS`.

## Что реализовано в коде

### Bucketing (гипотеза C)

- `src/trainer/sdxl/buckets.py` — Kohya-compatible `assign_bucket`, `compute_add_time_ids`, `make_bucket_resolutions`
- Preprocess/service — non-square bake, метаданные crop в БД, invalidation latent cache при rebake
- `BucketBatchSampler` — батчи только внутри одного bucket resolution
- `ConceptDataset` + `concept_training_metadata.py` — per-image `add_time_ids` при обучении

Семантика `add_time_ids`: `(source_h, source_w, crop_top, crop_y, target_h, target_w)` — порядок Kohya.

### Баг collate `add_time_ids` (batch_size > 1)

**Симптом:** при `batch_size=2` SDXL micro-conditioning перемешивался между сэмплами в батче → рост loss, шум на генерации.

**Причина:** `ConceptDataset.__getitem__` возвращал `add_time_ids` как Python `tuple`. `default_collate` транспонирует список tuple'ов в 6 тензоров формы `(batch_size,)`, а не `(batch_size, 6)`. UNet в diffusers делает `time_ids.flatten().reshape((batch_size, -1))` — при `batch_size>1` значения разных картинок смешиваются.

**Fix (июль 2026):**

- `src/trainer/sdxl/dataset.py` — `add_time_ids` как `torch.tensor(..., dtype=float32)` в `__getitem__`
- `src/trainer/sdxl/trainer.py` — удалён `_stack_add_time_ids`, прямой `.to(device/dtype)` от батч-тензора
- Регрессионный тест: `tests/trainer/test_dataset_prepared.py::test_dataloader_collate_add_time_ids_without_cross_sample_mixing`

При `batch_size=1` баг численно не проявлялся (транспонирование `(6,1)` эквивалентно `(1,6)`).

## Прогон Winx_Bloom_CFTS (после fix collate)

**Подтверждено пользователем:** запуск выполнен **после** fix collate, с нуля (без resume).

**Конфиг:** `D:/SD/lora_output/Winx_Bloom_CFTS/Winx_Bloom_CFTS_config.yaml`

| Параметр | Значение |
|----------|----------|
| base_model | illustriousXL_v01.safetensors |
| rank / alpha | 16 / 16 |
| learning_rate | 1e-3 |
| lr_scheduler | cosine, warmup=0 |
| batch_size | 2 |
| enable_bucket | true |
| bucket_reso_steps | 64 |
| bucket_no_upscale | true |
| mixed_precision | float16 |
| optimizer | adamw_8bit |
| epochs | 10 |
| repeats | 2 |
| min_snr_gamma / noise_offset | 0 / 0 |

**Артефакты:** `D:/SD/lora_output/Winx_Bloom_CFTS/` — checkpoints, `samples/`, `loss_log.db`

### Прогресс сэмплов по эпохам

| Эпоха | Файл | Наблюдение |
|-------|------|------------|
| 1 | `samples/Winx_Bloom_CFTS_epoch1_00.png` | Связное изображение, узнаваемый персонаж, нормальные цвета |
| 2 | `samples/Winx_Bloom_CFTS_epoch2_00.png` | Сильная деградация: белый фон, фиолетовые артефакты |
| 3 | `samples/Winx_Bloom_CFTS_epoch3_00.png` | Абстрактные полосы, структура потеряна |
| 5 | `samples/Winx_Bloom_CFTS_epoch5_00.png` | Чистый TV-статик |
| 10 | `samples/Winx_Bloom_CFTS_epoch10_00.png` | Чистый TV-статик |

Шум одинаково в **встроенных превью** (latent sampling во время train job) и в **reForge** на финальных checkpoint'ах.

### Loss log (`loss_log.db`)

800 шагов, `loss/loss` на всём интервале **0.13–1.2** — без монотонного роста, без NaN/Inf.

**Интерпретация:** MSE на случайных timesteps остаётся «нормальным», пока LoRA-веса при итеративном denoising уже дают коллапс. Уточнённая причина: **fp16 LoRA weights + нет GradScaler** (гипотеза N), не параметры AdamW. См. `07-fp16-gradscaler-vs-kohya.md`.

## Выводы

1. **Fix collate необходим** при `batch_size>1`, но **не объясняет** деградацию ep1→ep10 (fix уже был применён).
2. **Bucketing сам по себе не решил likeness** — ep1 даёт надежду, дальше модель «ломается».
3. **Наиболее вероятная причина деградации (N):** LoRA weights в fp16 + `loss.backward()` без GradScaler. Kohya с тем же fp16+AdamW8bit использует fp32 weights + Accelerate scaler; betas/wd совпадают.
4. **lr=1e-3** усиливает проблему, но не объясняет расхождение с Kohya сам по себе.
5. **Train/infer mismatch add_time_ids** (M) — вторичный фактор (ep1 была ok).

## Рекомендуемый следующий шаг

**Сначала fix N в коде** (GradScaler + fp32 LoRA), затем retrain:

```yaml
learning_rate: 0.001              # Kohya-parity, после fix N
lr_warmup_steps: 40
mixed_precision: float16           # OK с GradScaler + fp32 LoRA
optimizer:
  type: adamw_8bit                  # как в Kohya
enable_bucket: true
batch_size: 2
epochs: 10
save_every_n_epochs: 1
```

bf16 **не обязателен** на GPU без bf16 support.

**Критерий успеха:** likeness к ep.3–5 без перехода в статик к ep.5–7. Лучший checkpoint — не финальный, а ранняя эпоха до деградации.

**Не resume'ить** checkpoint'ы jul-прогона **до fix N** при `batch_size>1` — conditioning в тех весах некорректен.

Post-N retrain: см. `08-fix-n-post-run-jul2026.md`. Usable checkpoint: ep2–3; ep4+ — статик.
