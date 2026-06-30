# Гипотезы

## На проверку (приоритет)

| ID | Гипотеза | Приоритет | Как проверить |
|----|----------|-----------|---------------|
| **G** | Effective LR в ~20× ниже Kohya: `(alpha/rank) × lr` = 0.5×1e-4 vs 1.0×1e-3 | **P0** | Retrain Bloom: rank=16, alpha=16, lr=1e-3 (Kohya parity) |
| **B** | `lora_alpha/rank = 0.5` ослабляет LoRA (часть G) | **P0** | Вместе с G: alpha=rank |
| **C** | Нет bucketing → square crop портит identity | **P0** | Реализовать bucket + retrain |
| **H** | `add_time_ids.original_size` = `(R,R)` вместо source dims (1216×918) | **P1** | Передавать source size из crop meta в train loop |
| **I** | `add_time_ids.crops_coords` = `(0,0)` вместо реального crop offset (~96,0) | **P1** | Вместе с H: coords из `preprocess.py` |
| **D** | Баг в train loop (target, latents, scheduler) | **P1** | Diff vs Kohya sd-scripts: encode → noise → loss → backward |
| **J** | fp16 → нестабильные градиенты (up_mean растёт без likeness) | **P1** | Retrain с `mixed_precision: bfloat16` |
| **E** | Overcooking (Chimera) | **P2** | Смотреть epoch 1–3, не 20; снизить lr/repeats |
| **F** | max_token_length 77 vs Kohya 250 | **P2** | Длинные caption обрезаются молча |
| **K** | AdamW8bit при низком LR → квантизационный шум | **P2** | Тест с `optimizer: adamw` (fp32 state) |

### Детали G (effective LR)

Peft масштабирует LoRA как `(alpha/rank) × BA`. Итоговый step size:

| | alpha/rank | lr | effective |
|---|------------|-----|-----------|
| lora-trainer (job 9/13) | 16/32 = 0.5 | 1e-4 | **5e-5** |
| Kohya (Winx) | 16/16 = 1.0 | 1e-3 | **1e-3** |

Разрыв **~20×**. Симптом совпадает: веса растут (up_mean 0.13→0.54), likeness нет — LoRA едва отходит от zero-init.

### Детали H/I (add_time_ids)

Hypothesis A отвергла **inference** AR mismatch. Это другая проблема — **training** conditioning:

- `original_size` должен быть размером исходника до crop, не `(1024, 1024)`
- `crops_coords_top_left` должен отражать center-crop offset, не `(0, 0)`
- Kohya с bucketing пишет оба значения автоматически per-image

Код: `trainer.py` → `_get_add_time_ids(original_size=(R,R), crops_coords=(0,0), ...)`.

## Отвергнутые

| Гипотеза | Доказательство |
|----------|----------------|
| Малый датасет | Bloom 80 img |
| Stale cache (latents/TE encode) | Bloom fresh cache |
| Dead LoRA / export | up_mean 0.13→0.54 |
| Eval без trigger | reForge с trigger плохо |
| Inference add_time_ids / AR mismatch (A) | 1024² = portrait по качеству/likeness |
| clip_skip | SDXL penultimate; Kohya не применяет |
| Init / FF targets / rank sampling fix | retrain — без улучшения |

## План (обновлён)

**P0 — один прогон с максимальным parity к Kohya:**

1. Bloom: rank=16, alpha=16, lr=1e-3, repeats=10, 10–15 epochs
2. `mixed_precision: bfloat16`, optimizer `adamw` (не 8bit)
3. Bucketing (или interim: хотя бы правильные add_time_ids H/I без bucketing)
4. Eval: reForge 832×1216, trigger + tags из caption
5. Success: likeness к ep.3–5

**Изоляция факторов (если P0 не помог):**

- Только G/B: Kohya LR+alpha без bucketing
- Только C: bucketing при старом lr
- Только H/I: fix add_time_ids при square crop

## Не приоритет сейчас

- TE cache invalidation by caption hash (disk cache: mtime image, не caption)
- Sampling config per project (UI eval)
- max_token_length 250 (пока не закрыт P0)
