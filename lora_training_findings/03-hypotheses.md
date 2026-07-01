# Гипотезы

## На проверку (приоритет)

| ID | Гипотеза | Приоритет | Как проверить |
|----|----------|-----------|---------------|
| **G** | Effective LR в ~20× ниже Kohya: `(alpha/rank) × lr` = 0.5×1e-4 vs 1.0×1e-3 | **P0** | Retrain Bloom: rank=16, alpha=16, lr=1e-3 **без** H/I fix |
| **B** | `lora_alpha/rank = 0.5` ослабляет LoRA (часть G) | **P0** | Вместе с G: alpha=rank |
| **C** | Нет bucketing → square crop портит identity | **P0** | Реализовать bucket + retrain |
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

### Детали H/I (add_time_ids) — отвергнуто

Реализован fix: per-image `time_ids` из `preprocess.py` (fitted size + `_crop_box` offset). Bloom retrain с **rank=16, alpha=16, lr=1e-3** (G/B parity) + H/I fix → **регрессия: чистый шум** (хуже «мыла» job 13).

Вероятные причины провала:

- Семантика **fitted+crop** ≠ Kohya (`source size` + virtual `get_crop_ltrb`) — conditioning вне распределения SDXL
- Train/infer mismatch усилился: train с crop coords, sampler/reForge всё ещё `(H,W,0,0,H,W)`
- До fix: неверно, но **единообразно** `(R,R,0,0,R,R)` на train и eval

**Вывод:** revert H/I fix перед следующими прогонами; если retry — только Kohya-семантика + выравнивание inference.

### G/B — частично проверено (с H/I, не изолированно)

Прогон Bloom с rank=16/16, lr=1e-3 дал шум **вместе с H/I fix**. Изолированный тест G/B (lr parity без H/I) **ещё не делали**.

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
| **Training add_time_ids H/I** (fitted + `_crop_box`) | Bloom + rank16/alpha16/lr1e-3 → **шум**, хуже job 13 |
| **G/B изолированно** | Не проверено: единственный прогон rank16/lr1e-3 был с H/I fix |

## План (обновлён)

**P0 — один прогон с максимальным parity к Kohya:**

1. **Revert H/I fix** в коде (вернуть фикс. `[R,R,0,0,R,R]` на train)
2. Bloom: rank=16, alpha=16, lr=1e-3, repeats=10, 10–15 epochs
3. `mixed_precision: bfloat16`, optimizer `adamw` (не 8bit)
4. Bucketing (гипотеза C)
5. Eval: reForge 832×1216, trigger + tags из caption
6. Success: likeness к ep.3–5

**Изоляция факторов:**

- **Следующий:** только G/B (+ J/K/bf16) без H/I и без bucketing
- Только C: bucketing при Kohya lr
- H/I: **не повторять** fitted+crop; только после bucketing + Kohya `get_crop_ltrb` + aligned inference

## Не приоритет сейчас

- TE cache invalidation by caption hash (disk cache: mtime image, не caption)
- Sampling config per project (UI eval)
- max_token_length 250 (пока не закрыт P0)
