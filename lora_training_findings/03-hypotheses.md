# Гипотезы

## На проверку (приоритет)

| ID | Гипотеза | Как проверить |
|----|----------|---------------|
| **B** | `lora_alpha/rank = 0.5` ослабляет LoRA (Kohya: alpha=rank) | Retrain Bloom: rank=32, alpha=32 (или 16/16) |
| **C** | Нет bucketing → square crop портит identity | Реализовать bucket + retrain |
| **D** | Баг в train loop (target, latents, scheduler) | Diff vs Kohya sd-scripts: encode → noise → loss → backward |
| **E** | Overcooking (Chimera) | Смотреть epoch 1–3, не 20; снизить lr/repeats |
| **F** | max_token_length 77 vs Kohya 250 | Длинные caption обрезаются |

## Отвергнутые

| Гипотеза | Доказательство |
|----------|----------------|
| Малый датасет | Bloom 80 img |
| Stale cache | Bloom fresh cache |
| Dead LoRA / export | up_mean 0.13→0.54 |
| Eval без trigger | reForge с trigger плохо |
| add_time_ids mismatch (A) | 1024² = portrait по качеству/likeness |
| clip_skip | SDXL penultimate; Kohya не применяет |
| Init / FF targets / rank sampling fix | retrain — без улучшения |

## План после P0

1. Bloom, rank=16, alpha=16, bucketing, 10 epochs
2. Eval: reForge 832×1216, trigger + tags из caption
3. Success: likeness к ep.3–5

## Не приоритет сейчас

- TE cache invalidation by caption hash (future bug)
- Sampling config per project (UI eval)
- Per-image add_time_ids (корректность, не root cause после A)
