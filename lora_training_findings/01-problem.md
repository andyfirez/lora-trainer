# Проблема

## Суть

Character LoRA из `lora-trainer` даёт **мыло без likeness**; Kohya_ss на тех же данных — заметно лучше.

**Base model:** `illustriousXL_v01.safetensors`

## Симптомы

- Деградация с ростом эпох (overcooking) — на job 13
- Нет likeness к персонажу из датасета — job 9/13
- С H/I fix + rank16/lr1e-3: **полный шум** (реgression, июнь 2026)
- С bucketing + fix collate + rank16/lr1e-3 **до fix N**: ep1 ok → ep2+ статик (jul 2026)
- С fix N + lr3e-4 constant: **ep1–3 хорошо → ep4+ шум** (jul 2026); Kohya без шума → **root cause D** (train scheduler, `12`)
- Одинаково плохо в **встроенном sampler** и **reForge** (не eval-only bug) → испорчены веса, не inference

## Кейсы

| LoRA | Dataset | img | Job | rank/alpha | repeats | steps | loss trend | результат |
|------|---------|-----|-----|------------|---------|-------|------------|-----------|
| Winx_Chimera_CFTS | `datasets/Winx_Chimera` | 13 | 9 | 32/16 | 5 | 660 | ↑ ep.2–4 | плохо |
| Winx_Bloom_CFTS | `datasets/Winx_Bloom` | 80 | 13 | 32/16 | 2 | 1600 | ↓ к ep.20 | **тот же** (мыло) |
| Winx_Bloom_CFTS | `datasets/Winx_Bloom` | 80 | — | 16/16 | 2 | 800 | стабилен 0.13–1.2 | **шум** (rank16, lr1e-3, + H/I fix, июнь) |
| Winx_Bloom_CFTS | `datasets/Winx_Bloom` | 80 | — | 16/16 | 2 | 800 | стабилен | **ep1 ok → ep2+ статик** (bucketing + fix collate, jul, **до fix N**) |
| Winx_Bloom_CFTS | `datasets/Winx_Bloom` | 80 | — | 16/16 | 2 | 800 | стабилен | **ep1–3 ok → ep4+ статик** (fix N, lr1e-3, jul) |
| Winx_Bloom_CFTS | `datasets/Winx_Bloom` | 80 | — | 16/16 | 2 | 800 | стабилен | **ep1–3 лучше → ep4+ шум** (fix N, lr3e-4 constant, jul) |

Bloom — контрольный. Fix N + мягкий LR улучшили ep1–3; ep4+ шум → **M (train/infer add_time_ids)**, не только G.

## Данные по кейсам

**Общее:** job 9/13 — square bake 1024×1024 без bucketing. Июль 2026 — offline bucketing (non-square prepared). Исходники преимущественно landscape (~1216×918).

| | Chimera | Bloom |
|---|---------|-------|
| Non-square sources | 13/13 | 76/80 |
| Шумные caption | да (13 img) | менее критично |
| LoRA weights ep1→20 | растут | up_mean 0.13→0.54 |

**Chimera-only:** sampling config id=5 без trigger (Melanie prompt) — ломает UI eval, но reForge с trigger тоже плохо.

## Исключено

| Гипотеза | Почему |
|----------|--------|
| Мало картинок | Bloom 80 img — тот же результат |
| Stale cache | Bloom: fresh encode |
| LoRA не учится | Веса растут, checkpoint'ы различаются |
| Только eval/trigger UI | reForge с trigger — плохо |
| add_time_ids / AR mismatch (inference) | Hypothesis A: 1024² и 832×1216 — оба без likeness |
| Training add_time_ids H/I (fitted+crop) | Bloom + rank16/lr1e-3 → шум, хуже baseline |
| clip_skip gap | SDXL всегда penultimate; Kohya игнорирует clip_skip при train |
| loss_type l2 vs MSE | Эквивалентны |
| MSE + min_snr=0 | Kohya тоже 0 в Winx config |

## ROOT CAUSE найден (июль 2026)

**Train loop зашумляет латенты через `EulerDiscreteScheduler` вместо DDPM** (гипотеза D). См. `12-train-scheduler-bug-jul2026.md`.

`model_loader.py` берёт `noise_scheduler = pipeline.scheduler` из single-file чекпойнта → `EulerDiscreteScheduler`. Его `add_noise` = `x + σ·noise` (не variance-preserving) → вход в UNet раздут до ×14.6 на высоких timestep'ах → систематически неверный градиент, накапливается по эпохам (ep4+ шум). Kohya использует DDPMScheduler → стабилен при любых LR.

**Fix:** собирать train `noise_scheduler` как `DDPMScheduler` из бет pipeline; sampling не трогать. Требует retrain для верификации.

**Отклонено по пути:** M (inference add_time_ids — no-op), O (веса здоровы, растут плавно), предпосылка N (Kohya обучал с `full_fp16=true`, fp16-веса — стабильно). G (LR) объясняет лишь частичное улучшение (замедление накопления ошибки), не root cause.
