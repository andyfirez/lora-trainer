# Проблема

## Суть

Character LoRA из `lora-trainer` даёт **мыло без likeness**; Kohya_ss на тех же данных — заметно лучше.

**Base model:** `illustriousXL_v01.safetensors`

## Симптомы

- Деградация с ростом эпох (overcooking)
- Нет likeness к персонажу из датасета
- Одинаково плохо в **встроенном sampler** и **reForge** (не eval-only bug)

## Кейсы

| LoRA | Dataset | img | Job | rank/alpha | repeats | steps | loss trend | результат |
|------|---------|-----|-----|------------|---------|-------|------------|-----------|
| Winx_Chimera_CFTS | `datasets/Winx_Chimera` | 13 | 9 | 32/16 | 5 | 660 | ↑ ep.2–4 | плохо |
| Winx_Bloom_CFTS | `datasets/Winx_Bloom` | 80 | 13 | 32/16 | 2 | 1600 | ↓ к ep.20 | **тот же** |

Bloom — контрольный: другой персонаж, 6× данных, свежий cache (80 encoded, 0 loaded). Результат не улучшился.

## Данные по кейсам

**Общее:** все bake в **1024×1024** без bucketing (`preprocess.py`). Исходники преимущественно landscape (~1216×918).

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
| add_time_ids / AR mismatch | Hypothesis A: 1024² и 832×1216 — оба без likeness |
| clip_skip gap | SDXL всегда penultimate; Kohya игнорирует clip_skip при train |
| loss_type l2 vs MSE | Эквивалентны |
| MSE + min_snr=0 | Kohya тоже 0 в Winx config |

## Открыто

Root cause **не доказан**. Наиболее вероятные направления: **bucketing**, **alpha/rank scale**, **train loop vs Kohya** (см. `03-hypotheses.md`).
