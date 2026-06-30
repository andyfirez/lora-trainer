# Summary

## Цель расследования

Выяснить, почему LoRA, обученная в `lora-trainer`, даёт «мыльные» результаты без likeness по сравнению с Kohya_ss и OneTrainer (SDXL, illustriousXL_v01).

## Кейсы

| LoRA | Dataset | Images | Job | Result |
|------|---------|--------|-----|--------|
| `Winx_Chimera_CFTS` | `D:/SD/datasets/Winx_Chimera` | 13 | job_9 | Плохо |
| **`Winx_Bloom_CFTS`** | **`D:/SD/datasets/Winx_Bloom`** | **80** | **job_13** | **Тот же плохий результат** |

Bloom — **контрольный эксперимент**: другой персонаж, в 6× больше данных, свежий cache, мягче repeats. Результат не улучшился.

## Симптомы

1. Изображения **мыльные**, likeness **отсутствует**.
2. Проблема в **reForge** и во встроенном sampler → не только eval UI.
3. **Повторяется на разных датасетах** → systemic pipeline bug, не «плохой датасет Chimera».

## Что подтверждено / отвергнуто

| Гипотеза | Статус |
|----------|--------|
| Только маленький датасet (13 img) | **Отвергнуто** — Bloom 80 img, тот же результат |
| Stale TE/latent cache | **Отвергнуто для Bloom** (80 encoded, 0 loaded); для Chimera маловероятно (датасет не менялся) |
| LoRA не обучается (dead weights) | **Отвергнуто** — веса растут epoch1→20 (up_mean 0.13→0.54) |
| Trigger только в sampling UI | **Не root cause** — reForge с trigger тоже плохо |
| Rank mismatch при sampling | **Исправлено** |
| Узкий LoRA scope | **Исправлено** (FF layers) |
| Kohya init | **Исправлено**, качество не изменилось |
| **add_time_ids train ≠ inference** | **Отвергнуто как root cause** (hypothesis A: 1024² и 832×1216 — оба без likeness) |
| **Нет bucketing** (76/80 Bloom non-square) | **Подтверждено**, affects all runs |
| **lora_alpha/rank = 0.5** (Kohya uses 1.0) | **Contributing factor** |
| clip_skip (Kohya config: 2) | **Отвергнуто для SDXL** — Kohya игнорирует; default=2 = старый `hidden_states[-2]` |
| min_snr/noise_offset = 0 | В проблемных прогонах отключено |
| Overcooking только на Chimera | **Недостаточно** — Bloom loss падает, результат всё равно плохой |

## Хронология

1. Диагностика vs Kohya → targets, loss, defaults.
2. Fix rank mismatch, Kohya init.
3. Winx_Chimera — качество не улучшилось.
4. Анализ cache, bucketing, hyperparams, caption noise.
5. **Winx_Bloom (job_13)** — другой датасет, тот же результат → фокус на systemic pipeline.

## Текущий статус

**Root cause не доказан на 100%**, но Bloom сужает круг:

- Это **не** «мало картинок» и **не** «LoRA не учится».
- Это **systemic pipeline gap**, но **не add_time_ids** (hypothesis A отвергнута).
- Следующий приоритет: **bucketing + alpha=rank**, затем глубже копать train loop vs Kohya.
- clip_skip добавлен в код (default=2), но **не меняет SDXL-пайплайн** — см. §16 в `03-findings.md`.
