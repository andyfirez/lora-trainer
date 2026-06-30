# LoRA Training Findings

Расследование качества character LoRA в `lora-trainer` vs Kohya_ss (SDXL, illustriousXL_v01). Июнь 2026.

| Файл | Содержание |
|------|------------|
| [01-problem.md](01-problem.md) | Проблема, кейсы, что уже исключено |
| [02-tried-solutions.md](02-tried-solutions.md) | Опробованные fixes и эксперименты |
| [03-hypotheses.md](03-hypotheses.md) | Гипотезы: отвергнутые, открытые, приоритеты |
| [04-pipelines.md](04-pipelines.md) | Как работает lora-trainer и эталон (Kohya) |

**Статус:** root cause не доказан. LoRA обучается (веса растут), likeness не появляется — bug в train/preprocess pipeline, не в sampling UI. Code review выявил: effective LR ~20× ниже Kohya, неверные add_time_ids, нет bucketing (см. `03-hypotheses.md`).

**Артефакты:** `D:/SD/lora_output/Winx_Bloom_CFTS/`, `D:/SD/lora_output/Winx_Chimera_CFTS/`
