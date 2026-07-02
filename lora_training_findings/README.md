# LoRA Training Findings

Расследование качества character LoRA в `lora-trainer` vs Kohya_ss (SDXL, illustriousXL_v01). Июнь–июль 2026.

| Файл | Содержание |
|------|------------|
| [01-problem.md](01-problem.md) | Проблема, кейсы, что уже исключено |
| [02-tried-solutions.md](02-tried-solutions.md) | Опробованные fixes и эксперименты |
| [03-hypotheses.md](03-hypotheses.md) | Гипотезы: отвергнутые, открытые, приоритеты |
| [04-pipelines.md](04-pipelines.md) | Как работает lora-trainer и эталон (Kohya) |
| [05-bucketing-run-jul2026.md](05-bucketing-run-jul2026.md) | Bucketing + прогон Winx_Bloom после fix collate |
| [06-add-time-ids-collate-bug.md](06-add-time-ids-collate-bug.md) | Техническая записка: баг collate при batch_size>1 |
| [07-fp16-gradscaler-vs-kohya.md](07-fp16-gradscaler-vs-kohya.md) | fp16 LoRA weights без GradScaler vs Kohya (Accelerate) |
| [08-fix-n-post-run-jul2026.md](08-fix-n-post-run-jul2026.md) | Post-N retrain Winx_Bloom: ep1–3 ok, ep4+ статик |
| [09-lr-constant-post-run-jul2026.md](09-lr-constant-post-run-jul2026.md) | lr=3e-4 constant: лучше, но ep4+ шум; M → P0 |
| [10-fix-m-validation.md](10-fix-m-validation.md) | Fix M inference add_time_ids — код + валидация (итог: no-op) |
| [11-revised-ep4-noise-jul2026.md](11-revised-ep4-noise-jul2026.md) | Пересмотр: M отклонена; ep4+ шум → train loop/weights (O, D) |

**Статус (июль 2026):** Fix M **отклонён как root cause** — ручная валидация дала no-op (`11`). Ключевой довод: ep4+ шум виден и в reForge, где Kohya-LoRA стабильна → проблема в **весах checkpoint'ов** lora-trainer, не в inference. **Текущий P0:** weight-stats анализ (O) + train loop audit vs Kohya (D). Usable checkpoint: ep2–3 (lr3e-4 constant).

**Артефакты:** `D:/SD/lora_output/Winx_Bloom_CFTS/`, `D:/SD/lora_output/Winx_Chimera_CFTS/`
