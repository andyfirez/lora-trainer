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
| [11-revised-ep4-noise-jul2026.md](11-revised-ep4-noise-jul2026.md) | Пересмотр: M и O отклонены; weight-stats vs Kohya |
| [12-train-scheduler-bug-jul2026.md](12-train-scheduler-bug-jul2026.md) | **ROOT CAUSE:** train forward process на EulerDiscreteScheduler |

**Статус (июль 2026): ROOT CAUSE найден (гипотеза D).** train loop зашумляет латенты через `EulerDiscreteScheduler` (из `pipeline.scheduler` single-file чекпойнта), а не через DDPM. Его `add_noise` не variance-preserving — вход в UNet раздут до ×14.6 на высоких timestep'ах → неверный градиент, накапливается по эпохам. Kohya использует DDPMScheduler → стабилен. **Fix:** собирать train `noise_scheduler` как `DDPMScheduler` (`model_loader.py`); требует retrain. Отклонены: M (inference add_time_ids, no-op), O (веса здоровы), предпосылка N (Kohya `full_fp16=true`). См. `12`.

**Артефакты:** `D:/SD/lora_output/Winx_Bloom_CFTS/`, `D:/SD/lora_output/Winx_Chimera_CFTS/`
