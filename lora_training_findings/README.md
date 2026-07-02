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

**Статус (июль 2026):** Fix N + понижение LR (3e-4, constant) **существенно улучшили** качество ep1–3. Шум с ep4 **сохраняется** — LR alone не объясняет. **P0:** гипотеза **M** (train/infer `add_time_ids` mismatch). Kohya стабилен при любых LR — структурное расхождение пайплайна. Usable checkpoint: **ep2–3**. См. `09-lr-constant-post-run-jul2026.md`.

**Артефакты:** `D:/SD/lora_output/Winx_Bloom_CFTS/`, `D:/SD/lora_output/Winx_Chimera_CFTS/`
