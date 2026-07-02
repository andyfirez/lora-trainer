# Проблема

## Суть

Character LoRA из `lora-trainer` даёт **мыло без likeness**; Kohya_ss на тех же данных — заметно лучше.

**Base model:** `illustriousXL_v01.safetensors`

## Симптомы

- Деградация с ростом эпох (overcooking) — на job 13
- Нет likeness к персонажу из датасета — job 9/13
- С H/I fix + rank16/lr1e-3: **полный шум** (реgression, июнь 2026)
- С bucketing + fix collate + rank16/lr1e-3 **до fix N**: ep1 ok → ep2+ статик (jul 2026)
- С fix N + lr3e-4 constant: **ep1–3 хорошо → ep4+ шум** (jul 2026); Kohya без шума → M
- Одинаково плохо в **встроенном sampler** и **reForge** (не eval-only bug)

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

## Открыто

Root cause likeness **не доказан**. После lr3e-4 constant retrain + отклонения M (июль 2026):

1. **Порча/насыщение весов LoRA к ep4+ (O)** — **P0**; ep4+ шум и в reForge (Kohya-LoRA там стабильна) → проблема в весах checkpoint'ов. См. `11-revised-ep4-noise-jul2026.md`
2. **Train loop audit (D)** — **P0**; построчный diff vs Kohya sd-scripts
3. **Likeness** — ep2–3 приемлемы; полная стабильность ep4+ не достигнута
4. **LR/scheduler (G)** — частично подтверждена; недостаточна одна
5. **M (train/infer add_time_ids)** — **отклонена** (validation no-op); для high-res генерации inference conditioning `(H,W,0,0,H,W)` корректно

Fix N **подтверждён** для ранней деградации. Fix M отклонён. Следующий шаг — weight-stats анализ (O) + train loop audit (D).
