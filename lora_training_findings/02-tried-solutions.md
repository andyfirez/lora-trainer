# Опробованные решения

## Изменения в коде

| Fix | Файлы | Эффект на likeness |
|-----|-------|---------------------|
| FF + attention LoRA targets | `lora_targets.py` | нет |
| Min-SNR, noise offset | `loss.py`, `config.py` | нет (в Winx прогонах были 0) |
| Kohya init (Kaiming A, zero B) | `lora_peft.py` | нет |
| Rank mismatch при sampling | `lora_export.py`, `sampler/sdxl/service.py` | только inference bug |
| clip_skip config | `prompt_encoding.py`, `te_cache.py` | нет (default=2 = старый `hidden_states[-2]`) |
| Defaults (rank=32, alpha=32, lr=5e-5…) | `config.py` | не подтверждено |
| **add_time_ids H/I** (fitted + crop, per-image, июнь) | `preprocess.py`, `dataset.py`, `trainer.py` | **регрессия: шум** (Bloom + rank16/lr1e-3) |
| **Aspect-ratio bucketing** (offline bake, Kohya semantics, июль) | `buckets.py`, preprocess, DB, trainer | ep1 ok, ep2+ статик при lr1e-3 (до fix N) |
| **add_time_ids collate fix** (batch_size>1) | `dataset.py`, `trainer.py` | fix корректности conditioning |
| **Fix N: fp32 LoRA + GradScaler** | `mixed_precision.py`, `trainer.py`, `checkpoint_state.py` | ep2–3 ok; ep4+ при lr1e-3; lr3e-4 constant — лучше, ep4+ шум остаётся (M) |
| **Диагностика fp16 stack (N)** | code review jul 2026 | выявлено: fp16 LoRA + no GradScaler vs Kohya |

Детали N: `07-fp16-gradscaler-vs-kohya.md`. Post-run N: `08`. lr3e-4 constant: `09-lr-constant-post-run-jul2026.md`. Bucketing: `05-bucketing-run-jul2026.md`.

## Эксперименты пользователя

| Эксперимент | Результат |
|-------------|-----------|
| min_snr=0, noise_offset=0 | плохо |
| rank 16↔32, repeats, epochs | плохо |
| Retrain после Kohya init | плохо |
| reForge + trigger | плохо |
| **Hypothesis A:** inference 1024² vs 832×1216 | оба без likeness → **отвергнуто** |
| **Bloom + rank16/alpha16/lr1e-3 + H/I fix** (июнь) | **шум** с первых эпох |
| **Bloom + bucketing + fix collate + rank16/lr1e-3** (июль, до fix N) | **ep1 ok → ep2+ статик**, loss стабилен |
| **Bloom + fix N + rank16/lr1e-3, warmup=0** (jul) | **ep1–3 ok → ep4+ статик** |
| **Bloom + fix N + lr3e-4, constant scheduler** (jul) | **значительно лучше ep1–3; ep4+ шум**; Kohya без шума при любых настройках → M |

Post-run: `08`, `09-lr-constant-post-run-jul2026.md`.

## Что не делали

- max_token_length > 77 (гипотеза F)
- **Fix M:** aligned inference `add_time_ids` (train crop coords → sampler/reForge) — **P0, следующий шаг**
- Аудит train loop vs sd-scripts построчно (D)

## Что сделали (июль 2026)

- Offline bucketing: non-square prepared images, `BucketBatchSampler`, per-image Kohya `add_time_ids`
- Fix collate `add_time_ids` для `batch_size>1`
- **Fix N:** fp32 LoRA weights + GradScaler при fp16; resume scaler state
- Unit/integration tests: `test_buckets.py`, `test_bucket_batch_sampler.py`, `test_mixed_precision.py`, collate regression test
