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

## Эксперименты пользователя

| Эксперимент | Результат |
|-------------|-----------|
| min_snr=0, noise_offset=0 | плохо |
| rank 16↔32, repeats, epochs | плохо |
| Retrain после Kohya init | плохо |
| reForge + trigger | плохо |
| **Hypothesis A:** inference 1024² vs 832×1216 | оба без likeness → **отвергнуто** |

Hypothesis A: `Winx_Bloom_CFTS_epoch5`, seed=42, prompt с trigger → `lora_output/Winx_Bloom_CFTS/hypothesis_a/`. Скрипт: `scripts/hypothesis_a_resolution_test.py`.

## Что не делали

- Bucketing
- alpha = rank (32/32 или 16/16) + lr=1e-3 (Kohya parity, гипотеза G)
- Fix add_time_ids: per-image original_size + crop coords (гипотезы H/I)
- mixed_precision bfloat16 (гипотеза J)
- optimizer adamw fp32 вместо adamw_8bit (гипотеза K)
- max_token_length > 77 (гипотеза F)
- Аудит train loop (noise target, VAE encode, timestep sampling) vs sd-scripts построчно
