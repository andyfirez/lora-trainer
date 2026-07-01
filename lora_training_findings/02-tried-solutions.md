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
| **add_time_ids H/I** (fitted + crop, per-image) | `preprocess.py`, `dataset.py`, `trainer.py`, `runner.py` | **регрессия: шум** (Bloom + rank16/lr1e-3) |

## Эксперименты пользователя

| Эксперимент | Результат |
|-------------|-----------|
| min_snr=0, noise_offset=0 | плохо |
| rank 16↔32, repeats, epochs | плохо |
| Retrain после Kohya init | плохо |
| reForge + trigger | плохо |
| **Hypothesis A:** inference 1024² vs 832×1216 | оба без likeness → **отвергнуто** |
| **Bloom + rank16/alpha16/lr1e-3 + H/I fix** | **шум** (хуже «мыла» job 13); G/B не изолирован |

Hypothesis A: `Winx_Bloom_CFTS_epoch5`, seed=42, prompt с trigger → `lora_output/Winx_Bloom_CFTS/hypothesis_a/`. Скрипт: `scripts/hypothesis_a_resolution_test.py`.

H/I fix: per-image `time_ids` из fit-short-side bake (`compute_add_time_ids_for_bake`). Прогон Bloom с Kohya LR/alpha — eval reForge/sampler → изображения деградируют в шум.

## Что не делали

- Bucketing
- **G/B изолированно:** rank=16, alpha=16, lr=1e-3 **без** H/I fix (единственный прогон был с H/I)
- mixed_precision bfloat16 (гипотеза J)
- optimizer adamw fp32 вместо adamw_8bit (гипотеза K)
- max_token_length > 77 (гипотеза F)
- Kohya-семантика add_time_ids (`source size` + `get_crop_ltrb`) с aligned inference
- Аудит train loop (noise target, VAE encode, timestep sampling) vs sd-scripts построчно
