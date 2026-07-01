# Пайплайны: lora-trainer vs Kohya

## lora-trainer (SDXL) — июль 2026

```
Dataset (.jpg + .txt)
  → preprocess: assign_bucket → scale+crop → non-square bake → .prepared/{resolution}/
  → DB: bucket_width/height, scale_to_*, crop_x/y per image
  → latent cache: VAE encode → *_sdxl.npz (any size ÷8)
  → TE cache: CLIP-L/G penultimate → *_te.npz  (clip_skip=2, max 77 tokens)
  → train loop (trainer.py):
      UNet loaded fp16; LoRA lora_A/lora_B inherit fp16  ⚠️ vs Kohya fp32
      BucketBatchSampler if enable_bucket (same WxH per batch)
      latents + noisy_latents
      add_time_ids = per-image Kohya tuple → torch.tensor in __getitem__
      autocast(fp16) forward; MSE in fp32
      loss.backward() без GradScaler  ⚠️ vs Kohya accelerator.backward + scaler
      clip_grad_norm_(1.0); AdamW8bit step
  → export .safetensors (Kohya-compatible metadata)
  → sampling: latent k-sample + VAE decode
      add_time_ids inference: (H, W, 0, 0, H, W)  ⚠️ mismatch с train при bucketing
```

**Ключевые файлы:** `buckets.py`, `preprocess.py`, `concept_training_metadata.py`, `bucket_batch_sampler.py`, `dataset.py`, `latent_cache.py`, `te_cache.py`, `trainer.py`, `loss.py`, `latent_sampling/session.py`

**Ограничения:** fp16 LoRA weights + no GradScaler (гипотеза N, см. `07-fp16-gradscaler-vs-kohya.md`); inference add_time_ids mismatch (M); max_token_length=77.

## Kohya sd-scripts (эталон Winx)

**Config:** `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/config_*.json`

```
Dataset (folder 10_* = 10 repeats)
  → bucketing: сохраняет AR, bucket_reso_steps=256
  → cache latents + TE (sdxl_cache_text_encoder_outputs)
  → train: MSE l2, enable_bucket, max_token_length=250
  → LoRA network weights fp32 (default; --full_fp16 опционально)
  → mixed_precision fp16 → autocast forward only
  → accelerator.backward(loss) → GradScaler при fp16
  → add_time_ids: per-image original_size + crop coords из bucket
```

## Расхождения (значимые)

| | Kohya (Winx) | lora-trainer (job_9/13) | lora-trainer (jul 2026 Bloom) |
|---|--------------|-------------------------|-------------------------------|
| enable_bucket | ✅ | ❌ | ✅ |
| add_time_ids (train) | per-image Kohya | `[R,R,0,0,R,R]` | per-image Kohya ✅ |
| add_time_ids (infer) | aligned in pipeline | `(H,W,0,0,H,W)` | `(H,W,0,0,H,W)` ⚠️ |
| effective LR `(α/rank)×lr` | 1.0 × 1e-3 | 0.5 × 1e-4 | 1.0 × 1e-3 |
| lr_warmup | есть в типичных cfg | 0 | 0 |
| max_token_length | 250 | 77 | 77 |
| mixed_precision | fp16 + autocast | fp16 | fp16 |
| LoRA weight dtype | fp32 (default) | fp16 | fp16 ❌ |
| backward | GradScaler (Accelerate) | no scaler | no scaler ❌ |
| optimizer | AdamW8bit (типично) | AdamW8bit | AdamW8bit ✅ same betas/wd |
| AdamW betas/wd | 0.9/0.999, 0.01 | same | same |
| batch collate time_ids | корректный | N/A (bs=1?) | fix L ✅ |
| результат | likeness | мыло | ep1 ok → ep2+ статик |

## add_time_ids

### Baseline (job 9/13)

Фикс. `[R,R,0,0,R,R]` — неверно для landscape, но единообразно с inference `(H,W,0,0,H,W)`. Результат: мыло, без likeness.

### H/I fix fitted+crop (отвергнуто, июнь 2026)

Fitted size + `_crop_box`. Bloom + rank16/lr1e-3 → шум с ep1. **Не повторять.**

### Bucketing Kohya semantics (июль 2026)

`buckets.py`: source size + crop после scale-to-cover. Пример landscape 1216×918 → bucket 768×1024: `(918, 1216, 3, 0, 768, 1024)`.

Bloom retrain: ep1 норм, ep2+ статик при lr1e-3/fp16/adamw8bit. Collate fix L применён. Root cause stack: **N** (см. `07-fp16-gradscaler-vs-kohya.md`).

### fp16 stack N (июль 2026, не fix)

LoRA weights fp16 + `loss.backward()` без GradScaler. Kohya при том же fp16+AdamW8bit держит fp32 weights и использует Accelerate GradScaler. AdamW hyperparams совпадают.

### Collate bug L (fix, июль 2026)

Tuple в batch → transpose → mixed conditioning. Tensor в dataset → `(B,6)`. См. `06-add-time-ids-collate-bug.md`.

### Inference (latent sampling / reForge)

`session.py`: `(height, width, 0, 0, height, width)`. При bucketing train использует ненулевые crop — **mismatch** (гипотеза M). Ep1 jul-прогона была ok → mismatch не единственная причина позднего шума.

## OneTrainer

Тот же SDXL паттерн: TE `default_layer=-2`, bucketing в data loader, penultimate для обоих encoders. Детальный diff не делался — эталон для пользователя Kohya.

## Kohya reference run (работает)

- rank=16, alpha=16, unet_lr=1e-3, 40 ep, enable_bucket
- sample_prompt с trigger `Winx_Chimera_CFTS`
- Output: `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/`

**Примечание:** Kohya fp16+AdamW8bit стабилен за счёт fp32 LoRA weights + GradScaler. lora-trainer jul 2026 эквивалентен Kohya `--full_fp16` **без** GradScaler — см. `07-fp16-gradscaler-vs-kohya.md`.
