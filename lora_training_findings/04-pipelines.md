# Пайплайны: lora-trainer vs Kohya

## lora-trainer (SDXL)

```
Dataset (.jpg + .txt)
  → preprocess: fit short side → center crop → square bake → .prepared/{resolution}/
  → latent cache: VAE encode → *_sdxl.npz
  → TE cache: CLIP-L/G penultimate → *_te.npz  (clip_skip=2 ≡ hidden_states[-2], max 77 tokens)
  → train loop (trainer.py):
      latents + noisy_latents
      add_time_ids = [R,R, 0,0, R,R]   # BUG: original_size=crop size, crop offset always 0
      prompt_embeds из cache или live encode
      UNet forward + MSE loss (optional min_snr, noise_offset)
      LoRA on UNet (attention + FF), TE frozen
      effective step ≈ (alpha/rank) × lr × grad
  → export .safetensors (Kohya-compatible metadata)
  → sampling: latent k-sample + VAE decode (832×1216 default in configs)
```

**Ключевые файлы:** `preprocess.py`, `latent_cache.py`, `te_cache.py`, `trainer.py`, `loss.py`, `latent_sampling/session.py`

**Ограничения:** нет bucketing; add_time_ids не per-image; TE cache valid по mtime image (не caption); UI sampling configs id=5/7 без trigger персонажа.

## Kohya sd-scripts (эталон Winx)

**Config:** `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/config_*.json`

```
Dataset (folder 10_* = 10 repeats)
  → bucketing: сохраняет AR, bucket_reso_steps=256
  → cache latents + TE (sdxl_cache_text_encoder_outputs)
  → train: MSE l2, enable_bucket, max_token_length=250
  → add_time_ids: per-image original_size + crop coords из bucket
  → clip_skip в UI — для SDXL train игнорируется (penultimate)
```

## Расхождения (значимые)

| | Kohya (Winx) | lora-trainer (job_9/13) |
|---|--------------|-------------------------|
| enable_bucket | ✅ | ❌ |
| add_time_ids | per-image (source size + crop) | фикс. `[R,R,0,0,R,R]` ❌ |
| effective LR `(α/rank)×lr` | 1.0 × 1e-3 = **1e-3** | 0.5 × 1e-4 = **5e-5** ❌ |
| max_token_length | 250 | 77 ❌ |
| rank / alpha | 16 / 16 (scale 1.0) | 32 / 16 (scale 0.5) ❌ |
| unet_lr | 1e-3 | 1e-4 ❌ |
| mixed_precision | bf16 (typical SDXL) | fp16 ❌ |
| optimizer | AdamW fp32 | AdamW8bit ❌ |
| epochs | 40 | 20 |
| repeats | 10 (folder) | 5 / 2 |
| loss | l2 | MSE ✅ |
| min_snr / noise_offset | 0 / 0 | 0 / 0 ✅ |
| LoRA targets | attn + FF | attn + FF ✅ |
| LoRA init | Kaiming A, zero B | ✅ |
| TE encoding SDXL | penultimate | penultimate ✅ |
| sample AR | 832×1216 | 832×1216 ✅ |

## add_time_ids: что не так в lora-trainer

Landscape source ~1216×918, bake 1024×1024 center crop:

| Поле | Должно быть | Сейчас |
|------|-------------|--------|
| original_size | ~(1216, 918) | (1024, 1024) |
| crops_coords_top_left | ~(96, 0) | (0, 0) |
| target_size | bucket/crop size | (1024, 1024) ✅ |

SDXL использует original_size как quality/resolution conditioning. Неверные coords дают ложную spatial info.

Hypothesis A проверяла **inference** resolution — отвергнута. H/I — про **training** conditioning (другой слой проблемы).

## OneTrainer

Тот же SDXL паттерн: TE `default_layer=-2`, bucketing в data loader, penultimate для обоих encoders. Детальный diff не делался — эталон для пользователя Kohya.

## Kohya reference run (работает)

- rank=16, alpha=16, unet_lr=1e-3, 40 ep, enable_bucket
- sample_prompt с trigger `Winx_Chimera_CFTS`
- Output: `LoRA_training/Winx_Chimera_CFTS/illustriousXL_v01/v1/`
