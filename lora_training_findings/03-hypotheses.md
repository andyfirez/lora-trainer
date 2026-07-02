# Гипотезы

## На проверку (приоритет)

| ID | Гипотеза | Приоритет | Как проверить |
|----|----------|-----------|---------------|
| **O** | Порча/насыщение весов LoRA к ep4+ (объединяет E) | **P0** | Weight-stats ep1/ep3/ep4/ep10 lora-trainer vs Kohya (без GPU) |
| **D** | Баг в train loop (target, latents, scheduler, timesteps) | **P0** | Diff vs Kohya sd-scripts построчно |
| **R** | bucket_reso_steps 64 vs Kohya 256 | **P1** | Parity bucket'ов и per-image add_time_ids |
| **F** | max_token_length 77 vs Kohya 250 | **P2** | Длинные caption обрезаются молча |
| **M** | Train/infer add_time_ids mismatch | **Отклонена как root cause** (validation no-op). Косметический preview-fix. См. `10`, `11` |

**Пересмотр (июль 2026):** M была P0, но ручная валидация дала no-op (`11-revised-ep4-noise-jul2026.md`). Ключевой довод: ep4+ шум виден и в reForge, где Kohya-LoRA стабильна на тех же данных → root cause в **весах checkpoint'ов** (train loop), а не в inference conditioning. Fix M структурно не может это объяснить.

## Реализовано / закрыто

| ID | Гипотеза | Статус |
|----|----------|--------|
| **G** | LR/scheduler — overcooking на высоком lr | **Частично подтверждена** (lr3e-4 constant лучше; ep4+ шум остаётся → не root cause alone). См. `09` |
| **E** | Overcooking — best checkpoint ранняя эпоха | **Частично подтверждена** (Bloom post-N: ep2–3 best, ep4+ статик) |
| **C** | Нет bucketing → square crop портит identity | **Реализовано** (июль 2026). Likeness не закрыт; при lr1e-3 — поздний статик |
| **L** | Collate `add_time_ids` ломает conditioning при batch_size>1 | **Fix** (июль 2026). См. `06-add-time-ids-collate-bug.md` |
| **N** | LoRA weights fp16 + нет GradScaler ≠ Kohya fp32 weights + Accelerate scaler | **Частично подтверждена** (fix убрал ep2 деградацию; ep4+ остаётся). См. `07`, `08` |
| **B** | alpha/rank = 0.5 ослабляет LoRA | Закрыто для job 9/13 (alpha=16/rank=32). Bloom jul: alpha=rank |
| **J/K (как root cause jul run)** | fp16/AdamW8bit сами по себе | **Снято:** Kohya работает с тем же AdamW8bit; проблема в stack N, не в betas/wd |

### Детали N (fp16 stack vs Kohya)

**Документ:** `07-fp16-gradscaler-vs-kohya.md`

lora-trainer **до fix N** (июль 2026):

- UNet + LoRA adapters в **fp16** (`unet.weight_dtype=float16`, PEFT наследует dtype base layer)
- `loss.backward()` **без** `GradScaler`

**После fix N:** LoRA trainable weights в fp32 + GradScaler при `mixed_precision=float16` (Kohya default stack).

Kohya sd-scripts (default, без `--full_fp16`):

- LoRA trainable weights в **fp32**
- `mixed_precision=fp16` только для autocast forward
- `accelerator.backward(loss)` → **GradScaler** автоматически

AdamW8bit параметры (beta1=0.9, beta2=0.999, wd=0.01, eps=1e-8) **совпадают** — не причина.

**Симптом jul-прогона:** ep1 ok → ep2+ статик, loss 0.13–1.2 стабилен — согласуется с накоплением fp16-ошибки в весах при lr1e-3, а не с багом MSE.

**Fix (код, реализован июль 2026):** `mixed_precision.py` + trainer: GradScaler при fp16 + fp32 LoRA weights.

**Контрольный прогон (post-N):** ep1–3 ok, ep4+ статик при `lr=1e-3`, `warmup=0`. В шуме картинка почти не меняется (насыщение LoRA). См. `08-fix-n-post-run-jul2026.md`.

### Детали G/J/K (дивергенция при Kohya-parity LR)

**Jul-прогон до fix N** (`05-bucketing-run-jul2026.md`, после fix L):

- Config: rank=16, alpha=16, lr=1e-3, cosine, warmup=0, fp16, adamw_8bit, enable_bucket, batch_size=2
- Сэмплы: ep1 ok → **ep2** деградация → ep5/ep10 TV-статик
- Loss: 0.13–1.2, без роста

**Post-N retrain lr1e-3** (`08`): ep1–3 ok → ep4+ статик.

**Post-N retrain lr3e-4 constant** (`09-lr-constant-post-run-jul2026.md`):

- Качество **значительно лучше**, чем lr1e-3
- ep4+ **шум сохраняется**
- Kohya без шума при любых настройках → **M (train/infer add_time_ids)**, не G alone

**Вывод:** fix N + G (мягкий LR) **необходимы**, но **недостаточны**. **P0 → fix M** в `session.py`.

### Детали M (train/infer add_time_ids) — ОТКЛОНЕНА как root cause

**Train:** `buckets.py` — `(source_h, source_w, crop_y, crop_x, target_h, target_w)`, ненулевой crop для landscape.

**Infer/preview (до fix):** `latent_sampling/session.py` — `(H, W, 0, 0, H, W)`.

**Fix M (июль 2026):** `resolve_reference_add_time_ids` — median train `add_time_ids` для matching bucket; preview + standalone sampler.

**Валидация (job 25, ep1–10): no-op.** Сэмплинг `832×1216`, а датасет с `target_resolution=1024` не имеет такого bucket'а → match не найден → fallback на старый хардкод. Изображения идентичны до-fix.

**Почему M не root cause:**
1. Цель — генерировать **выше** train-разрешения; для high-res txt2img правильное conditioning `(H,W,0,0,H,W)`, а не train crop.
2. ep4+ шум виден и в **reForge**; там Kohya-LoRA стабильна на тех же данных при одинаковом inference pipeline → проблема в весах lora-trainer, не в inference.

Fix оставлен косметическим для in-app preview при сэмплинге ровно в train-bucket'е. Подробно: `11-revised-ep4-noise-jul2026.md`.

### Детали L (collate bug)

При `batch_size>1` tuple `add_time_ids` транспонировался collate → UNet получал смешанный conditioning. Fix: tensor в `__getitem__`. Обязателен при batch_size>1; не объясняет jul-прогон после fix (см. G/J/K).

### Детали H/I (add_time_ids fitted+crop, июнь 2026) — отвергнуто

Старая реализация: fitted size + `_crop_box`. Bloom + rank16/lr1e-3 → шум **с первых эпох**. Заменена Kohya bucketing semantics в `buckets.py` (июль).

### Детали G (effective LR, job 9/13)

| | alpha/rank | lr | effective |
|---|------------|-----|-----------|
| lora-trainer (job 9/13) | 16/32 = 0.5 | 1e-4 | **5e-5** |
| Kohya (Winx) | 16/16 = 1.0 | 1e-3 | **1e-3** |

Job 9/13: мыло без likeness (низкий LR). Bloom jul с lr1e-3: ep1 ok → статик (высокий LR + нестабильный stack).

## Отвергнутые

| Гипотеза | Доказательство |
|----------|----------------|
| Малый датасет | Bloom 80 img |
| Stale cache (latents/TE encode) | Bloom fresh cache |
| Dead LoRA / export | up_mean растёт, checkpoint'ы различаются |
| Eval без trigger | reForge с trigger плохо |
| Inference add_time_ids / AR mismatch (A) | 1024² = portrait по качеству/likeness |
| clip_skip | SDXL penultimate; Kohya не применяет |
| Init / FF targets / rank sampling fix | retrain — без улучшения |
| **Training add_time_ids H/I** (fitted + `_crop_box`) | Bloom июнь → шум с ep1 |
| **Bucketing alone fixes likeness** | Bloom jul → ep1 ok, ep2+ статик при lr1e-3 |
| **Collate bug alone explains jul run** | Fix применён до прогона; деградация ep2+ сохранилась |
| **AdamW8bit betas/wd как root cause** | Совпадают с Kohya; Kohya стабилен с тем же optimizer |
| **bf16 обязателен для стабильности** | Kohya stable на fp16; нужен fp32 weights + GradScaler (N) |

## План (обновлён, июль 2026 — после отклонения M)

**Завершено:**

1. ~~Fix N (GradScaler + fp32 LoRA)~~ **готово** (`07`, `08`)
2. ~~Retrain G (lr3e-4 constant)~~ **выполнено** — улучшение ep1–3, ep4+ шум остаётся (`09`)
3. ~~Fix M + ручная валидация~~ **выполнено — M отклонена (no-op)** (`10`, `11`)

**P0 (текущее):**

1. **Weight-stats анализ (O)** — нормы `lora_down`/`lora_up`, max abs, рост по эпохам: ep1/ep3/ep4/ep10 lora-trainer vs Kohya. Без GPU. См. `11`.
2. **Train loop audit (D)** — построчный diff `trainer.py` train step vs Kohya sd-scripts: target/prediction_type, timesteps sampling, latents scaling, clip_grad_norm, steps-per-epoch.
3. **Проверка цели:** ep2–3 checkpoint → reForge `832×1216` — подтвердить high-res генерацию с ранних checkpoint'ов.

**Usable checkpoint:** ep2–3 прогона lr3e-4 constant.

**P1:**

- bucket_reso_steps parity 64 vs 256 (R)
- max_token_length 250 (F)

**Не приоритет:**

- TE cache invalidation by caption hash
- Повторять H/I fitted+crop
- Train-side/ inference изменения add_time_ids (M снята)

## Не приоритет сейчас

- TE cache invalidation by caption hash (disk cache: mtime image, не caption)
- Sampling config per project (UI eval)
- max_token_length 250 (пока не закрыт P0)
