# Гипотезы

## На проверку (приоритет)

| ID | Гипотеза | Приоритет | Как проверить |
|----|----------|-----------|---------------|
| **M** | Train/infer add_time_ids mismatch (train crop coords, infer `(H,W,0,0,H,W)`) | **P0** | Fix `latent_sampling/session.py`; retrain lr3e-4 constant |
| **D** | Баг в train loop (target, latents, scheduler) | **P1** | Diff vs Kohya sd-scripts построчно |
| **F** | max_token_length 77 vs Kohya 250 | **P2** | Длинные caption обрезаются молча |

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

### Детали M (train/infer add_time_ids)

**Train:** `buckets.py` — `(source_h, source_w, crop_y, crop_x, target_h, target_w)`, ненулевой crop для landscape.

**Infer/preview:** `latent_sampling/session.py` — `(H, W, 0, 0, H, W)`.

LoRA на ep3–4+ «заточена» под train conditioning; inference OOD → шум. Kohya согласует train/generation — стабилен независимо от LR.

См. `09-lr-constant-post-run-jul2026.md`, `04-pipelines.md`.

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

## План (обновлён, июль 2026)

**P0 — fix M (train/infer add_time_ids):**

1. ~~Fix N (GradScaler + fp32 LoRA)~~ **готово**
2. ~~Retrain G (lr3e-4 constant)~~ **выполнено** — улучшение ep1–3, ep4+ шум остаётся (`09`)
3. **Код:** выровнять `latent_sampling/session.py` с train bucket semantics
4. Retrain Bloom: lr3e-4 constant (или Kohya-parity после M)
5. Success: ep4–10 без шума; стабильность как в Kohya

**Usable checkpoint:** ep2–3 прогона lr3e-4 constant.

**P1:**

- Train loop audit vs Kohya (D)
- max_token_length 250 (F)

**Не приоритет:**

- TE cache invalidation by caption hash
- Повторять H/I fitted+crop

## Не приоритет сейчас

- TE cache invalidation by caption hash (disk cache: mtime image, не caption)
- Sampling config per project (UI eval)
- max_token_length 250 (пока не закрыт P0)
