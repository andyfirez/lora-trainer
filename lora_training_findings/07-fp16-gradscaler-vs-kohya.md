# fp16 LoRA weights + отсутствие GradScaler vs Kohya (июль 2026)

Техническая записка: почему `mixed_precision: float16` + `adamw_8bit` в lora-trainer **не эквивалентны** тому же стеку в Kohya sd-scripts, и как это связано с деградацией ep1→ep2+ статик при `lr=1e-3`.

## Симптом (Winx_Bloom_CFTS, июль 2026)

- Config: `mixed_precision: float16`, `optimizer: adamw_8bit`, `learning_rate: 1e-3`, `lr_warmup_steps: 0`, bucketing on, fix collate on
- Сэмплы: ep1 — нормальная картинка; ep2+ — деградация → TV-статик
- `loss/loss` в UI: 0.13–1.2 на всех шагах, **без роста и без NaN**

Пользователь подтвердил: Kohya_ss на том же GPU с fp16 + AdamW8bit работал стабильно. **Параметры оптимизатора (beta1/beta2/eps/weight_decay) совпадают с Kohya** — проблема не в них.

## Параметры AdamW8bit в lora-trainer

`build_optimizer` (`src/trainer/optimizer_config.py`):

```python
return AdamW8bit(params, lr=lr, betas=betas, weight_decay=opt.weight_decay)
# betas = (0.9, 0.999), weight_decay = 0.01, eps = 1e-8 (default bitsandbytes)
```

Presets (`frontend/src/lib/optimizer_presets.json`): те же beta1/beta2/weight_decay, что типично для Kohya. **Гиперпараметры AdamW здесь не подозрительны.**

## Расхождение #1: dtype обучаемых весов LoRA

### lora-trainer (до fix N)

1. UNet загружается в fp16:

```python
# trainer.py — load_sdxl_components(unet_dtype=config.unet.weight_dtype)
# TrainConfig default: unet.weight_dtype = float16
```

2. PEFT оборачивает UNet; `lora_A` / `lora_B` наследуют dtype базового слоя → fp16.

3. Optimizer получал **fp16 trainable params** (LoRA adapters).

### lora-trainer (после fix N, июль 2026)

После `get_peft_model` / resume: `cast_trainable_params_to_fp32` — только trainable LoRA в fp32; frozen base UNet остаётся fp16.

Backward при `mixed_precision=float16`: `GradScaler.scale/unscale/step/update`. Resume сохраняет `grad_scaler_state_dict` в `.state.pt`.

### Kohya sd-scripts (по умолчанию, без `--full_fp16`)

1. UNet и LoRA network загружаются в **fp32**.
2. `mixed_precision="fp16"` в Accelerate влияет только на **autocast forward** (промежуточные активации в fp16).
3. Trainable LoRA weights остаются **fp32**; градиенты копятся в fp32 master weights.
4. Флаг `--full_fp16` явно переводит network в fp16 — авторы помечают как экспериментальный, «может повлиять на стабильность».

**Итог:** lora-trainer по умолчанию ближе к режиму Kohya `--full_fp16`, а не к стандартному mixed precision Kohya.

## Расхождение #2: GradScaler при backward

### lora-trainer (до fix N)

```python
# trainer.py (legacy)
loss.backward()                    # без GradScaler
torch.nn.utils.clip_grad_norm_(...) 
optimizer.step()
```

### lora-trainer (после fix N)

```python
grad_scaler.scale(loss).backward()
# на sync grad accum:
grad_scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
grad_scaler.step(optimizer)
grad_scaler.update()
```

- MSE считается в fp32 (`model_pred.float()`, `target.float()`) — это правильно.
- `loss.backward()` без масштабирования loss.
- При fp16 **weights** LoRA градиенты считаются и накапливаются в fp16-арифметике → риск underflow (мелкие градиенты → 0) и накопления ошибки округления.

### Kohya sd-scripts

```python
accelerator.backward(loss)
if accelerator.sync_gradients:
    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
optimizer.step()
```

При `mixed_precision="fp16"` HuggingFace Accelerate автоматически использует `torch.cuda.amp.GradScaler`:

- `scaler.scale(loss).backward()` — loss умножается перед backward
- `scaler.unscale_()` перед clip
- `scaler.step(optimizer)` — пропуск шага при inf/nan в градиентах

Цитата из обсуждения PR sd-scripts (#1259): *"Without [GradScaler], mixed precision training with fp32 weights and fp16 autocasting will degrade a lot."* GradScaler нужен **только для fp16**, не для bf16.

## Сводная таблица

| | Kohya (default fp16) | lora-trainer (до fix N) | lora-trainer (после fix N) |
|---|---------------------|------------------------|----------------------------|
| UNet base weights | fp32 | fp16 | fp16 |
| LoRA trainable weights | fp32 | fp16 | fp32 ✅ |
| Forward compute | autocast fp16 | autocast fp16 | autocast fp16 |
| Loss MSE | fp32 | fp32 ✅ | fp32 ✅ |
| Backward | GradScaler | без scaler ❌ | GradScaler ✅ |
| AdamW8bit betas/wd | 0.9/0.999, 0.01 | 0.9/0.999, 0.01 ✅ | 0.9/0.999, 0.01 ✅ |

## Почему loss «нормальный», а картинка — статик

Per-step MSE на **одном случайном timestep** может оставаться в разумном диапазоне, пока веса LoRA уже испорчены для **итеративного denoising** (20–30 шагов с накоплением ошибки).

При `lr=1e-3` (в 20× выше дефолта lora-trainer 5e-5) fp16-градиенты без scaling дают крупные, шумные обновления → к ep2 веса уходят в область, где UNet на inference выдает статик. Loss при этом не обязан монотонно расти.

Ep1 «нормальная» картинка согласуется с моделью **постепенной** fp16-деградации, а не мгновенного бага conditioning.

## AdamW8bit — вторичный фактор

AdamW8bit квантует optimizer state (m/v) для экономии VRAM. При **fp32 LoRA weights + GradScaler** (как Kohya) это работает стабильно.

При **fp16 LoRA weights без GradScaler** квантизация optimizer state добавляет ещё один источник шума, но **не является корневой причиной** — корневая: fp16 weights + no loss scaling.

## Рекомендуемый fix в коде (реализован, июль 2026)

При `mixed_precision == float16`:

1. **`cast_trainable_params_to_fp32`** — после `get_peft_model` / resume LoRA: только `requires_grad` params (LoRA A/B) в fp32; frozen UNet base остаётся fp16.
2. **`GradScaler`** — `scaler.scale(loss).backward()`, `unscale_` перед clip, `scaler.step` + `update`.
3. Resume: `grad_scaler_state_dict` в `.state.pt` (старые checkpoint'ы без поля — fresh scaler).

Модуль: `src/trainer/sdxl/mixed_precision.py`. Интеграция: `src/trainer/sdxl/trainer.py`.

**bf16 не обязателен** — GradScaler только для float16, не для bf16/float32.

## Контрольный прогон после fix

**Выполнен** (fresh retrain, тот же конфиг + fix N в коде). Результаты: `08-fix-n-post-run-jul2026.md`.

| Критерий | Результат |
|----------|-----------|
| ep2–3 без деградации | **Да** (vs jul до N: ep2 уже плохо) |
| ep4+ без статика | **Нет** — снова шум |
| likeness к ep3–5 | Частично — ep2–3 usable |

**Вывод:** fix N необходим; для полной стабильности нужен **G** (LR/warmup).

Рекомендуемый следующий конфиг:

```yaml
mixed_precision: float16
optimizer:
  type: adamw_8bit
learning_rate: 0.0002              # или 0.0005
lr_warmup_steps: 40                # ~5% от total steps
lr_scheduler: cosine
enable_bucket: true
batch_size: 2
epochs: 5                          # опционально
```

**Usable checkpoint текущего прогона:** `Winx_Bloom_CFTS_epoch2` или `_epoch3`.

## Связанные документы

- `05-bucketing-run-jul2026.md` — симптомы jul-прогона
- `08-fix-n-post-run-jul2026.md` — результаты post-N retrain
- `03-hypotheses.md` — гипотеза **N** (fp16 stack), **G** (LR)
- `04-pipelines.md` — diff train loop vs Kohya

## Ссылки (Kohya sd-scripts)

- [train_network.py — full_fp16 только при явном флаге](https://github.com/kohya-ss/sd-scripts/blob/main/train_network.py)
- [train_network.py — accelerator.backward(loss)](https://github.com/kohya-ss/sd-scripts/blob/main/train_network.py)
- [PR #1259 — GradScaler required for fp16 mixed precision](https://github.com/kohya-ss/sd-scripts/pull/1259)
