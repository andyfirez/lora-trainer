# Прогон Winx_Bloom после fix N (fp32 LoRA + GradScaler)

Отчёт о контрольном retrain после реализации гипотезы **N** (`mixed_precision.py`, июль 2026).

## Контекст

Jul-прогон **до fix N** (`05-bucketing-run-jul2026.md`): ep1 ok → **ep2** сильная деградация → ep5+ TV-статик при `lr=1e-3`, fp16, adamw_8bit.

Fix N в коде:

- `cast_trainable_params_to_fp32` после `get_peft_model` / resume LoRA
- `GradScaler` при `mixed_precision=float16` (backward / unscale / step / update)
- Resume: `grad_scaler_state_dict` в `.state.pt`

**Прогон:** fresh start, **без resume** jul checkpoint'ов (до fix N).

## Конфиг

Тот же `D:/SD/lora_output/Winx_Bloom_CFTS/Winx_Bloom_CFTS_config.yaml`, ключевые параметры:

| Параметр | Значение |
|----------|----------|
| rank / alpha | 16 / 16 |
| learning_rate | 1e-3 |
| lr_scheduler | cosine |
| lr_warmup_steps | **0** |
| mixed_precision | float16 |
| optimizer | adamw_8bit |
| enable_bucket | true |
| batch_size | 2 |
| epochs | 10 |
| repeats | 2 |
| save_every_n_epochs | 1 |

Fix collate (L) и bucketing (C) — включены, как в jul-прогоне.

## Результат (подтверждено пользователем)

| Эпоха | Наблюдение |
|-------|------------|
| 1 | Нормальное качество |
| 2 | **Нормальное качество** (vs jul: сильная деградация уже на ep2) |
| 3 | **Нормальное качество** |
| 4+ | Снова деградация → шум / TV-статик |

**Дополнительный симптом:** при переходе в шум **изображение почти не меняется** между сэмплами — типичный признак **насыщения весов LoRA** (overcooking): residual-поток UNet перестаёт зависеть от входного шума/timestep, выдаёт near-fixed-point паттерн.

Loss (ожидаемо): без монотонного роста и NaN — как в jul-прогоне; MSE на случайных timesteps не отражает коллапс итеративного denoising.

## Сравнение с jul-прогоном (до fix N)

| | До fix N | После fix N |
|---|----------|-------------|
| ep1 | ok | ok |
| ep2 | **деградация** | **ok** |
| ep3 | абстракт / полосы | **ok** |
| ep4+ | статик | деградация → статик |
| «Картинка не меняется» в шуме | да | да |

**Вывод по N:** fix **частично подтверждён** — precision stack (fp16 LoRA + no GradScaler) был реальной причиной **ранней** деградации (ep2). Он **не устраняет** позднюю деградацию при `lr=1e-3`, `warmup=0`, 10 эпох.

## Интерпретация: гипотеза G (LR / overcooking)

При `lr=1e-3`, `cosine`, `lr_warmup_steps=0` LR остаётся высоким на протяжении многих эпох:

| Эпоха (из 10) | Множитель cosine | Эффективный LR |
|---------------|------------------|----------------|
| 1 (10%) | ~0.98 | ~9.8e-4 |
| 2 (20%) | ~0.91 | ~9.1e-4 |
| 3 (30%) | ~0.79 | ~7.9e-4 |
| 4 (40%) | ~0.66 | ~6.6e-4 |
| 5 (50%) | ~0.50 | ~5.0e-4 |

К ep3–4 модель уже получила ~30–40% полного cosine-цикла при почти максимальном LR. Для rank16/alpha16 (effective scale = 1.0) и малого датасета (80 img, repeats=2) это достаточно для **переобучения / насыщения LoRA** — совпадает с симптомом «стабильный шум, картинка не меняется».

Fix N убрал fp16-precision instability; **G (LR без warmup + медленный cosine decay)** объясняет оставшуюся деградацию с ep4+.

## Практические рекомендации

### Сейчас (без retrain)

- **Использовать checkpoint ep2 или ep3** как финальный (`save_every_n_epochs=1` → файлы `Winx_Bloom_CFTS_epoch2.safetensors` / `_epoch3`).
- Не использовать ep4+ для inference.

### Следующий эксперимент (гипотеза G)

```yaml
learning_rate: 0.0002   # или 0.0005 — fallback из 03-hypotheses
lr_warmup_steps: 40     # ~5% от ~800 steps
lr_scheduler: cosine
epochs: 5               # опционально: меньше эпох, если best на ep2–3
# остальное как в jul-конфиге + fix N в коде
```

**Критерий успеха:** ep2–5 без статика; likeness стабилен до конца прогона или best на ep3–4.

### Диагностика (опционально)

- Логировать норму весов LoRA (`||lora_A||`, `||lora_B||`) по эпохам — ожидаемый «взрыв» магнитуды к ep4+.
- Сравнить ep3 checkpoint в reForge с ep2 (должны быть близки по качеству).

## Статус гипотез

| ID | Статус после прогона |
|----|---------------------|
| **N** | **Частично подтверждена** — fix убрал раннюю деградацию; недостаточен один |
| **G** | **P0** — основной подозреваемый для ep4+ |
| **E** | **Частично подтверждена** — best checkpoint ранняя эпоха (ep2–3), не финальный |

## Связанные документы

- `07-fp16-gradscaler-vs-kohya.md` — fix N в коде
- `05-bucketing-run-jul2026.md` — jul-прогон до fix N
- `09-lr-constant-post-run-jul2026.md` — lr3e-4 constant; M → P0
- `03-hypotheses.md` — G, E, M
