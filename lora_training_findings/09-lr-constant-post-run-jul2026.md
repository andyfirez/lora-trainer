# Прогон Winx_Bloom: lr=3e-4, constant scheduler (после fix N)

Отчёт о retrain с пониженным LR и constant scheduler (июль 2026). Следующий шаг после `08-fix-n-post-run-jul2026.md` (гипотеза G).

## Контекст

| Прогон | LR | Scheduler | warmup | ep1–3 | ep4+ |
|--------|-----|-----------|--------|-------|------|
| Jul, до fix N | 1e-3 | cosine | 0 | ep1 ok, ep2 деградация | статик |
| Post-N (`08`) | 1e-3 | cosine | 0 | **ok** | статик |
| **Этот прогон** | **3e-4** | **constant** | 0 | **значительно лучше** | **шум сохраняется** |

Fix N (fp32 LoRA + GradScaler) — в коде. Bucketing + fix collate — включены.

## Конфиг (ключевые отличия от post-N)

| Параметр | Значение |
|----------|----------|
| learning_rate | **0.0003** (3e-4) |
| lr_scheduler | **constant** |
| lr_warmup_steps | 0 |
| rank / alpha | 16 / 16 |
| mixed_precision | float16 |
| optimizer | adamw_8bit |
| enable_bucket | true |
| batch_size | 2 |
| repeats | 2 |

Остальное — как в `Winx_Bloom_CFTS_config.yaml` (base illustriousXL, cache latents/TE, 10 epochs).

## Результат (подтверждено пользователем)

- **Качество заметно лучше**, чем при `lr=1e-3` + cosine (post-N прогон).
- **ep1–3** — приемлемое качество.
- **С ep4** — снова появляется шум / деградация (хотя общий уровень лучше, чем на 1e-3).
- В **Kohya_ss** на тех же данных шума не было **при любых настройках** — указывает на **структурное** расхождение пайплайнов, а не только на гиперпараметры.

## Выводы по гипотезам

### G (LR / overcooking) — частично подтверждена

Понижение LR с 1e-3 до 3e-4 и переход на constant scheduler **существенно улучшили** ранние и средние эпохи. Гипотеза G объясняет **раннюю** деградацию на высоком LR, но **не объясняет полностью** шум с ep4 при мягких настройках.

| ID | Статус |
|----|--------|
| **G** | **Частично подтверждена** — LR/scheduler критичны; недостаточны для полной стабильности |
| **M** | **P0** — главный подозреваемый для оставшегося ep4+ шума |

### M (train/infer add_time_ids mismatch) — основной подозреваемый

**Train** (`buckets.py`): per-image Kohya semantics — `(source_h, source_w, crop_y, crop_x, target_h, target_w)` с **ненулевым crop** для большинства landscape-исходников (76/80 non-square).

Пример: исходник 1216×918 → bucket 768×1024 → `(918, 1216, 3, 0, 768, 1024)`.

**Inference / preview** (`latent_sampling/session.py`):

```python
# _build_add_time_ids(height, width, device)
[height, width, 0, 0, height, width]
```

То есть «source == target, crop = (0,0)» — **другой вектор micro-conditioning**, чем при bucketing train.

**Почему проявляется на поздних эпохах, а не с ep1:**

1. ep1–2: LoRA-дельта мала → поведение в основном задаёт base UNet, tolerant к OOD conditioning.
2. ep3–4+: веса LoRA выросли → модель сильнее «заточена» под train-распределение `add_time_ids` (crop ≠ 0).
3. Inference с `(H,W,0,0,H,W)` — out-of-distribution для обученной LoRA → нестабильный denoising → шум.
4. Симптом «картинка почти не меняется» в шуме — near-fixed-point при неверном conditioning.

**Почему Kohya стабилен «при любых настройках»:**

Kohya согласует `add_time_ids` между train и generation (те же bucket/crop semantics в pipeline). Это **не гиперпараметр** — lora-trainer не лечится дальнейшим снижением LR без fix M.

Шум одинаков в **встроенном preview** и **reForge** — оба используют inference conditioning `(H,W,0,0,H,W)`, не train crop coords.

## Сравнение прогонов Bloom (июль 2026)

| Прогон | ep2 | ep3 | ep4+ | Kohya-parity? |
|--------|-----|-----|------|---------------|
| До fix N, lr1e-3 cosine | деградация | полосы | статик | нет (N + G) |
| Post-N, lr1e-3 cosine | ok | ok | статик | частично (N fix) |
| **Post-N, lr3e-4 constant** | **лучше** | **лучше** | **шум** | G лучше; M не fix |

## Практические рекомендации

### Сейчас

- **Usable checkpoint:** ep2–3 текущего прогона (не ep4+).
- Не интерпретировать ep4+ preview как «overcooking от LR» — при 3e-4 constant это слабое объяснение.

### Следующий шаг (P0)

**Fix M в коде:** выровнять `latent_sampling/session.py` (и eval/reForge path при необходимости) с train semantics `add_time_ids`.

Варианты реализации (для обсуждения при fix):

- Для preview: `(target_h, target_w, 0, 0, target_h, target_w)` как минимум согласованный «no crop at target» case.
- Лучше: передавать типичный или явный crop из sampling config / bucket metadata.

**Критерий успеха после fix M:** ep4–10 без шума при `lr=3e-4` constant (или Kohya-parity lr) — как в Kohya.

### Не приоритет сейчас

- Дальнейшее снижение LR (<2e-4) без fix M — маловероятно устранит ep4+ шум.
- max_token_length 250 (F) — вторично.

## Связанные документы

- `08-fix-n-post-run-jul2026.md` — post-N прогон lr1e-3
- `04-pipelines.md` — diff train/infer add_time_ids
- `03-hypotheses.md` — статус G, M
- `07-fp16-gradscaler-vs-kohya.md` — fix N
