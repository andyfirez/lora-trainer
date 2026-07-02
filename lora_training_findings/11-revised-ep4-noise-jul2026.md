# Пересмотр гипотезы M и ep4+ шум (июль 2026)

Пересмотр после ручной валидации fix M. **M понижена с P0**; root cause ep4+ шума смещён на train loop / checkpoint weights.

## Что произошло

Ручная валидация fix M выполнена: standalone sampling job 25 (`source_job_id=23`, Bloom lr3e-4 constant), ep1–10, prompt/seed фиксированы.

**Результат:** изображения **полностью идентичны** прогону до fix M. Fix не оказал никакого эффекта.

## Почему fix M оказался no-op

Sampling job 25: `sample_width=832`, `sample_height=1216`.

Датасет 2 (`Winx_Bloom`, `target_resolution=1024`, `enable_bucket=true`) — реальное распределение bucket'ов (из `dataset_image_crops`):

| bucket (WxH) | N изображений |
|--------------|---------------|
| 1024×960 | 17 |
| 1024×768 | 16 |
| 1024×832 | 11 |
| 1024×1024 | 8 |
| 1024×896 | 7 |
| 960×1024 | 5 |
| 768×1024 | 4 |
| 896×1024 | 4 |
| 832×1024 | 2 |
| остальные | по 1 |

**Максимальная сторона любого bucket'а — 1024** (ограничено `target_resolution`). Bucket'а `832×1216` **не существует и не может существовать**.

`resolve_reference_add_time_ids()` ищет точное совпадение `(bucket_width, bucket_height) == (832, 1216)` → не находит → возвращает `None` → сэмплер падает на старый хардкод `(1216, 832, 0, 0, 1216, 832)`. В логах job 25 строка `using aligned add_time_ids` **отсутствует для всех 10 эпох** — подтверждение no-op.

## Ключевой аргумент против M как root cause

Ошибка в самой постановке гипотезы M:

1. **Цель пользователя** — обучать на невысоких исходниках (bucket'ы ≤1024) и **генерировать выше** (`832×1216` и т.д.) в reForge/ComfyUI. Для high-res txt2img правильное inference conditioning — это именно `(H, W, 0, 0, H, W)` = «хочу картинку такого размера», а **не** train-time crop coords. Fix M подставлял бы train crop coords, что для high-res генерации неверно по смыслу.

2. **Шум ep4+ виден и в reForge.** reForge использует один и тот же inference pipeline для Kohya-LoRA и lora-trainer-LoRA. Kohya-LoRA в reForge на `832×1216` работает без шума. Если бы root cause был в том, как передаются `add_time_ids` при инференсе, страдали бы **обе** LoRA одинаково. Страдает только lora-trainer-LoRA → проблема в **весах checkpoint'ов**, которые пишет lora-trainer, а не в inference conditioning.

3. Fix M затрагивает только built-in preview и standalone sampler lora-trainer. Он **структурно не способен** объяснить расхождение «Kohya хорошо / lora-trainer плохо в одном и том же reForge».

**Вывод:** M не является root cause ep4+ шума. Оставляем как низкоприоритетный косметический fix для in-app preview (когда сэмплим ровно в train-bucket'е), но снимаем с списка объяснений деградации.

## Куда смещается диагноз

Симптоматика указывает на **порчу весов LoRA в процессе обучения**, а не на inference:

| Наблюдение | Интерпретация |
|------------|---------------|
| ep2–3 usable → ep4+ шум/«статик» | LoRA деградирует к середине обучения |
| Loss стабилен (0.13–1.2), без взрыва | Нет NaN; баг в training signal или в накоплении ошибки весов, а не в расходимости loss |
| Kohya стабилен при любых LR | Структурное расхождение train loop, не гиперпараметры |
| Шум и в preview, и в reForge | Проблема в checkpoint'ах, не в sampler UI |
| «Картинка почти не меняется» в шуме | near-fixed-point — веса LoRA доминируют и уводят denoising |

## Новые/пересмотренные гипотезы

### O: порча/насыщение весов LoRA к ep4+ — ОТКЛОНЕНА (проверено)

Прямая проверка выполнена: эффективная ‖ΔW‖ = `(alpha/rank)·‖up·down‖` по 700 модулям, lora-trainer Bloom ep1–10 vs здоровая Kohya-LoRA (Chimera, ep40 final).

| checkpoint | mean ‖ΔW‖ | max ‖ΔW‖ | up_mean | down_mean | max_abs | рост/эпоху |
|------------|-----------|----------|---------|-----------|---------|-----------|
| ep1 | 0.708 | 3.55 | 0.72 | 2.48 | 0.056 | — |
| ep2 | 0.815 | 4.80 | 0.81 | 2.51 | 0.067 | ×1.15 |
| ep3 | 0.892 | 5.85 | 0.87 | 2.53 | 0.098 | ×1.10 |
| ep4 | 0.929 | 6.27 | 0.90 | 2.54 | 0.111 | ×1.04 |
| ep5 | 0.964 | 6.77 | 0.93 | 2.55 | 0.122 | ×1.04 |
| ep10 | 1.387 | 9.09 | 1.23 | 2.66 | 0.297 | ×1.02 |
| **Kohya ep40 (здоровая)** | **7.97** | **46.3** | **4.68** | **4.91** | **0.318** | — |

**Выводы:**
1. Веса lora-trainer растут **плавно** (×1.02–1.15/эпоху), к ep10 всего ×2 от ep1. Нет взрыва, нет насыщения, нет runaway-модуля (max/mean ≈ 6.5 у lora-trainer, ≈ 5.8 у Kohya — сопоставимо).
2. Здоровая Kohya-LoRA имеет ‖ΔW‖ **в ~6 раз больше** (mean 7.97 vs 1.39) и при этом стабильна → магнитуда/overcooking не объясняют шум.
3. Излом кривой роста весов **не совпадает** с ep4 (моментом появления шума).

**Гипотеза O отклонена. E (overcooking) по магнитуде тоже не подтверждается.**

**Побочная находка (важно для N):** конфиг Kohya — `full_fp16 = true`, `ss_mixed_precision = fp16`. То есть Kohya обучал стабильную LoRA с **fp16-весами сети**, а не fp32. Предпосылка fix N («стабильность из-за fp32 LoRA weights») **не подтверждается** — fp16-веса могут быть стабильны. GradScaler мог играть роль, но «fp16 vs fp32 weights» как root cause снимается.

**Структурная деталь для дальнейшего:** у Kohya `up_mean ≈ down_mean` (сбалансированы), у lora-trainer `down`(≈2.5) ≫ `up`(0.7–1.2). Вероятно, следствие 40 vs 10 эпох (up zero-init растёт дольше), но стоит держать в уме.

### P0 — D: аудит train loop vs Kohya sd-scripts (построчно)

Сверить то, что реально влияет на веса:

- **target / prediction_type** — сейчас читается из scheduler (`epsilon` vs `v_prediction`, `trainer.py:427-431`). Проверить, что для illustrious scheduler даёт то же, что Kohya.
- **timesteps sampling** — `torch.randint` uniform (`trainer.py:390`). Совпадает ли с Kohya (min/max, распределение)?
- **latents scaling** — `vae.config.scaling_factor` (`trainer.py:382`).
- **noise_offset / min_snr_gamma** — в Winx оба 0 → не источник, но зафиксировать parity.
- **clip_grad_norm=1.0** (`trainer.py:456`) — совпадает ли с Kohya `max_grad_norm`?
- **steps per epoch** — `repeats × images / batch`. Совпадает ли определение «эпохи» с Kohya (у Kohya repeats задаётся в имени папки)?

### P1 — R: bucket_reso_steps parity

Kohya Winx config: `bucket_reso_steps=256`. lora-trainer default: `64` (`config.py:126`). Разные шаги → разные bucket'ы → разный train signal и per-image `add_time_ids`. Не объясняет ep4 alone, но нарушает parity с эталоном.

### P2 — F: max_token_length 77 vs Kohya 250

Длинные caption'ы молча обрезаются на 77 токенах (`te_cache`, tokenizer). Возможная потеря тегов → искажение обучающего сигнала. Ниже приоритет, но дёшево проверить.

## Практический workaround (не fix)

ep2–3 checkpoint'ы Bloom (lr3e-4 constant) пригодны. Быстрый тест цели пользователя: **ep2–3 → reForge → `832×1216`**. Если результат хороший — high-res генерация с ранних checkpoint'ов **работает**, и проблема локализована именно в поздних epoch weights.

## Конкретные следующие шаги

1. **Weight-stats анализ** ep1/ep3/ep4/ep10 lora-trainer vs Kohya — скрипт чтения `.safetensors`, таблица норм по слоям/эпохам. (P0, без GPU, быстро)
2. **ep2–3 → reForge 832×1216** — подтвердить, что high-res inference достижим с ранних checkpoint'ов. (workaround проверки цели)
3. **Train loop diff (D)** — построчная сверка `trainer.py` train step с Kohya sd-scripts по пунктам выше. (P0)
4. По результату (1)+(3) — сформулировать конкретный fix train loop.

## Связанные документы

- `09-lr-constant-post-run-jul2026.md` — прогон, где ep4+ шум зафиксирован (тогда M был P0)
- `10-fix-m-validation.md` — валидация fix M (результат: no-op)
- `03-hypotheses.md` — обновлённые приоритеты
