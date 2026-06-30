# Attempts and Fixes

Хронология изменений и экспериментов по улучшению качества LoRA в `lora-trainer`.

---

## Фаза 1: Первичная диагностика

### Что выявили

- LoRA targets в `lora-trainer` уже были уже, чем в типичном Kohya recipe (только attention, без FF).
- Отсутствовали Min-SNR loss и noise offset.
- Defaults (rank, alpha, lr, epochs) отличались от community-standard.

### Действия

| Изменение | Файлы | Статус |
|-----------|-------|--------|
| Расширение UNet targets на FF-слои | `src/trainer/sdxl/lora_targets.py` | ✅ Внедрено |
| Min-SNR weighting | `src/trainer/sdxl/loss.py` | ✅ Внедрено |
| Noise offset | `src/trainer/sdxl/loss.py` | ✅ Внедрено |
| Обновление defaults (rank=32, alpha=32, lr=5e-5, epochs=30, min_snr, noise_offset) | `src/trainer/config.py` | ✅ Внедрено |
| UI для новых параметров | frontend | ✅ Внедрено |

### Результат

Частичное улучшение не подтверждено однозначно. Пользователь продолжил видеть плохое качество; возникли проблемы с overtraining на поздних эпохах.

---

## Фаза 2: Rank mismatch при sampling

### Симптом

```
The size of tensor a (32) must match the size of tensor b (16)
```

### Причина

Sampler использовал `TrainConfig` defaults (`lora_rank=32`), checkpoint был rank=16.

### Действия

| Изменение | Файлы |
|-----------|-------|
| `infer_kohya_lora_metadata()` — чтение rank/alpha/TE из safetensors | `src/trainer/sdxl/lora_export.py` |
| Динамическая пересборка PEFT stack при load | `src/sampler/sdxl/service.py` |
| Unit tests | `tests/trainer/test_lora_export.py` |

### Результат

✅ Sampling error исправлен. **Качество likeness не изменилось** — это был inference bug, не train bug.

---

## Фаза 3: Сравнение с Kohya config

### Что сделали

- Проанализировали `config_illustriousXL_v01_Winx_Chimera_CFTS_v1.json` и `config_illustriousXL_v01_Leggins_CFTS_v1.json`.
- Сопоставили параметры с возможностями `lora-trainer`.

### Выводы

- `loss_type: l2` в Kohya ≈ MSE в lora-trainer — **не причина расхождения**.
- Критичные gaps: **bucketing**, `max_token_length`, `clip_skip`.
- Пользователь ранее пробовал `min_snr_gamma=0`, `noise_offset=0` — skepticism про bucketing обоснован частично, но bucketing важен для **не-квадратных** изображений, не только для loss weighting.

---

## Фаза 4: LoRA initialization (Kohya-compatible)

### Проблема

PEFT по умолчанию инициализирует LoRA иначе, чем Kohya (gaussian vs Kaiming A + zero B).

### Действия

| Изменение | Файлы |
|-----------|-------|
| `build_sdxl_lora_config()` с `init_lora_weights=True` | `src/trainer/sdxl/lora_peft.py` |
| Использование helper в trainer и sampler | `src/trainer/sdxl/trainer.py`, `src/sampler/sdxl/service.py` |
| Unit tests (zero lora_B, identity before train) | `tests/trainer/test_lora_peft.py` |

### Результат

✅ Код внедрён и протестирован. Пользователь **перезапустил тренинг** — **«ничего не поменялось»**. Init не был root cause.

---

## Фаза 5: Углублённое расследование Winx_Chimera_CFTS

### Что проверили

| Проверка | Результат |
|----------|-----------|
| Реальный путь датасета | `D:/SD/datasets/Winx_Chimera` (из SQLite `datasets` table) |
| TE/latent cache | `.prepared/1024/*_te.npz`, `*_sdxl.npz` — 13 файлов каждого |
| job_9 cache load | 0 encoded, 13 loaded from disk (cache не пересоздавался) |
| Caption uniqueness | 13 unique captions |
| Image aspect ratios | Преимущественно landscape, не квадрат |
| Loss dynamics | avr_loss растёт после epoch 1 |
| Sampling config id=5 | Без trigger, от Melanie — eval issue |
| reForge test | Тот же плохой результат — train issue |

### Отвергнутые гипотезы (по подтверждению пользователя / данным)

- ❌ «Проблема только в trigger при sampling» — reForge с trigger тоже плохо.
- ❌ «Stale cache из-за смены датасета» — датасет не менялся.

### Оставшиеся гипотезы

- ✅ Overcooking (aggressive lr × repeats × epochs).
- ✅ Square crop без bucketing на landscape images.
- ✅ Шумные caption на маленьком датасете.
- ⚠️ TE cache invalidation bug — риск при будущих изменениях caption, не текущий root cause.

---

## Фаза 6: Эксперименты пользователя (вне кода)

| Эксперимент | Результат |
|-------------|-----------|
| min_snr_gamma=0, noise_offset=0 | Качество всё ещё плохое |
| Разные hyperparams (rank 16→32, repeats, epochs) | Качество всё ещё плохое |
| Перезапуск после Kohya init fix | Без улучшения |
| Тест в reForge | Без улучшения |

---

## Фаза 7: Winx_Bloom — контрольный эксперимент (2026-06-30)

### Что проверили

| Проверка | Результат |
|----------|-----------|
| Новый персонаж, 80 images | job_13 COMPLETED |
| Fresh cache | 80 encoded, 0 loaded |
| LoRA weight growth | up_mean 0.13 → 0.54 (ep1→20) |
| Visual / reForge result | **Тот же плохий** |
| Sampling config id=7 | Melanie prompt, без Bloom trigger |

### Вывод

- ❌ «Только маленький датасет» — Bloom 80 img, тот же результат.
- ❌ «Stale cache» — Bloom cache свежий.
- ❌ «LoRA не обучается» — веса растут, checkpoint'ы различаются.
- ✅ **Systemic pipeline** (add_time_ids, bucketing, alpha scale) — единственное объяснение, покрывающее оба кейса.

---

## Сводка: что реально изменилось в коде

```
src/trainer/sdxl/lora_targets.py    — FF targets
src/trainer/sdxl/loss.py            — min_snr, noise_offset
src/trainer/config.py               — updated defaults
src/trainer/sdxl/lora_peft.py         — Kohya init (NEW)
src/trainer/sdxl/lora_export.py     — metadata inference (NEW)
src/sampler/sdxl/service.py         — dynamic rank, build_sdxl_lora_config
tests/trainer/test_lora_peft.py     — NEW
tests/trainer/test_lora_export.py    — extended
```

**Не реализовано:** bucketing, clip_skip, max_token_length, caption dropout, cache invalidation by caption hash.
