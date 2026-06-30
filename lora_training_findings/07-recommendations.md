# Recommendations

Приоритизированные рекомендации (обновлено после Winx_Bloom, job_13).

---

## A. Немедленные эксперименты (без изменения кода)

### 1. Проверка alpha scale

Повторить короткий Bloom/Chimera прогон с **`lora_alpha = lora_rank`** (32/32 или 16/16), остальное без изменений.

Если likeness появится хотя бы слабо → alpha scale был significant contributor.

### 2. ~~Inference при training resolution~~ — **выполнено, отвергнуто**

1024×1024 и 832×1216 — оба без likeness. См. §F.

### 3. Eval prompt

```
Winx_Bloom_CFTS, <tags from dataset caption>
Negative: realistic, 3d
LoRA weight: 0.8–1.0
```

Смотреть epoch 1–5, не epoch 20.

### 4. Отдельный sampling config в UI

Создать config с trigger `Winx_Bloom_CFTS` / `Winx_Chimera_CFTS` — для regression test в built-in sampler (не влияет на reForge).

---

## B. Изменения в коде (приоритет после Bloom)

### P0 — Critical (единственные fixes, объясняющие оба кейса)

| Feature | Why |
|---------|-----|
| **Bucketing / multi-resolution** | 76/80 Bloom non-square; Kohya uses `enable_bucket` |
| **Default lora_alpha = lora_rank** | Текущий scale 0.5 vs Kohya 1.0 |
| Per-image add_time_ids при train | P2 после отвержения hypothesis A — mismatch не root cause |

### P1 — High

| Feature | Why |
|---------|-----|
| TE cache invalidation by caption hash | Future-proofing (caption/trigger changes) |
| Sampling config per LoRA project | UI eval broken (id=5, id=7 → Melanie) |

~~clip_skip~~ — **не нужен для SDXL** (Kohya игнорирует; default=2 = старый hardcode). Параметр оставлен в UI для совместимости, но не влияет на root cause.

### P2 — Medium

| Feature | Why |
|---------|-----|
| min_snr_gamma / noise_offset defaults | Отключены в проблемных прогонах |
| max_token_length | Long captions truncated at 77 |
| Early checkpoint picker | Chimera loss rises ep.2–4 |

---

## C. Validation experiment (после P0 fixes)

1. Winx_Bloom dataset (80 img) — уже есть.
2. rank=16, alpha=16, bucketing on, correct add_time_ids.
3. 10 epochs, sample every 2.
4. Eval: reForge 832×1216 **и** 1024×1024 для сравнения.

**Success:** likeness visible by epoch 3–5 at intended inference AR.

---

## D. Что больше не нужно проверять

- ❌ «Удалить cache и переобучить» — Bloom уже с fresh cache, не помогло.
- ❌ «Добавить больше картинок» — 80 img достаточно для sanity check.
- ❌ «Поменять только init LoRA / FF targets» — уже сделано, не помогло.
- ❌ «Loss должен падать → всё OK» — Bloom loss падает, результат плохой.

---

## E. Open questions

1. ~~Насколько inference at 1024×1024 улучшит Bloom без code changes?~~ → **Проверено (hypothesis A), см. §F**
2. Совместим ли Kohya export scale с reForge при alpha≠rank?

---

## F. Hypothesis A test results (2026-06-30)

**Setup:** `Winx_Bloom_CFTS_epoch5`, seed=42, steps=40, CFG=7.5, euler_a, prompt с trigger.

| Resolution | add_time_ids | Output |
|------------|--------------|--------|
| 832×1216 (portrait) | `[1216,832,0,0,1216,832]` | `hypothesis_a/epoch5_portrait_832x1216_seed42_00.png` |
| 1024×1024 (square) | `[1024,1024,0,0,1024,1024]` = train | `hypothesis_a/epoch5_square_1024x1024_seed42_00.png` |

**Script:** `scripts/hypothesis_a_resolution_test.py`

### Наблюдения

1. **Trigger + правильный prompt** — orange hair + blue eyes (Bloom palette), в отличие от старых samples без trigger (Melanie/teal). Это eval issue, не train fix.
2. **Оба resolution — плохое качество, likeness отсутствует** (подтверждено пользователем). Разница 832×1216 vs 1024×1024 **не даёт** заметного улучшения likeness.
3. Inference при train AR **не спасает** результат → mismatch add_time_ids **не объясняет** провал один.

### Вердict

| | |
|---|---|
| **Гипотеза A (add_time_ids / AR mismatch = root cause)** | **Отвергнута** |
| **add_time_ids mismatch** | Возможный minor factor для portrait vs square, но **не подтверждён** как значимый contributor |

**Следующий приоритет:** гипотеза B (alpha=rank), bucketing, другие gaps vs Kohya.
