# LoRA Training Findings

Документация по расследованию качества LoRA в `lora-trainer` (июнь 2026).

## Содержание

| Файл | Описание |
|------|----------|
| [01-summary.md](01-summary.md) | Краткое саммари и текущий статус |
| [02-problem-and-symptoms.md](02-problem-and-symptoms.md) | Симптомы и контекст проблемы |
| [03-findings.md](03-findings.md) | Технические находки по пайплайну обучения |
| [04-attempts-and-fixes.md](04-attempts-and-fixes.md) | Что уже было сделано для улучшения |
| [05-kohya-comparison.md](05-kohya-comparison.md) | Сравнение с Kohya_ss |
| [06-winx-chimera-case-study.md](06-winx-chimera-case-study.md) | Кейс Winx_Chimera_CFTS (13 images) |
| [07-recommendations.md](07-recommendations.md) | Рекомендации и следующие шаги |
| [08-winx-bloom-case-study.md](08-winx-bloom-case-study.md) | **Кейс Winx_Bloom_CFTS (80 images) — контрольный эксперимент** |

## Ключевой вывод (обновлено 2026-06-30)

Проблема **воспроизводится на двух разных датасетах** (Chimera 13 img, Bloom 80 img). LoRA **обучается** (веса растут, loss падает), но likeness **не появляется**.

Наиболее вероятная причина — **systemic pipeline gap**, а не конкретный датасет:

1. ~~**add_time_ids mismatch**~~ — hypothesis A отвергнута (1024² = portrait, оба без likeness).
2. **Нет bucketing**: 76/80 изображений Bloom — non-square, force crop в 1024².
3. **lora_alpha/rank = 0.5** при Kohya alpha=rank (scale 1.0).
4. ~~**clip_skip**~~ — **red herring**: default=2 = старый `hidden_states[-2]`; Kohya игнорирует для SDXL train.

Output артефакты: `D:/SD/lora_output/Winx_Bloom_CFTS/`, `D:/SD/lora_output/Winx_Chimera_CFTS/`.
