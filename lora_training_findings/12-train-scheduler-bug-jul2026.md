# ROOT CAUSE: train forward process использует EulerDiscreteScheduler (июль 2026)

**Гипотеза D подтверждена. Найден root cause ep4+ шума и общей деградации.**

lora-trainer зашумляет латенты при обучении через **inference-scheduler** (`EulerDiscreteScheduler`), а не через DDPM. Его `add_noise` не variance-preserving и подаёт в UNet латенты, отмасштабированные до ×14.6 на высоких timestep'ах.

## Как обнаружено

### Шаг 1 — веса не виноваты (O отклонена)

Сравнение эффективной ‖ΔW‖ по 700 модулям: lora-trainer ep1–10 растут плавно (×2 к ep10), здоровая Kohya-LoRA имеет ‖ΔW‖ в ~6 раз больше и стабильна. Магнитуда/overcooking не объясняют шум. См. `11-revised-ep4-noise-jul2026.md`.

### Шаг 2 — какой scheduler реально используется

`model_loader.py` при загрузке single-file чекпойнта (`illustriousXL_v01.safetensors`) берёт `noise_scheduler = pipeline.scheduler`:

```python
# src/trainer/sdxl/model_loader.py:116-124
pipeline = StableDiffusionXLPipeline.from_single_file(...)
return SDXLComponents(
    ...
    noise_scheduler=pipeline.scheduler,   # <-- EulerDiscreteScheduler!
    ...
)
```

Проверка фактического класса:

```
scheduler class: EulerDiscreteScheduler
num_train_timesteps = 1000
prediction_type = epsilon
beta_schedule = scaled_linear
```

Поле типизировано как `DDPMScheduler`, но фактически это `EulerDiscreteScheduler` (по умолчанию для SDXL в `from_single_file`).

### Шаг 3 — численное подтверждение неверного зашумления

`train loop` (`trainer.py:393`) вызывает `noise_scheduler.add_noise(latents, noise, timesteps)`. Сравнение с корректным DDPM на одинаковых входах:

| timestep | DDPM ‖noisy‖ (correct) | Euler ‖noisy‖ (current) | ratio |
|----------|------------------------|-------------------------|-------|
| 0 | 255.7 | 255.8 | 1.00 |
| 100 | 256.0 | 270.7 | 1.06 |
| 300 | 256.5 | 333.8 | 1.30 |
| 500 | 256.9 | 488.7 | 1.90 |
| 700 | 257.1 | 894.9 | 3.48 |
| 900 | 257.1 | 2161.1 | 8.41 |
| 999 | 257.1 | 3765.7 | **14.65** |

## Почему это ломает обучение

**DDPM (правильно, variance-preserving):**

```
noisy = √ᾱ·x + √(1−ᾱ)·noise     →  ‖noisy‖ ≈ const
```

**EulerDiscreteScheduler.add_noise:**

```
noisy = x + σ·noise,   σ = √((1−ᾱ)/ᾱ)     →  ‖noisy‖ растёт как 1/√ᾱ
```

Связь: `Euler_noisy = DDPM_noisy / √ᾱ`. Множитель `1/√ᾱ` — это ровно `scale_model_input`, который Euler применяет **при инференсе** перед UNet. Но в train loop `scale_model_input` **не вызывается** — латенты `x + σ·noise` идут в UNet напрямую (`trainer.py:420-425`).

Итог: UNet получает timestep-эмбеддинг «t» и латент, отмасштабированный не так, как модель обучена ожидать для этого t. Рассинхрон нарастает с timestep (до ×14.6 при t=999). Это OOD-вход → **систематически неверный градиент**, особенно на высоких t (глобальная структура/композиция).

## Согласованность с наблюдениями

| Наблюдение | Объяснение через scheduler bug |
|------------|-------------------------------|
| ep1–3 ok → ep4+ шум | Неверный high-t сигнал накапливается по эпохам |
| Веса растут плавно, без взрыва | Магнитуда ок; кривая **функция** из-за неверного входа (O отклонена корректно) |
| Loss стабильно высокий (0.13–1.2) | На высоких t цель epsilon из мис-масштабированного входа почти невыучиваема |
| Kohya стабилен при любых LR | Kohya использует **DDPMScheduler** для train — корректный forward process |
| Шум и в preview, и в reForge | Испорчены сами веса checkpoint'а, а не inference |
| Понижение LR помогает лишь частично | Меньший шаг замедляет накопление ошибки, но не убирает мис-масштаб |

## Предлагаемый fix

Использовать **правильный DDPMScheduler** для train forward process, независимо от того, какой scheduler отдаёт pipeline (Kohya делает именно так).

Вариант A (точечно, `model_loader.py`): собирать train `noise_scheduler` как `DDPMScheduler` из тех же бет, что у pipeline, вместо `pipeline.scheduler`:

```python
DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    prediction_type="epsilon",
)
```

Беты брать из `pipeline.scheduler.config` (`beta_start/beta_end/beta_schedule/num_train_timesteps/prediction_type`), чтобы сохранить parity с чекпойнтом и поддержать v-pred модели.

Sampling-путь (`build_inference_scheduler`) **не трогать** — там Euler корректен для инференса.

## Верификация fix (требует retrain)

1. Пересобрать train `noise_scheduler` как DDPM.
2. Unit-тест: `add_noise` train-scheduler'а даёт variance-preserving ‖noisy‖ (≈const по t).
3. Retrain Bloom (те же гиперпараметры, lr3e-4 constant или lr1e-3).
4. Критерий: ep4–10 без шума, likeness сохраняется на поздних эпохах.

## Пересмотр прошлых fix

- **N (fp32 LoRA + GradScaler):** предпосылка «Kohya стабилен из-за fp32 весов» **неверна** — конфиг Kohya `full_fp16=true` (fp16-веса). N не был root cause. Возможно оставить GradScaler для fp16-стабильности, но это не главная причина.
- **M (aligned add_time_ids):** отклонена ранее (`11`); подтверждается — inference был ни при чём.
- **G (LR):** объясняет, почему меньший LR частично помогал (замедление накопления ошибки), но не root cause.

## Связанные документы

- `11-revised-ep4-noise-jul2026.md` — отклонение M и O, вывод к train loop
- `03-hypotheses.md` — D как подтверждённый root cause
- `04-pipelines.md` — train vs Kohya forward process
