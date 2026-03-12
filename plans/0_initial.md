# Aionoscope: синтетический датасет временных рядов с разделением Process vs View (GPU, PyTorch)
**Назначение:** технический план для AI coding agent (реализация библиотеки/репозитория).

---

## 0) Контекст и мотивация

Нужно построить синтетический датасет временных рядов для сравнения **SSL** (JEPA/MAE/Masked prediction и др.) и **supervised** подходов. Датасет должен:

- включать задачи, напоминающие ECG-классификацию:
  - **локальная морфология** (shape/local): различия в форме событий/мотивов в коротких окнах;
  - **глобальный ритм** (rhythm/global): различия в распределении интервалов/паттернах событий (дальние зависимости).
- быть **GPU-friendly**: генерация в PyTorch тензорах, без Python-циклов по батчу; приоритет — векторизация и простые операции (einsum, conv1d, broadcasting).
- иметь **строгое разделение**:
  - **Process (latent process)**: “истинная” динамика (события, скрытые режимы, латентные компоненты);
  - **View (observation model / representation)**: как процесс наблюдается (лиды ECG, единицы измерения, нормировки, sampling/aggregation, пропуски, квантизация, шум, задержки).
- легко расширяться: новые процессы и новые views должны подключаться как модульные компоненты в цепочках.

---

## 1) Цели (Goals)

### G1. Универсальный каркас Process → View → Observation
- Процесс порождает LatentState: события, латентные компоненты, метки.
- View преобразует LatentState в наблюдение x (многоканальный сигнал) + метки.

### G2. Набор базовых процессов, покрывающих “ECG-like” и “метрики серверов”
- Event-driven pulse train (ECG-like).
- Regime-switching / ON-OFF bursts.
- Trend + seasonality + anomalies (для метрик, телеметрии, финансовых рядов).
- Irregular sampling/point processes (опционально, этап 2).

### G3. Набор базовых views (представлений)
- Linear mixing (multi-lead ECG, multi-sensor).
- Unit transforms (absolute ↔ percent of capacity, log, z-score, clipping).
- Sampling / downsampling / aggregation windows.
- Missingness, задержки, timestamp jitter.
- Noise models (white/colored), baseline wander, quantization/rounding.

### G4. Поддержка SSL multi-view батчей
- Из одного LatentState получать несколько независимых Observation (две/три “view” одного процесса).
- Удобный API для обучения: вернуть `{"view_a": Observation, "view_b": Observation, ...}`.

### G5. Репродуцируемость и анти-“shortcut” дизайн
- Отделение nuisance от меток.
- Метки рассчитываются в Process (истина), а View не должен “подсказывать” класс.
- Контрольные проверки/бенчмарки против “коротких путей”.

---

## 2) Non-goals (чего не делаем на первом этапе)

- Полноценная физиологическая симуляция ECG (вся электрофизиология, 12-lead кинематика и т.д.).
- Полная реализация нерегулярных событийных потоков с календарным временем и сложными marked point processes (можно добавить позже).
- Реальный dataset формат (WFDB и т.п.). Здесь только синтетика.

---

## 3) Требования

### Функциональные
- Генерация батчей на GPU: `x: [B, C, L]`, метки `y` как dict.
- Возможность задать:
  - тип процесса (например, rhythm классы),
  - тип морфологии/событий (shape классы),
  - параметры сложности (SNR, jitter, missingness, etc.).
- Возможность собрать конфигурации как композиции модулей.

### Нефункциональные
- Быстро: генерация батча не должна доминировать над forward модели.
- Чистый API: минимальная связанность модулей.
- Тестируемость: unit-тесты на формы, размеры, диапазоны, детерминизм, инвариантности.
- Расширяемость: новые модули добавляются без переписывания ядра.

---

## 4) Ключевые абстракции и API

### 4.1. Типы данных

#### `LatentState`
Содержит “истину” процесса.
```python
@dataclass
class LatentState:
    centers: torch.Tensor           # [B, N, 1] (нормированное время 0..1) или [B,N] в сэмплах
    latent: torch.Tensor            # [B, K, L] латентные компоненты (может быть None на этапе 2)
    y: dict[str, torch.Tensor]      # {"shape": [B], "rhythm": [B], ...}
    meta: dict[str, Any]            # параметры процесса: HR, jitter, regime params, seed, etc.
````

#### `Observation`

Содержит наблюдение (то, что “видит” модель).

```python
@dataclass
class Observation:
    x: torch.Tensor                 # [B, C, L]
    y: dict[str, torch.Tensor]      # те же метки
    meta: dict[str, Any]            # параметры view: A-matrix, capacity, quantization, etc.
```

---

### 4.2. Интерфейсы модулей

#### `Process` (latent)

```python
class Process(nn.Module):
    def forward(self, batch_size: int, device: torch.device, *, rng: torch.Generator | None = None) -> LatentState:
        ...
```

#### `View` (observation model)

```python
class View(nn.Module):
    def forward(self, z: LatentState, *, rng: torch.Generator | None = None) -> Observation:
        ...
```

#### `Pipeline` (orchestrator)

```python
class SynthPipeline(nn.Module):
    def __init__(self, process: Process, views: dict[str, View]):
        ...
    def forward(self, batch_size: int, device: torch.device, rng: torch.Generator | None = None) -> dict[str, Observation]:
        ...
```

---

## 5) Архитектура пакета (предлагаемая структура репозитория)

```
aiono/
  core/
    types.py            # LatentState, Observation
    rng.py              # seed utilities, torch.Generator helpers
    pipeline.py         # SynthPipeline
    utils.py            # broadcasting helpers, shape checks
  processes/
    base.py             # Process interface
    pulse_train.py      # event-driven ECG-like pulse processes
    regimes.py          # regime-switching / ON-OFF
    trend_season.py     # trend+seasonality+anomalies
  kernels/
    morph.py            # gaussian / laplace / DoG / PQRST mixture
    motifs.py           # motif bank (опционально этап 2)
  views/
    base.py             # View interface
    ecg_leads.py        # A0 + ΔA mixing, delays
    units.py            # absolute<->percent, log, zscore, clipping
    sampling.py         # downsample, aggregation windows, resampling
    missingness.py      # dropout, hold, random gaps
    noise.py            # white/colored, baseline wander, quantization
  tasks/
    labels.py           # правила генерации меток, sanity-checks
  datasets/
    iterable.py         # IterableDataset / generator wrapper
  tests/
    test_shapes.py
    test_determinism.py
    test_shortcuts.py
  examples/
    ecg_shape_vs_rhythm.py
    server_metrics_abs_vs_pct.py
  configs/
    *.yaml (опционально)
```

---

## 6) Базовый процесс: PulseTrainProcess (ECG-like)

### 6.1. Состав процесса

1. Rhythm generator: генерирует `centers` (regular / irregular / missed / bigeminy / bursty).
2. Morph kernel: задаёт форму события (gaussian / sharp / biphasic DoG / PQRST mixture).
3. Renderer: строит `latent` компоненты (рекомендуется K=3: P/QRS/T или K=1 для простого старта).
4. Labels: `y_shape` и `y_rhythm` определяются параметрами генерации (НЕ view).

### 6.2. Минимальный набор классов (MVP)

* Rhythm classes: `regular`, `irregular`, `missed_beat`.
* Shape classes: `gaussian`, `sharp_laplace`, `biphasic_dog`.
* Latent: либо K=1 (все события одной формой), либо K=3 (P/QRS/T) — предпочтительно K=3 для multi-lead realism.

---

## 7) Базовые views

### 7.1. ECG multi-lead view (вариант 3)

**Цель:** фиксированная “каноническая” матрица `A0: [C, K]` + небольшое отклонение `ΔA` + задержки по lead.

* `A = A0 + Normal(0, jitter_std)`
* `x = einsum("bck,bkl->bcl", A, latent)`
* опционально: delays, per-lead baseline, per-lead noise.

### 7.2. Units view (для метрик серверов)

**Цель:** один и тот же процесс может быть представлен в разных единицах.

* `absolute`: `x_abs = u(t) + noise`
* `percent`: `x_pct = 100 * u(t) / capacity`, `capacity` — параметр view, затем clipping [0,100]
* `log`: `log1p`, `softplus`, и т.п.

### 7.3. Sampling/Aggregation view

* Downsample: `x[:, :, ::k]` или low-pass + decimate.
* Aggregation: оконные mean/max (симуляция monitoring dashboards).

### 7.4. Missingness view

* random point dropout
* contiguous gaps
* hold-last-value (“staleness”)

### 7.5. Noise / baseline / quantization view

* baseline wander (низкочастотный синус/случайный тренд)
* colored noise (через простую IIR/conv фильтрацию)
* quantization/rounding

---

## 8) План реализации (подробные этапы)

### Этап 1 — MVP (1–2 процесса, 2–3 views, multi-view батч)

**Deliverables:**

* `LatentState`, `Observation`, `SynthPipeline`.
* `PulseTrainProcess` (K=3 latent) + 3 rhythm + 3 shape классы.
* `ECGLeadsView` (C=2..12).
* `NoiseView` + `NormalizeView` (минимальный набор).
* Пример: `examples/ecg_shape_vs_rhythm.py` возвращает два view батча: `{"clean": ..., "noisy": ...}`.

**Acceptance criteria:**

* Генерация батча на GPU без Python-циклов по B (допускается небольшой цикл по C, если C=12 и это не узкое место; позже оптимизировать).
* Размерности и типы стабильны.
* Метки не зависят от view nuisance параметров.

### Этап 2 — “универсальные процессы” и “универсальные views”

**Deliverables:**

* `TrendSeasonAnomalyProcess` (для метрик).
* `UnitsView` (absolute vs percent of capacity).
* `SamplingAggregationView`, `MissingnessView`.
* Пример: `examples/server_metrics_abs_vs_pct.py` создаёт два view одного процесса: `abs` и `pct`.

**Acceptance criteria:**

* Один и тот же LatentState → два представления → SSL-ready.
* Инвариантность: можно обучать модель различать процессные метки, игнорируя единицы.

### Этап 3 — анти-shortcut тесты и оптимизация

**Deliverables:**

* `tests/test_shortcuts.py`: простые baseline-классификаторы на статистиках (mean/std/energy/peak_count) не должны решать “сложные” классы слишком хорошо.
* Профилирование генерации (torch.profiler) и оптимизация горячих мест.
* Опционально: `torch.compile` совместимость (без динамических Python-ветвлений в hot path).

---

## 9) Примеры использования (API)

### 9.1. ECG: shape vs rhythm + multi-view для SSL

```python
process = PulseTrainProcess(
    seq_len=2048, num_pulses=8,
    rhythm_classes=["regular", "irregular", "missed_beat"],
    shape_classes=["gaussian", "sharp", "biphasic"],
    latent_mode="pqrst3"  # K=3
)

A0 = make_canonical_A0(num_leads=12, num_latent=3)  # фиксированная матрица
views = {
    "ecg_clean": ECGLeadsView(A0=A0, jitter_std=0.03, max_delay=3),
    "ecg_noisy": nn.Sequential(
        ECGLeadsView(A0=A0, jitter_std=0.03, max_delay=3),
        BaselineWanderView(...),
        PerLeadNoiseView(...),
        NormalizeView(...)
    )
}

pipe = SynthPipeline(process=process, views=views)
batch = pipe(batch_size=64, device=torch.device("cuda"))

x1 = batch["ecg_clean"].x   # [64, 12, 2048]
x2 = batch["ecg_noisy"].x   # [64, 12, 2048]
y_shape = batch["ecg_clean"].y["shape"]
y_rhythm = batch["ecg_clean"].y["rhythm"]
```

### 9.2. Server metrics: один процесс → два представления (absolute vs percent)

```python
process = TrendSeasonAnomalyProcess(seq_len=1440, components=4, task_labels=["regime", "anomaly_type"])

views = {
    "abs": nn.Sequential(UnitsAbsoluteView(), NoiseView(), MissingnessView()),
    "pct": nn.Sequential(UnitsPercentOfCapacityView(capacity_dist=...), ClippingView(0,100), NoiseView())
}

pipe = SynthPipeline(process, views)
batch = pipe(batch_size=128, device="cuda")

x_abs = batch["abs"].x
x_pct = batch["pct"].x
y = batch["abs"].y
```

---

## 10) Best practices (инженерные)

### 10.1. Репродуцируемость

* Везде принимать `torch.Generator` (или seed) и использовать его при sampling.
* Не использовать глобальный `torch.manual_seed` внутри модулей.
* В `meta` писать seed/параметры, чтобы можно было воспроизвести конкретный батч.

### 10.2. Разделение ответственности

* `Process` отвечает за “истину” и метки.
* `View` отвечает за представление и измерительные искажения.
* Метки не должны зависеть от view-параметров (если явно не тестируем robustness).

### 10.3. Производительность

* Избегать Python-циклов по B и N. Предпочитать broadcasting/einsum/conv1d.
* Кэшировать `t_grid` как buffer в модулях.
* Стараться держать вычисления в float32 (или bfloat16 при необходимости), но шум/квантизация требуют аккуратности.
* По возможности избегать построения огромных промежуточных тензоров [B,N,L], если N и L большие:

  * для pulse train можно использовать impulse + conv1d (этап 3/опционально).

### 10.4. Тестируемость

* Тест на размерности и dtype.
* Тест на детерминизм при фиксированном `torch.Generator`.
* Тест на диапазоны (после clipping, quantization).
* Тест на отсутствие “коротких путей” (baseline-features не дают слишком высокий accuracy).

### 10.5. Документация и примеры

* Каждый процесс и view документировать: что он моделирует, какие параметры, какие инвариантности ожидаются.
* В `examples/` держать минимальные runnable скрипты.

---

## 11) Список задач (Task checklist для coding agent)

### Core

* [ ] Реализовать `LatentState`, `Observation`.
* [ ] Реализовать `SynthPipeline(process, views)`.

### Processes

* [ ] `PulseTrainProcess` (ритм + морфология + latent renderer K=3).
* [ ] `TrendSeasonAnomalyProcess` (для метрик).

### Views

* [ ] `ECGLeadsView(A0, jitter_std, max_delay)`.
* [ ] `UnitsAbsoluteView`, `UnitsPercentOfCapacityView`, `ClippingView`.
* [ ] `NoiseView`, `BaselineWanderView`, `NormalizeView`.
* [ ] `SamplingAggregationView`, `MissingnessView`.

### Tests

* [ ] shape/dtype tests
* [ ] determinism tests
* [ ] shortcut baseline tests

### Examples

* [ ] `ecg_shape_vs_rhythm.py`
* [ ] `server_metrics_abs_vs_pct.py`

---

## 12) Примечания по расширению (после MVP)

* Добавить “motif bank” процесс (вариант B) через impulse + conv1d.
* Добавить marked point process для нерегулярных потоков событий (переход к irregular time).
* Добавить multi-task режим: y может включать сразу несколько ортогональных меток.
* Добавить конфиги (YAML) и фабрики для сборки пайплайнов.

---

## 13) Критерии готовности проекта (Definition of Done)

* Есть минимум 2 процесса (pulse_train и trend+season).
* Есть минимум 4 views (ECG leads, units transform, sampling/aggregation, missingness/noise).
* Есть примеры, которые генерируют multi-view батчи и демонстрируют:

  * shape vs rhythm задачи на ECG-like,
  * abs vs pct на server metrics.
* Есть тесты на детерминизм и на отсутствие тривиальных shortcut-решений.
* Генерация батчей работает на CUDA и не вызывает заметных CPU bottleneck на разумных размерах (например B=256, L=2048, C=12).

---

```
