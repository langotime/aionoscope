# Plan: Parameter samplers (processes + views) + meta export

## Контекст / проблема
Сейчас многие параметры в процессах/вьюхах задаются константами (например `PulseTrainProcess.frequency_hz`, `PulseTrainProcess.amplitude`, `NoiseView.noise_std`, `MissingnessView.dropout_prob` и т.д.). Для обучения/пробинга нам нужно:
1) чтобы эти параметры **семплировались** по заданным правилам (константа — тоже правило),
2) чтобы **реализованные значения** (sampled values) были доступны в `meta` как регрессионные/классификационные таргеты,
3) при этом соблюдать текущие ограничения репозитория:
   - всегда принимать `torch.Generator` и использовать его (без `torch.manual_seed` внутри модулей),
   - не хранить в `meta` full-size шум/маски (`[B,C,L]`, `[B,K,L]`, …), только seeds + низкоразмерные параметры,
   - fail-fast без “тихих” fallback’ов.

Это дополняет и продолжает идею из `plans/8_sampled_params_meta.md`: вводим единый механизм выборки параметров + единый способ публикации их в meta.

---

## Цель (что считаем “готово”)
1) В библиотеке есть набор базовых `Sampler`-ов (часто используемые правила семплинга).
2) Процессы и views принимают **sampler-ы вместо констант** (или `SamplerLike = value | Sampler` как совместимый переходный API).
3) Любой sampled параметр (включая константы, поданные через `ConstantSampler`) записывается в meta по единым правилам:
   - процесс: `LatentState.meta["samples"][<scope>][<param>]`
   - view: в мета конкретного view (оно уже namespaced внутри `Observation.meta["views"]`)
4) Добавлены тесты на:
   - детерминизм sampler-ов (фиксированный `torch.Generator` → одинаковые значения),
   - корректные shapes/dtypes для sampled значений,
   - наличие sampled значений в meta.

---

## Термины
- **config-параметры**: то, что определяет *формы тензоров / схему / размеры* (например `seq_len`, `components`, `kernel_size`, `num_leads`). Должно быть фиксировано для батча и обычно задаётся константами на этапе `__init__`.
- **sampled-параметры**: то, что может меняться между батчами/сэмплами *без изменения формы выходных тензоров* (например `noise_std`, `amplitude`, `jitter_std`, вероятности missingness).
- **shape-affecting sampled**: параметры, которые потенциально меняют форму внутренних/выходных тензоров (например частота → число событий; downsampling stride/window → длина). Для них нужен отдельный явный дизайн (см. ниже).

---

## Предлагаемое ядро: Sampler API

### Базовый контракт
`Sampler` — объект, который умеет семплить тензор параметров заданной формы, используя `torch.Generator`.

Предлагаемая сигнатура (MVP):
```python
class Sampler(Protocol):
    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ...

    def spec(self) -> dict[str, Any]:
        """Минимально сериализуемое описание правила (для логов/доков)."""
```

Принципы:
- `sample()` **всегда** использует переданный `rng` (никаких глобальных сидов).
- возвращаемый тензор имеет **ровно** `shape` и `dtype`, на `device`.
- `spec()` возвращает “конфиг” (константы/границы/параметры распределения) — полезно для отладки и экспорта в docs; хранить в meta опционально (см. ниже).

### Нормализация входных значений: `SamplerLike`
Чтобы не ломать текущий API и поддержать “константа как правило”, вводим тип:
- `SamplerLike[T] = T | Sampler`

И helper:
- `sampler_from(value_or_sampler, *, name: str) -> Sampler`
  - если передан `Sampler` — вернуть как есть,
  - если передано число/тензор — завернуть в `ConstantSampler`.

Важно: это **не silent default** — пользователь явно передал константу как параметр, и она интерпретируется как “constant rule”.

---

## Список “наиболее популярных” sampler-ов (MVP)
Ниже — минимальный набор, который покрывает 90% параметров в синтетических генераторах.

### 1) `ConstantSampler(value)`
- Возвращает тензор, заполненный `value`.
- Для чисел и небольших константных тензоров.

### 2) `UniformSampler(low, high)`
- Непрерывное равномерное распределение.
- Требование: `high > low`.

### 3) `LogUniformSampler(low, high)`
- Равномерно в log-домене: `exp(U(log(low), log(high)))`.
- Для положительных масштабов (`noise_std`, `amplitude`, …).
- Требование: `low > 0`, `high > low`.

### 4) `NormalSampler(mean, std, *, clamp: tuple[low, high] | None)`
- Гаусс.
- Опциональный clamp для “разумных” границ (в виде явного параметра, без скрытых ограничений).
- Требование: `std > 0`.

### 5) `RandIntSampler(low, high)`
- Дискретное равномерное по целым: `[low, high)` (или `[low, high]` — выбрать и зафиксировать в реализации).
- Для `gap_length`, `max_delay` (если решим делать sampled), числа событий, и т.п.

### 6) `BernoulliSampler(p)`
- Возвращает `bool` тензор по вероятности `p`.
- Для gated параметров/включений.

### 7) `CategoricalSampler(probs)`
- Возвращает `int64` индексы по `probs` (1D список вероятностей).
- Для выбора режима/класса как параметра (если хотим единый механизм вместе с `SampleLabelsNode`).

### 8) `ChoiceSampler(choices, probs | None)`
- Возвращает значения из `choices` (python list) путём семплинга индекса через `CategoricalSampler`.
- Удобно для малых дискретных наборов (`rounding`, `mode`, preset-ы).
- В реализации важно решить: возвращаем `torch.Tensor` (индексы) + таблица `choices` в meta, или возвращаем python-значения (хуже для векторизации). MVP: возвращать индексы + `choice_names` в meta.

Не-MVP (можно добавить позже, если будет реальная потребность): BetaSampler, TriangularSampler, Mixture/OneOfSampler, Correlated samplers (совместная выборка нескольких параметров одним вызовом).

---

## Как процессы и views используют sampler-ы

### Общий helper: “sample + record”
Вместо того чтобы каждый модуль руками писал в meta, вводим 2 helper-а:
- `process_sample_and_record(meta, *, scope: str, name: str, sampler: Sampler, shape: tuple[int,...], rng, device, dtype) -> torch.Tensor`
- `view_sample_and_record(view_meta, *, name: str, sampler: Sampler, shape: tuple[int,...], rng, device, dtype) -> torch.Tensor`

Оба:
- семплируют значение,
- записывают его в meta под согласованным ключом,
- возвращают тензор значения для дальнейших вычислений.

### Meta схема (итог)
#### Процесс
Следуем `plans/8_sampled_params_meta.md`:
- `LatentState.meta["samples"]`: `dict[str, dict[str, Any]]`
  - `scope`:
    - для монолитных процессов: `"TrendSeasonAnomalyProcess"`, `"PulseTrainProcess"`, …
    - для graph-ноды: `"<NodeClass>:<out_key>"` (например `"EventTrainNode:events"`).

Пример:
```python
meta["samples"]["PulseTrainProcess"]["frequency_hz"] = frequency_hz  # [B] или [B,1]
meta["samples"]["PulseTrainProcess"]["amplitude"] = amplitude        # [B]
```

#### View
Так как view уже namespaced внутри `Observation.meta["views"]`, достаточно держать в meta view’хи:
- `meta["samples"]`: dict с sampled параметрами
- `meta["spec"]`: (опционально) dict с `Sampler.spec()` для читаемого конфига

Пример entry в `Observation.meta["views"]`:
```python
{
  "view": "NoiseView",
  "seed": ...,
  "samples": {"noise_std": noise_std},   # [B]
  "spec": {"noise_std": {"kind": "log_uniform", ...}},
}
```

---

## Shape-affecting параметры: явные правила
Некоторые параметры нельзя семплировать per-sample без ragged’а:
- `SamplingAggregationView.stride/window` меняют `L'`
- `PulseTrainProcess.frequency_hz` влияет на число событий → меняет `E` (размерность events)
- `KernelConvView.kernel_size/padding` потенциально завязаны на частоту (через `spacing`)

Предлагаем KISS-политики:
1) **MVP правило:** shape-affecting параметры семплируются **per-batch** (одно значение на батч) и применяются ко всем `B`. В meta всё равно пишем в форме `[B]` через broadcast, чтобы это было удобным dense target.
2) Явно документируем, какие параметры являются shape-affecting, и запрещаем для них “per-sample shape” в первом релизе sampler-ов (через fail-fast проверки в соответствующих классах, а не “угаданные” эвристики).
3) Расширение (не в MVP): поддержка variable event count через `E_max + mask` для events и/или фиксированная морфология kernel bank (не зависящая от RR spacing), если потребуется per-sample heart rate.

Open point для ревью: хотим ли мы сразу делать “variable E with mask”, или достаточно “per-batch frequency” на первом шаге?

---

## Конкретные изменения по классам (первый проход)

### Processes
#### `PulseTrainProcess`
Что переводим на sampler-ы:
- `frequency_hz: SamplerLike[float]` (shape-affecting → MVP per-batch)
- `amplitude: SamplerLike[float]` (per-sample)
- `missed_gap_factor: SamplerLike[float]` (per-sample)

Как это реализуем (варианты; выбрать один после ревью):
- Вариант A (минимальные изменения графа): обновить `EventTrainNode` так, чтобы:
  - `num_events` мог быть computed внутри `forward` (из sampled `frequency_hz`),
  - `amplitude` и `missed_gap_factor` могли быть тензорами `[B]`,
  - node сама записывала `frequency_hz/amplitude/missed_gap_factor/num_events/spacing` в `meta["samples"]["EventTrainNode:<out_key>"]`.
- Вариант B (яснее по ответственности): добавить `SampleParamsNode` перед `EventTrainNode`, который:
  - семплит параметры,
  - пишет их в `state.data` и `meta["samples"]`,
  - `EventTrainNode` читает уже готовые тензоры из `state.data`.

Рекомендация: Вариант B (лучшее разделение; меньше “магии” внутри генератора событий).

#### `TrendSeasonAnomalyProcess`
Уже есть множество sampled тензоров, но они не сохраняются (см. `plans/8_sampled_params_meta.md`). В рамках sampler-инициативы:
- не обязательно сразу делать sampler-ы на все config-параметры,
- но обязательно привести meta экспорт к единому стандарту (`meta["samples"]["TrendSeasonAnomalyProcess"]`).

### Views
#### `NoiseView`
- `noise_std: SamplerLike[float]` → семплим `[B]` или `[B,1,1]` и применяем broadcast к `[B,C,L]`.
- в meta пишем `samples.noise_std` и `spec.noise_std`.

#### `MissingnessView`
- `dropout_prob/gap_prob/hold_prob: SamplerLike[float]` (per-sample)
- `gap_length: SamplerLike[int]` (MVP per-batch, иначе появятся неодинаковые маски по длине; можно расширить позже)
- маски **не** храним в meta (как в `plans/8_sampled_params_meta.md`), только seed + sampled probs/length.

#### `ECGLeadsView` (опционально в первом проходе)
- `jitter_std: SamplerLike[float]` (per-sample или per-batch)
- `max_delay: SamplerLike[int]` (shape-preserving, но влияет на распределение delays; MVP per-batch)
- sampled `delays` уже сохраняются; надо добавить `samples.jitter_std/max_delay` для пробинга.

---

## Публичный API (как пользователь будет писать код)
Пример (после внедрения):
```python
from aiono import (
  PulseTrainProcess, NoiseView,
  ConstantSampler, UniformSampler, LogUniformSampler,
)

process = PulseTrainProcess(
  seq_len=2048,
  frequency_hz=UniformSampler(1.2, 2.5),     # per-batch MVP
  sample_rate_hz=500.0,
  rhythm_classes=[...],
  shape_classes=[...],
  latent_mode="pqrst3",
  amplitude=LogUniformSampler(0.5, 2.0),     # per-sample
  missed_gap_factor=UniformSampler(1.8, 3.0) # per-sample
)

noisy_view = NoiseView(noise_std=LogUniformSampler(0.05, 0.25))
```

В meta:
- `obs.meta["process"]["samples"]["PulseTrainProcess"]["amplitude"]` → `[B]`
- `obs.view_meta("NoiseView")["samples"]["noise_std"]` → `[B]`

---

## План реализации (одна фаза, но с проверяемыми шагами)
1) **Core: samplers**
   - Добавить `aiono/core/samplers.py`: интерфейс + реализации MVP sampler-ов.
   - Добавить `aiono/core/sampling_params.py` (или аналогичный): `sampler_from(...)`, `process_sample_and_record(...)`, `view_sample_and_record(...)`.
   - Экспортировать sampler-ы из `aiono/__init__.py`.

2) **Meta schema plumbing**
   - Зафиксировать ключ `meta["samples"]` на process-стороне (как в `plans/8_sampled_params_meta.md`).
   - Для view: добавить в meta каждого view `samples/spec` (не ломая текущий `Observation.view_meta()`).

3) **Processes**
   - `aiono/processes/pulse_train.py`: перевести параметры на sampler-ы, записывать sampled значения в `meta["samples"]`.
   - (Если нужно) `aiono/processes/nodes.py`: добавить `SampleParamsNode` и/или расширить `EventTrainNode` для чтения sampled параметров.
   - `aiono/processes/trend_season.py`: сохранить already-sampled параметры в `meta["samples"]` (без изменения поведения генерации).

4) **Views**
   - `aiono/views/noise.py:NoiseView` → `noise_std` sampler + meta export.
   - `aiono/views/missingness.py:MissingnessView` → prob sampler-ы + meta export (без хранения масок).
   - (Опционально) `aiono/views/ecg_leads.py:ECGLeadsView` → sampler-ы для `jitter_std/max_delay` + meta export.

5) **Docs + examples**
   - Обновить `DOCUMENTATION.md` и 1-2 примера в `examples/` (и соответствующие `.ipynb`) так, чтобы было видно:
     - как передавать sampler-ы,
     - где искать sampled значения в meta.

6) **Tests (pytest)**
   - `tests/test_samplers_determinism.py`: детерминизм + shapes/dtypes для всех MVP sampler-ов.
   - `tests/test_meta_samples_process_pulse_train.py`: наличие `meta["samples"]["PulseTrainProcess"]` и expected shapes.
   - `tests/test_meta_samples_view_noise_missingness.py`: наличие `samples` в view meta и shapes.
   - Прогон: `uv run pytest`.

---

## Вопросы для ревью (нужны ответы до кода)
1) `PulseTrainProcess.frequency_hz`: достаточно ли **per-batch** семплинга (одно значение на батч), или вам принципиально нужен **per-sample** heart rate уже в первой версии?
2) Хотим ли мы хранить `Sampler.spec()` в meta (`spec`), или достаточно только realized `samples`?
3) Для дискретных параметров типа `ChoiceSampler`: вы хотите получать в meta **индексы + список имён**, или “готовые” python-значения (хуже для тензорных таргетов)?

