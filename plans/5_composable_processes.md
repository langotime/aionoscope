# Plan: ProcessGraph + event-first латенты (реализация)

## Что делаем
Делаем **компонуемые графы процессов**: процесс собирается из переиспользуемых узлов (`ProcessNode`) с branching/merging **на уровне латентных структур** (event streams, параметры, латентные компоненты), а **рендеринг в регулярные отсчёты** (например, `impulse → conv1d`) выполняется во **Views**.

Цель: можно строить и очень простые ряды (одиночное событие, серия событий одного типа), и сложные смеси (несколько потоков событий + медленные компоненты + режимы), чтобы поддержать curriculum для SSL.

## Зафиксированные решения (по требованиям)
- **ProcessGraph** — базовая архитектура.
- **ProcessChain** — частный случай/синтаксический сахар (линейный граф).
- **Repeatability между запусками** важна: фиксированный seed + фиксированный граф ⇒ повторяемый output.
- **Process генерирует латенты** (в т.ч. события). **View отвечает за рендеринг** в регулярные отсчёты, каналы, сенсоры и т.п.
- Численные мерджи типа `sum/mul/ratio/% of` полезны **в Views**, когда появляются массивы отсчётов. В ProcessGraph мерджим в основном **структуры**: event streams, выбор режима, фильтры, порождение событий, конкатенацию источников.

## Конкретный API (предложение)

### 1) События: один канонический формат (без ragged Python структур)
Чтобы не плодить форматы и не уехать в Python loops, канонический формат событий — padded тензоры + mask.

Файл: `toyts/core/events.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

@dataclass(frozen=True)
class EventSchema:
    # стабильное перечисление типов (важно для kernel bank / routing)
    type_names: list[str]          # len=T
    param_names: list[str]         # len=P (MVP: ["amplitude"])
    time_unit: str                 # MVP: "samples" (0..L-1)

@dataclass(frozen=True)
class EventBatch:
    # Пакет событий фиксированной длины E с маской валидности.
    times: torch.Tensor            # float32 [B, E]   (в units schema.time_unit)
    type_ids: torch.Tensor         # int64   [B, E]   (0..T-1)
    params: torch.Tensor           # float32 [B, E, P]
    mask: torch.Tensor             # bool    [B, E]
    schema: EventSchema
    meta: dict[str, Any]

    def to(self, device: torch.device) -> EventBatch: ...
```

Ключевые правила:
- `event types ≠ channels`. Событие имеет `type_id`, а **view** решает маппинг `type_id → channel(s)` и/или `type_id → kernel`.

### 2) Расширение `LatentState` под события
Файл: `toyts/core/types.py`
- Добавить поле: `events: EventBatch | None`.
- Для event-only процессов: `latent=None`, `events!=None`.
- Для dense-only процессов: `events=None`, `latent!=None`.
- Для смешанных: допускается оба.

### 3) ProcessGraph: задаем граф прямо в коде (без `requires/provides`)
Файл: `toyts/processes/graph.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

@dataclass
class ProcessState:
    batch_size: int
    device: torch.device
    data: dict[str, Any]
    y: dict[str, torch.Tensor]
    meta: dict[str, Any]

class ProcessOp(nn.Module):
    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        raise NotImplementedError

GraphSpec = ProcessOp | list[ProcessOp]  # list = Seq([...]) sugar

class ProcessNode(ProcessOp):
    # leaf op: читает/пишет ключи в state.data/state.y (явно через параметры узла)
    ...

class Seq(ProcessOp):
    def __init__(self, ops: list[ProcessOp]) -> None: ...

class Switch(ProcessOp):
    # per-sample branching по label (или по state.data)
    def __init__(
        self,
        *,
        label_key: str,
        cases: dict[int, GraphSpec],
        default: GraphSpec | None = None,
    ) -> None: ...

class Parallel(ProcessOp):
    # запускает несколько подграфов и кладет их outputs в namespace
    def __init__(self, *, branches: dict[str, GraphSpec]) -> None: ...

class Scope(ProcessOp):
    # prefix all writes in a subgraph to avoid key collisions
    def __init__(self, *, prefix: str, op: GraphSpec) -> None: ...

class ProcessGraph(nn.Module):
    # graph задается как nested структура: list => Seq([...]), плюс явные ops для branching/merging
    def __init__(self, *, graph: GraphSpec, outputs: set[str], name: str) -> None: ...

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState: ...

class ProcessChain(ProcessGraph):
    # syntactic sugar: цепочка операций (линейный граф)
    def __init__(self, *, nodes: list[ProcessOp], outputs: set[str], name: str) -> None: ...
    ...
```

Примеры задания графа “как код”:

```python
# 1) Линейная часть как список (ProcessChain sugar)
process = ProcessChain(
    name="single_event",
    outputs={"events"},
    nodes=[
        SampleLabelsNode(...),
        SingleEventNode(out_key="events", ...),
    ],
)

# 2) Branching по label внутри графа
process = ProcessGraph(
    name="pulse_train_events",
    outputs={"events"},
    graph=[
        SampleLabelsNode(...),
        Switch(
            label_key="rhythm",
            cases={
                0: [RegularIntervalsNode(...), BuildEventsNode(out_key="events", ...)],
                1: [IrregularIntervalsNode(...), BuildEventsNode(out_key="events", ...)],
                2: [MissedBeatIntervalsNode(...), BuildEventsNode(out_key="events", ...)],
            },
        ),
    ],
)

# 3) Параллельные источники + merge (в ProcessGraph мерджим структуры/события)
process = ProcessGraph(
    name="trend_plus_incidents",
    outputs={"events"},
    graph=[
        Parallel(
            branches={
                "trend": [TrendEventsNode(out_key="events", ...)],
                "inc": [IncidentTrainNode(out_key="events", ...)],
            }
        ),
        UnionEventsNode(in_keys=["trend.events", "inc.events"], out_key="events"),
    ],
)
```

Правила/контракты:
- **Порядок выполнения определяется кодом**:
  - линейные части — как `Seq([...])` (или через `ProcessChain`),
  - branching/merging — через явные `Switch/Parallel/Scope` + merge-nodes (например, `UnionEventsNode`).
- Узлы **не требуют** ручного объявления `requires/provides`: зависимости выражаются через явные `in_key/out_key` параметры узлов, и отсутствие входов ловится fail-fast при выполнении.
- Граф можно менять “как код”: переставлять элементы списка, удалять узлы, вставлять новые — без пересборки деклараций зависимостей.
- `meta` в конце обязательно содержит:
  - `process`: имя процесса/графа,
  - `seed`: seed процесса,
  - `trace_seeds`: seed каждого op (по execution trace),
  - параметры процесса (seq_len, частоты, имена типов событий и т.п.).

RNG:
- `ProcessGraph.forward`: `rng_make_generator` → передаёт generator в `graph`.
- Каждый container-op (`Seq/Switch/Parallel`) делает `rng_split` для своих дочерних ops в фиксированном порядке.
- Repeatability гарантируется между запусками при фиксированном графе и seed.

## Branching/merging: какие блоки нужны для высокой гибкости

### Branching (routing)
Варианты реализации branching:
1) “compute-all-then-select” (простота): вычислить кандидаты для всех B и выбрать `torch.where` (дороже по compute).
2) “masked execution” (эффективнее): вычислять только для subset (через индексацию), затем вставлять обратно.

Блоки:
- `BranchByLabel(label_key, routes=...)`: режимы/классы → разные подграфы.
- `SelectByMask(mask_key, a_key, b_key, out_key)`: выбрать один из вариантов по маске.
- `ConditionalApply(mask_key, node, out_prefix=...)`: применить узел только к subset.
- `Fork(in_key, out_keys=[...])`: сохранить несколько версий потока (например, raw/jittered).

### Merging (структуры/события, не массивы отсчётов)
Блоки:
- `UnionEvents(a_key, b_key, out_key)`: суперпозиция потоков + сортировка по времени.
- `DedupeEvents(in_key, out_key, min_dt)`: refractory / схлопывание коллизий по времени.
- `FilterEvents(in_key, predicate=..., out_key)` / `GateEvents(in_key, mask_key, out_key)`: фильтрация.
- `MapType(in_key, mapping, out_key)`: remap `type_id` (слияние типов, переименование, доменные адаптеры).
- `DeriveEvents(in_key, rule=..., out_key)`: порождение новых событий из старых (триггер, задержка, “response after request”, “aftershock”).
- `SuppressNearby(in_key, by_key, window, out_key)`: ингибирование событий типа A в окне вокруг B.
- `ConcatSources(keys=[...], out_key="sources")`: структурная сборка источников (без numeric sum).

Примечание: `sum/mul/ratio/% of` и похожие операции лучше делать **после рендеринга в views**, потому что они имеют смысл на уровне наблюдаемого сигнала.

## Multivariate: как работаем с каналами и типами
Канонический подход:
1) ProcessGraph генерирует **EventBatch** (тип + параметры) и/или набор latent-компонент.
2) View решает:
   - как типы событий рендерятся в компоненты/каналы,
   - как компоненты миксуются и как выглядят сенсоры (матрицы, фильтры, resampling).
3) Если нужно “разделить по типам” — делаем это масками/индексацией на тензорах, но не держим dict как основной формат.

## Рендеринг событий во Views (интеграция `impulse → conv1d`)
Цель: убрать огромные промежуточные [B, N, K, L] и позволить держать events как first-class.

Предлагаемые views (MVP):
- `EventStreamView`: `LatentState(events=...) -> Observation(x=packed_events, ...)` для event-only представления.
- `EventImpulseView`: `LatentState(events=...) -> Observation(x=impulse[B,T,L], ...)` (scatter-add по типам).
- `KernelConvView`: `Observation(impulse) -> Observation(x=dense[B,K,L], ...)` (conv1d kernel bank).

Для ECG pipeline нужно разрешить схему:
`LatentState(events) -> Observation(latent_components) -> Observation(leads) -> ...`

Для этого:
- адаптировать `ECGLeadsView`, чтобы он принимал `Observation` как источник “latent components” (где `x` имеет форму [B,K,L]) и возвращал `Observation` [B,C,L].

## План работ (конкретные шаги реализации)
1) **Core types**
   - `toyts/core/events.py`: `EventSchema`, `EventBatch`.
   - `toyts/core/types.py`: добавить `LatentState.events`.
2) **ProcessGraph runtime**
   - `toyts/processes/graph.py`: `ProcessState`, `ProcessOp`, `ProcessNode`, `Seq`, `Switch`, `Parallel`, `Scope`, `ProcessGraph`, `ProcessChain`.
   - Исполнение строго по структуре графа (list/Seq-части в порядке, branching через `Switch`, параллельные источники через `Parallel`).
   - Fail-fast проверки ключей на входе узлов (нет `requires/provides` деклараций).
   - RNG splitting внутри container-ops (`Seq/Switch/Parallel`) + `trace_seeds` в meta.
3) **Набор узлов (MVP для curriculum + домены)**
   - `LabelSamplerNode` (режимы/классы).
   - `SingleEventNode` (одно событие).
   - `EventTrainNode` (серия событий одного типа, регулярная/случайная).
   - `UnionEventsNode`, `DedupeEventsNode`, `MapTypeNode`.
   - `TimeJitterNode`, `TimeShiftNode`, `Gate/FilterNode`.
4) **ECG pulse train: refactor в event-first**
   - Процесс выдаёт `events` + `y` + `meta` (без построения dense PQRST в process).
   - Типы событий кодируют форму QRS (например, `qrs_gaussian/qrs_laplace/qrs_dog`) или `type_id + param shape_id`.
5) **Views для событий**
   - `EventStreamView`, `EventImpulseView`, `KernelConvView`.
   - Адаптация `ECGLeadsView` под вход `Observation(x=[B,K,L])`.
6) **Тесты (pytest)**
   - Детерминизм (seed ⇒ одинаковые события и одинаковый рендер).
   - Shapes/dtypes.
   - Регрессия/проверка, что избегаем [B,N,K,L] как основного пути.
7) **Документация и примеры**
   - Добавить в документацию примеры branching/merging (режимы, опциональные эффекты, параллельные источники, curriculum).
   - Показать: event-only view и dense-rendered view из одного процесса.
