from __future__ import annotations

import torch

from ..core.events import EventBatch, EventSchema
from .graph import ProcessNode, ProcessState


class SampleLabelsNode(ProcessNode):
    """Sample categorical labels into state.y."""

    def __init__(self, *, labels: dict[str, list[str]]) -> None:
        """Initialize label sampling with name-to-classes mapping."""
        super().__init__()
        if not labels:
            raise ValueError("labels must be non-empty.")
        for name, classes in labels.items():
            if not classes:
                raise ValueError(f"labels['{name}'] must be non-empty.")
        self.labels = {name: list(classes) for name, classes in labels.items()}

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Populate state.y with sampled label indices."""
        self._record_seed(state, rng)
        for name, classes in self.labels.items():
            indices = torch.randint(
                0,
                len(classes),
                (state.batch_size,),
                generator=rng,
                device=state.device,
            )  # [B]
            state.y[name] = indices

        label_names = state.meta.setdefault("label_names", {})
        for name, classes in self.labels.items():
            label_names[name] = classes
        return state


class SetLabelsNode(ProcessNode):
    """Set constant categorical labels into state.y."""

    def __init__(self, *, labels: dict[str, tuple[list[str], int]]) -> None:
        """Initialize constant labels.

        Args:
            labels: Mapping from label key -> (class_names, value_index).
        """
        super().__init__()
        if not labels:
            raise ValueError("labels must be non-empty.")

        normalized: dict[str, tuple[list[str], int]] = {}
        for name, (class_names, value) in labels.items():
            if not name:
                raise ValueError("Label name must be non-empty.")
            if not class_names:
                raise ValueError(f"labels['{name}'] class_names must be non-empty.")
            if len(set(class_names)) != len(class_names):
                raise ValueError(f"labels['{name}'] class_names must be unique.")
            if value < 0 or value >= len(class_names):
                raise ValueError(
                    f"labels['{name}'] value must be in [0, {len(class_names)}). Got {value}."
                )
            normalized[name] = (list(class_names), int(value))

        self.labels = normalized

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Populate state.y with constant label indices."""
        self._record_seed(state, rng)

        label_names = state.meta.setdefault("label_names", {})
        for name, (class_names, value) in self.labels.items():
            indices = torch.full(
                (state.batch_size,),
                fill_value=value,
                device=state.device,
                dtype=torch.int64,
            )  # [B]
            state.y[name] = indices

            if name in label_names and label_names[name] != class_names:
                raise ValueError(
                    f"Label '{name}' already exists in meta with different classes. "
                    f"Existing={label_names[name]}, new={class_names}."
                )
            label_names[name] = class_names

        return state


class SingleEventNode(ProcessNode):
    """Generate one event per sample."""

    def __init__(
        self,
        *,
        seq_len: int,
        schema: EventSchema,
        type_name: str,
        time_min: int,
        time_max: int,
        amplitude_min: float,
        amplitude_max: float,
        amplitude_param: str,
        out_key: str,
    ) -> None:
        """Initialize a single-event generator."""
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if time_min < 0 or time_max < 0:
            raise ValueError("time_min/time_max must be non-negative.")
        if time_max < time_min:
            raise ValueError("time_max must be >= time_min.")
        if time_max > seq_len - 1:
            raise ValueError("time_max must be <= seq_len - 1.")
        if amplitude_max < amplitude_min:
            raise ValueError("amplitude_max must be >= amplitude_min.")
        if not out_key:
            raise ValueError("out_key must be non-empty.")

        self.seq_len = seq_len
        self.schema = schema
        self.type_id = schema.type_id(type_name)
        self.time_min = time_min
        self.time_max = time_max
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.amplitude_index = schema.param_id(amplitude_param)
        self.out_key = out_key

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Create a single event stream and store it in state.data."""
        self._record_seed(state, rng)
        if state.device.type != rng.device.type:
            raise ValueError(
                "SingleEventNode rng device does not match state.device. "
                f"rng.device={rng.device}, state.device={state.device}."
            )

        times_idx = torch.randint(
            self.time_min,
            self.time_max + 1,
            (state.batch_size, 1),
            generator=rng,
            device=state.device,
        )  # [B, 1]
        times = times_idx.to(torch.float32)  # [B, 1]

        type_ids = torch.full(
            (state.batch_size, 1),
            self.type_id,
            device=state.device,
            dtype=torch.int64,
        )  # [B, 1]

        params = torch.zeros(
            (state.batch_size, 1, len(self.schema.param_names)),
            device=state.device,
            dtype=torch.float32,
        )  # [B, 1, P]
        amplitude = torch.rand(
            (state.batch_size, 1),
            generator=rng,
            device=state.device,
        )  # [B, 1]
        amplitude = self.amplitude_min + (self.amplitude_max - self.amplitude_min) * amplitude  # [B, 1]
        params[:, :, self.amplitude_index] = amplitude

        mask = torch.ones(
            (state.batch_size, 1),
            device=state.device,
            dtype=torch.bool,
        )  # [B, 1]

        events = EventBatch(
            times=times,
            type_ids=type_ids,
            params=params,
            mask=mask,
            schema=self.schema,
            meta={"seq_len": self.seq_len},
        )
        state.data[self.out_key] = events
        return state


class EventTrainNode(ProcessNode):
    """Generate a train of events with regular/irregular/missed spacing."""

    def __init__(
        self,
        *,
        seq_len: int,
        num_events: int,
        schema: EventSchema,
        mode: str,
        type_label_key: str | None,
        type_id: int | None,
        amplitude_min: float,
        amplitude_max: float,
        amplitude_param: str,
        missed_gap_factor: float,
        out_key: str,
        centers_out_key: str,
    ) -> None:
        """Initialize an event train generator."""
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if num_events <= 0:
            raise ValueError(f"num_events must be positive, got {num_events}.")
        if mode not in {"regular", "irregular", "missed_beat"}:
            raise ValueError(f"mode must be regular/irregular/missed_beat, got {mode}.")
        if amplitude_max < amplitude_min:
            raise ValueError("amplitude_max must be >= amplitude_min.")
        if missed_gap_factor <= 1:
            raise ValueError("missed_gap_factor must be > 1.")
        if (type_label_key is None) == (type_id is None):
            raise ValueError("Provide exactly one of type_label_key or type_id.")
        if not out_key or not centers_out_key:
            raise ValueError("out_key and centers_out_key must be non-empty.")

        self.seq_len = seq_len
        self.num_events = num_events
        self.schema = schema
        self.mode = mode
        self.type_label_key = type_label_key
        self.type_id = type_id
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.amplitude_index = schema.param_id(amplitude_param)
        self.missed_gap_factor = missed_gap_factor
        self.out_key = out_key
        self.centers_out_key = centers_out_key

    def _resolve_type_ids(self, state: ProcessState) -> torch.Tensor:
        """Resolve per-sample type ids from labels or a fixed id."""
        if self.type_label_key is not None:
            if self.type_label_key in state.y:
                type_idx = state.y[self.type_label_key]
            elif self.type_label_key in state.data:
                type_idx = state.data[self.type_label_key]
            else:
                raise ValueError(
                    f"type_label_key '{self.type_label_key}' not found in state."
                )
            if type_idx.ndim != 1 or type_idx.shape[0] != state.batch_size:
                raise ValueError(
                    "type_label must have shape [B]. "
                    f"Got {type_idx.shape}, batch_size={state.batch_size}."
                )
            type_ids = type_idx[:, None].expand(state.batch_size, self.num_events)  # [B, N]
            return type_ids.to(torch.int64)

        type_ids = torch.full(
            (state.batch_size, self.num_events),
            int(self.type_id),
            device=state.device,
            dtype=torch.int64,
        )  # [B, N]
        return type_ids

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Generate event centers and assemble an EventBatch."""
        self._record_seed(state, rng)

        base_interval = 1.0 / (self.num_events + 1)
        base_intervals = torch.full(
            (state.batch_size, self.num_events + 1),
            fill_value=base_interval,
            device=state.device,
            dtype=torch.float32,
        )  # [B, N+1]

        if self.mode == "regular":
            intervals = base_intervals  # [B, N+1]
        elif self.mode == "irregular":
            random_intervals = torch.rand(
                (state.batch_size, self.num_events + 1),
                generator=rng,
                device=state.device,
            )  # [B, N+1]
            random_intervals = random_intervals / random_intervals.sum(dim=1, keepdim=True)  # [B, N+1]
            intervals = random_intervals
        else:
            missed_indices = torch.randint(
                0,
                self.num_events + 1,
                (state.batch_size,),
                generator=rng,
                device=state.device,
            )  # [B]
            missed_multipliers = torch.ones(
                (state.batch_size, self.num_events + 1),
                device=state.device,
            )  # [B, N+1]
            missed_multipliers.scatter_(
                1,
                missed_indices[:, None],
                self.missed_gap_factor,
            )
            missed_intervals = base_intervals * missed_multipliers  # [B, N+1]
            missed_intervals = missed_intervals / missed_intervals.sum(dim=1, keepdim=True)  # [B, N+1]
            intervals = missed_intervals

        centers_normalized = intervals.cumsum(dim=1)[:, :-1]  # [B, N]
        phase_offset = torch.rand(
            (state.batch_size, 1),
            generator=rng,
            device=state.device,
        )  # [B, 1]
        centers_normalized = (centers_normalized + phase_offset) % 1.0  # [B, N]
        centers_normalized, _ = centers_normalized.sort(dim=1)  # [B, N]
        centers = centers_normalized * (self.seq_len - 1)  # [B, N]

        type_ids = self._resolve_type_ids(state)  # [B, N]

        amplitude = torch.rand(
            (state.batch_size, 1),
            generator=rng,
            device=state.device,
        )  # [B, 1]
        amplitude = self.amplitude_min + (self.amplitude_max - self.amplitude_min) * amplitude  # [B, 1]
        amplitude = amplitude.expand(state.batch_size, self.num_events)  # [B, N]

        params = torch.zeros(
            (state.batch_size, self.num_events, len(self.schema.param_names)),
            device=state.device,
            dtype=torch.float32,
        )  # [B, N, P]
        params[:, :, self.amplitude_index] = amplitude

        mask = torch.ones(
            (state.batch_size, self.num_events),
            device=state.device,
            dtype=torch.bool,
        )  # [B, N]

        events = EventBatch(
            times=centers,
            type_ids=type_ids,
            params=params,
            mask=mask,
            schema=self.schema,
            meta={
                "seq_len": self.seq_len,
                "mode": self.mode,
            },
        )
        state.data[self.out_key] = events
        state.data[self.centers_out_key] = centers

        spacing = (self.seq_len - 1) / (self.num_events + 1)
        state.meta["spacing_samples"] = spacing
        state.meta["phase_offset_samples"] = phase_offset
        return state


class UnionEventsNode(ProcessNode):
    """Concatenate event streams and sort by time."""

    def __init__(self, *, in_keys: list[str], out_key: str) -> None:
        """Initialize a union node with input keys."""
        super().__init__()
        if not in_keys:
            raise ValueError("in_keys must be non-empty.")
        if not out_key:
            raise ValueError("out_key must be non-empty.")
        self.in_keys = list(in_keys)
        self.out_key = out_key

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Union event streams into a single sorted EventBatch."""
        self._record_seed(state, rng)
        events_list: list[EventBatch] = []
        for key in self.in_keys:
            if key not in state.data:
                raise ValueError(f"UnionEventsNode missing key '{key}'.")
            events = state.data[key]
            if not isinstance(events, EventBatch):
                raise TypeError(
                    f"UnionEventsNode expects EventBatch at '{key}', got {type(events).__name__}."
                )
            events_list.append(events)

        schema = events_list[0].schema
        for events in events_list[1:]:
            if events.schema != schema:
                raise ValueError("UnionEventsNode schema mismatch across inputs.")

        times = torch.cat([events.times for events in events_list], dim=1)  # [B, E]
        type_ids = torch.cat([events.type_ids for events in events_list], dim=1)  # [B, E]
        params = torch.cat([events.params for events in events_list], dim=1)  # [B, E, P]
        mask = torch.cat([events.mask for events in events_list], dim=1)  # [B, E]

        sort_times = times.masked_fill(~mask, float("inf"))  # [B, E]
        indices = sort_times.argsort(dim=1)  # [B, E]
        times_sorted = torch.gather(times, 1, indices)  # [B, E]
        type_sorted = torch.gather(type_ids, 1, indices)  # [B, E]
        mask_sorted = torch.gather(mask, 1, indices)  # [B, E]
        indices_params = indices[:, :, None].expand(-1, -1, params.shape[2])  # [B, E, P]
        params_sorted = torch.gather(params, 1, indices_params)  # [B, E, P]

        state.data[self.out_key] = EventBatch(
            times=times_sorted,
            type_ids=type_sorted,
            params=params_sorted,
            mask=mask_sorted,
            schema=schema,
            meta=events_list[0].meta,
        )
        return state


class DedupeEventsNode(ProcessNode):
    """Remove events that are closer than min_dt in time."""

    def __init__(self, *, in_key: str, out_key: str, min_dt: float) -> None:
        """Initialize a de-duplication node."""
        super().__init__()
        if min_dt < 0:
            raise ValueError(f"min_dt must be non-negative, got {min_dt}.")
        if not in_key or not out_key:
            raise ValueError("in_key and out_key must be non-empty.")
        self.in_key = in_key
        self.out_key = out_key
        self.min_dt = min_dt

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Filter events by minimum temporal spacing."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"DedupeEventsNode expects EventBatch at '{self.in_key}', got {type(events).__name__}."
            )

        times = events.times  # [B, E]
        mask = events.mask  # [B, E]

        sort_times = times.masked_fill(~mask, float("inf"))  # [B, E]
        indices = sort_times.argsort(dim=1)  # [B, E]
        times_sorted = torch.gather(times, 1, indices)  # [B, E]
        mask_sorted = torch.gather(mask, 1, indices)  # [B, E]
        sort_times_sorted = torch.gather(sort_times, 1, indices)  # [B, E]

        keep = torch.ones_like(mask_sorted, dtype=torch.bool)  # [B, E]
        dt = sort_times_sorted[:, 1:] - sort_times_sorted[:, :-1]  # [B, E-1]
        keep[:, 1:] = dt >= self.min_dt
        mask_out = mask_sorted & keep  # [B, E]

        type_sorted = torch.gather(events.type_ids, 1, indices)  # [B, E]
        indices_params = indices[:, :, None].expand(-1, -1, events.params.shape[2])  # [B, E, P]
        params_sorted = torch.gather(events.params, 1, indices_params)  # [B, E, P]

        state.data[self.out_key] = EventBatch(
            times=times_sorted,
            type_ids=type_sorted,
            params=params_sorted,
            mask=mask_out,
            schema=events.schema,
            meta=events.meta,
        )
        return state


class MapTypeNode(ProcessNode):
    """Remap event type ids into a new schema."""

    def __init__(
        self,
        *,
        in_key: str,
        out_key: str,
        mapping: dict[int, int],
        output_schema: EventSchema,
    ) -> None:
        """Initialize a type remapping node."""
        super().__init__()
        if not in_key or not out_key:
            raise ValueError("in_key and out_key must be non-empty.")
        if not mapping:
            raise ValueError("mapping must be non-empty.")
        self.in_key = in_key
        self.out_key = out_key
        self.mapping = dict(mapping)
        self.output_schema = output_schema

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Apply the type-id mapping and return a new EventBatch."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"MapTypeNode expects EventBatch at '{self.in_key}', got {type(events).__name__}."
            )

        max_key = max(self.mapping.keys())
        lookup = torch.full(
            (max_key + 1,),
            -1,
            device=events.type_ids.device,
            dtype=torch.int64,
        )  # [T]
        for old_id, new_id in self.mapping.items():
            if new_id < 0 or new_id >= len(self.output_schema.type_names):
                raise ValueError(
                    "mapping produces type_id outside output_schema. "
                    f"Got {new_id}, num_types={len(self.output_schema.type_names)}."
                )
            lookup[int(old_id)] = int(new_id)

        remapped = lookup[events.type_ids]  # [B, E]
        if torch.any(remapped < 0):
            raise ValueError("mapping does not cover all type_ids in events.")

        state.data[self.out_key] = EventBatch(
            times=events.times,
            type_ids=remapped,
            params=events.params,
            mask=events.mask,
            schema=self.output_schema,
            meta=events.meta,
        )
        return state


class TimeShiftNode(ProcessNode):
    """Shift event times by a constant offset."""

    def __init__(self, *, in_key: str, out_key: str, shift: float) -> None:
        """Initialize a time shift node."""
        super().__init__()
        if not in_key or not out_key:
            raise ValueError("in_key and out_key must be non-empty.")
        self.in_key = in_key
        self.out_key = out_key
        self.shift = shift

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Apply a constant time shift to events."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"TimeShiftNode expects EventBatch at '{self.in_key}', got {type(events).__name__}."
            )

        shifted = events.times + self.shift  # [B, E]
        state.data[self.out_key] = EventBatch(
            times=shifted,
            type_ids=events.type_ids,
            params=events.params,
            mask=events.mask,
            schema=events.schema,
            meta=events.meta,
        )
        return state


class TimeJitterNode(ProcessNode):
    """Add Gaussian jitter to event times."""

    def __init__(self, *, in_key: str, out_key: str, jitter_std: float) -> None:
        """Initialize a time jitter node."""
        super().__init__()
        if jitter_std < 0:
            raise ValueError(f"jitter_std must be non-negative, got {jitter_std}.")
        if not in_key or not out_key:
            raise ValueError("in_key and out_key must be non-empty.")
        self.in_key = in_key
        self.out_key = out_key
        self.jitter_std = jitter_std

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Apply time jitter to events."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"TimeJitterNode expects EventBatch at '{self.in_key}', got {type(events).__name__}."
            )

        noise = torch.randn(
            events.times.shape,
            generator=rng,
            device=events.times.device,
        )  # [B, E]
        jittered = events.times + noise * self.jitter_std  # [B, E]
        state.data[self.out_key] = EventBatch(
            times=jittered,
            type_ids=events.type_ids,
            params=events.params,
            mask=events.mask,
            schema=events.schema,
            meta=events.meta,
        )
        return state


class GateEventsNode(ProcessNode):
    """Apply a boolean mask to an EventBatch."""

    def __init__(self, *, in_key: str, mask_key: str, out_key: str) -> None:
        """Initialize a gate node that masks events."""
        super().__init__()
        if not in_key or not mask_key or not out_key:
            raise ValueError("in_key, mask_key, and out_key must be non-empty.")
        self.in_key = in_key
        self.mask_key = mask_key
        self.out_key = out_key

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Filter events using a boolean mask tensor."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"GateEventsNode expects EventBatch at '{self.in_key}', got {type(events).__name__}."
            )
        gate = state.data.get(self.mask_key)
        if not isinstance(gate, torch.Tensor):
            raise TypeError(
                f"GateEventsNode expects tensor mask at '{self.mask_key}', got {type(gate).__name__}."
            )
        if gate.shape != events.mask.shape:
            raise ValueError(
                "GateEventsNode mask shape mismatch. "
                f"gate.shape={gate.shape}, events.mask.shape={events.mask.shape}."
            )
        if gate.dtype != torch.bool:
            raise ValueError(f"GateEventsNode mask must be bool, got {gate.dtype}.")

        mask = events.mask & gate  # [B, E]
        state.data[self.out_key] = EventBatch(
            times=events.times,
            type_ids=events.type_ids,
            params=events.params,
            mask=mask,
            schema=events.schema,
            meta=events.meta,
        )
        return state
