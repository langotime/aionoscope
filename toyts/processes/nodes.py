from __future__ import annotations

import torch

from ..core.events import EventBatch, EventSchema
from ..core.samplers import (
    ConstantSampler,
    Sampler,
    SamplerLike,
    WeightedPermutationSampler,
    sampler_from_value,
    sampler_sample,
)
from ..core.utils import SAMPLES_PREFIX
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


class SampleLabelNode(ProcessNode):
    """Sample a single categorical label into state.y.

    Supports either uniform sampling over class_names or an explicit sampler over
    class indices.
    """

    def __init__(
        self,
        *,
        label_key: str,
        class_names: list[str],
        sampler: SamplerLike[int] | None = None,
    ) -> None:
        super().__init__()
        if not label_key:
            raise ValueError("label_key must be non-empty.")
        if not class_names:
            raise ValueError("class_names must be non-empty.")
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique.")

        self.label_key = label_key
        self.class_names = list(class_names)
        self.sampler = None if sampler is None else sampler_from_value(sampler, name=label_key)

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        self._record_seed(state, rng)
        num_classes = len(self.class_names)

        if self.sampler is None:
            indices = torch.randint(
                0,
                num_classes,
                (state.batch_size,),
                generator=rng,
                device=state.device,
            )  # [B]
        else:
            indices = sampler_sample(
                sampler=self.sampler,
                shape=(state.batch_size,),
                rng=rng,
                device=state.device,
                dtype=torch.int64,
                name=self.label_key,
            )  # [B]
            if not torch.all((indices >= 0) & (indices < num_classes)):
                min_value = int(indices.min().item())
                max_value = int(indices.max().item())
                raise ValueError(
                    f"SampleLabelNode sampled '{self.label_key}' out of range. "
                    f"Expected 0 <= idx < {num_classes}, got min={min_value}, max={max_value}."
                )

        state.y[self.label_key] = indices.to(torch.int64)

        label_names = state.meta.setdefault("label_names", {})
        if self.label_key in label_names and label_names[self.label_key] != self.class_names:
            raise ValueError(
                f"Label '{self.label_key}' already exists in meta with different classes. "
                f"Existing={label_names[self.label_key]}, new={self.class_names}."
            )
        label_names[self.label_key] = self.class_names
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


class EnableComponentsNode(ProcessNode):
    """Sample per-sample component enable masks.

    Writes `state.meta["enabled"][key] = bool[B]` for each component key. When
    all samples have `num_enabled == 1`, also writes `state.y["component_id"] = int64[B]`
    with label names in `state.meta["label_names"]["component_id"]`.

    For imbalanced class sampling, provide `component_id` (a Sampler over class
    indices) together with `num_enabled=1`.

    For imbalanced k-hot mixtures, provide `component_order` (a Sampler producing
    permutations `[B, N]` of component indices), and this node enables the first
    `k` entries per sample.
    """

    def __init__(
        self,
        *,
        component_keys: list[str],
        num_enabled: SamplerLike[int],
        component_id: SamplerLike[int] | None = None,
        component_order: SamplerLike[int] | None = None,
    ) -> None:
        """Initialize enabled-mask sampling.

        Args:
            component_keys: Unique component identifiers.
            num_enabled: Sampler for the number of enabled components per sample (k-hot size).
            component_id: Optional Sampler for component class indices. Only supported when
                num_enabled is a constant 1.
            component_order: Optional Sampler for component index permutations (int64 [B, N]).
                When provided, components are selected by taking the first `k` indices per sample.
        """
        super().__init__()
        if not component_keys:
            raise ValueError("component_keys must be non-empty.")
        if any(not key for key in component_keys):
            raise ValueError("component_keys must contain only non-empty strings.")
        if len(set(component_keys)) != len(component_keys):
            raise ValueError("component_keys must be unique.")

        self.component_keys = list(component_keys)
        self.num_enabled_sampler = sampler_from_value(num_enabled, name="num_enabled")
        if isinstance(self.num_enabled_sampler, ConstantSampler):
            value = self.num_enabled_sampler.value
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "EnableComponentsNode num_enabled must be an integer. "
                    f"Got constant {value!r} ({type(value).__name__})."
                )
            if value <= 0:
                raise ValueError(f"num_enabled must be positive, got {value}.")
            if value > len(component_keys):
                raise ValueError(
                    "num_enabled must be <= len(component_keys). "
                    f"Got num_enabled={value}, len(component_keys)={len(component_keys)}."
                )

        if component_id is not None and component_order is not None:
            raise ValueError("EnableComponentsNode does not allow both component_id and component_order.")

        self.component_id_sampler: Sampler | None
        if component_id is None:
            self.component_id_sampler = None
        else:
            if (
                not isinstance(self.num_enabled_sampler, ConstantSampler)
                or self.num_enabled_sampler.value != 1
            ):
                raise ValueError(
                    "EnableComponentsNode component_id is only supported when num_enabled is a "
                    "constant 1."
                )
            component_id_sampler = sampler_from_value(component_id, name="component_id")
            if isinstance(component_id_sampler, ConstantSampler):
                value = component_id_sampler.value
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        "EnableComponentsNode component_id must be an integer. "
                        f"Got constant {value!r} ({type(value).__name__})."
                    )
                if value < 0 or value >= len(component_keys):
                    raise ValueError(
                        "EnableComponentsNode component_id must be in "
                        f"[0, {len(component_keys)}). Got {value}."
                    )
            self.component_id_sampler = component_id_sampler

        self.component_order_sampler: Sampler | None
        if component_order is None:
            self.component_order_sampler = None
        else:
            self.component_order_sampler = sampler_from_value(component_order, name="component_order")

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Sample k-hot masks and store them in process meta."""
        self._record_seed(state, rng)
        if state.device.type != rng.device.type:
            raise ValueError(
                "EnableComponentsNode rng device does not match state.device. "
                f"rng.device={rng.device}, state.device={state.device}."
            )

        batch_size = state.batch_size
        num_components = len(self.component_keys)
        k = sampler_sample(
            sampler=self.num_enabled_sampler,
            shape=(batch_size,),
            rng=rng,
            device=state.device,
            dtype=torch.int64,
            name="num_enabled",
        )  # [B]
        if not torch.all((k >= 1) & (k <= num_components)):
            k_min = int(k.min().item())
            k_max = int(k.max().item())
            raise ValueError(
                "EnableComponentsNode sampled num_enabled out of range. "
                f"Expected 1 <= num_enabled <= {num_components}. Got min={k_min}, max={k_max}."
            )

        if self.component_id_sampler is not None:
            if not torch.all(k == 1):
                raise ValueError(
                    "EnableComponentsNode component_id is only supported when all samples have "
                    "num_enabled == 1."
                )
            component_id = sampler_sample(
                sampler=self.component_id_sampler,
                shape=(batch_size,),
                rng=rng,
                device=state.device,
                dtype=torch.int64,
                name="component_id",
            )  # [B]
            if not torch.all((component_id >= 0) & (component_id < num_components)):
                id_min = int(component_id.min().item())
                id_max = int(component_id.max().item())
                raise ValueError(
                    "EnableComponentsNode sampled component_id out of range. "
                    f"Expected 0 <= component_id < {num_components}. Got min={id_min}, max={id_max}."
                )

            enabled_matrix = torch.zeros(
                (batch_size, num_components),
                device=state.device,
                dtype=torch.bool,
            )  # [B, N]
            scatter_value = torch.ones(
                (batch_size, 1),
                device=state.device,
                dtype=torch.bool,
            )  # [B, 1]
            enabled_matrix.scatter_(
                1,
                component_id[:, None],
                scatter_value,
            )

            state.y["component_count"] = k  # [B]
            state.y["component_id"] = component_id.to(torch.int64)  # [B]

            label_names = state.meta.setdefault("label_names", {})
            if "component_id" in label_names and label_names["component_id"] != self.component_keys:
                raise ValueError(
                    "EnableComponentsNode found existing label_names['component_id'] "
                    "with different classes."
                )
            label_names["component_id"] = list(self.component_keys)

            enabled_spec = state.meta.setdefault("enabled_spec", {})
            component_id_spec = self.component_id_sampler.spec()
            if "component_id" in enabled_spec and enabled_spec["component_id"] != component_id_spec:
                raise ValueError(
                    "EnableComponentsNode found existing enabled_spec['component_id'] "
                    "with different spec."
                )
            enabled_spec["component_id"] = component_id_spec
        else:
            if isinstance(self.component_order_sampler, WeightedPermutationSampler):
                positive = sum(prob > 0 for prob in self.component_order_sampler.probs)
                k_max = int(k.max().item())
                if k_max > positive:
                    raise ValueError(
                        "EnableComponentsNode num_enabled cannot exceed the number of positive-prob "
                        "components when using WeightedPermutationSampler. "
                        f"Got max(num_enabled)={k_max}, num_positive_probs={positive}."
                    )

            if self.component_order_sampler is not None:
                order = sampler_sample(
                    sampler=self.component_order_sampler,
                    shape=(batch_size, num_components),
                    rng=rng,
                    device=state.device,
                    dtype=torch.int64,
                    name="component_order",
                )  # [B, N]
                if not torch.all((order >= 0) & (order < num_components)):
                    order_min = int(order.min().item())
                    order_max = int(order.max().item())
                    raise ValueError(
                        "EnableComponentsNode sampled component_order out of range. "
                        f"Expected 0 <= component_order < {num_components}. "
                        f"Got min={order_min}, max={order_max}."
                    )
                expected = torch.arange(num_components, device=state.device, dtype=torch.int64)  # [N]
                sorted_order = order.sort(dim=1).values  # [B, N]
                if not torch.all(sorted_order == expected):
                    raise ValueError(
                        "EnableComponentsNode component_order must be a per-sample permutation of "
                        f"[0, {num_components})."
                    )

                enabled_spec = state.meta.setdefault("enabled_spec", {})
                order_spec = self.component_order_sampler.spec()
                if "component_order" in enabled_spec and enabled_spec["component_order"] != order_spec:
                    raise ValueError(
                        "EnableComponentsNode found existing enabled_spec['component_order'] "
                        "with different spec."
                    )
                enabled_spec["component_order"] = order_spec
            else:
                scores = torch.rand(
                    (batch_size, num_components),
                    generator=rng,
                    device=state.device,
                    dtype=torch.float32,
                )  # [B, N]
                order = scores.argsort(dim=1, descending=True)  # [B, N]
            rank = torch.arange(num_components, device=state.device, dtype=torch.int64)[None, :]  # [1, N]
            keep = rank < k[:, None]  # [B, N]

            enabled_matrix = torch.zeros(
                (batch_size, num_components),
                device=state.device,
                dtype=torch.bool,
            )  # [B, N]
            enabled_matrix.scatter_(1, order, keep)

            state.y["component_count"] = k  # [B]
            if torch.all(k == 1):
                state.y["component_id"] = order[:, 0].to(torch.int64)  # [B]
                label_names = state.meta.setdefault("label_names", {})
                if "component_id" in label_names and label_names["component_id"] != self.component_keys:
                    raise ValueError(
                        "EnableComponentsNode found existing label_names['component_id'] "
                        "with different classes."
                    )
                label_names["component_id"] = list(self.component_keys)

        enabled = state.meta.get("enabled")
        if enabled is None:
            enabled = {}
            state.meta["enabled"] = enabled
        if not isinstance(enabled, dict):
            raise ValueError(
                "EnableComponentsNode requires state.meta['enabled'] to be a dict. "
                f"Got {type(enabled).__name__}."
            )
        for idx, key in enumerate(self.component_keys):
            if key in enabled:
                raise ValueError(
                    f"EnableComponentsNode attempted to overwrite enabled['{key}']."
                )
            enabled[key] = enabled_matrix[:, idx]  # [B]

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
        amplitude: SamplerLike[float],
        amplitude_param: str,
        extra_params: dict[str, SamplerLike[float]] | None = None,
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
        if not out_key:
            raise ValueError("out_key must be non-empty.")

        self.seq_len = seq_len
        self.schema = schema
        self.type_id = schema.type_id(type_name)
        self.time_min = time_min
        self.time_max = time_max
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.amplitude_index = schema.param_id(amplitude_param)
        if extra_params is not None:
            if amplitude_param in extra_params:
                raise ValueError("extra_params must not override amplitude_param.")
            if any(not name for name in extra_params):
                raise ValueError("extra_params keys must be non-empty.")
            self.extra_param_samplers = {
                name: sampler_from_value(value, name=name) for name, value in extra_params.items()
            }
            for name in self.extra_param_samplers:
                if name not in schema.param_names:
                    raise ValueError(
                        f"extra_params includes '{name}' which is not in schema.param_names={schema.param_names}."
                    )
        else:
            self.extra_param_samplers = {}
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
        amplitude_base = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(state.batch_size,),
            rng=rng,
            device=state.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        amplitude = amplitude_base[:, None]  # [B, 1]
        params[:, :, self.amplitude_index] = amplitude
        for name, sampler in self.extra_param_samplers.items():
            index = self.schema.param_id(name)
            values = sampler_sample(
                sampler=sampler,
                shape=(state.batch_size,),
                rng=rng,
                device=state.device,
                dtype=torch.float32,
                name=name,
            )  # [B]
            params[:, :, index] = values[:, None]

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

        samples_base = f"{SAMPLES_PREFIX}/SingleEventNode:{self.out_key}"
        state.data[f"{samples_base}/time_idx"] = times_idx[:, 0]  # [B]
        state.data[f"{samples_base}/amplitude"] = amplitude_base  # [B]
        for name, sampler in self.extra_param_samplers.items():
            index = self.schema.param_id(name)
            state.data[f"{samples_base}/{name}"] = params[:, 0, index]  # [B]
        return state


class GateEventsByEnabledNode(ProcessNode):
    """Gate event streams using a per-sample enabled mask from process meta."""

    def __init__(self, *, in_key: str, enabled_key: str, out_key: str) -> None:
        """Initialize a gate op.

        Args:
            in_key: Key in state.data containing an EventBatch.
            enabled_key: Key in state.meta["enabled"] containing bool[B].
            out_key: Key to write the gated EventBatch to.
        """
        super().__init__()
        if not in_key or not enabled_key or not out_key:
            raise ValueError("in_key, enabled_key, and out_key must be non-empty.")
        self.in_key = in_key
        self.enabled_key = enabled_key
        self.out_key = out_key

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Apply per-sample gating to an EventBatch mask."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"GateEventsByEnabledNode expects EventBatch at '{self.in_key}', "
                f"got {type(events).__name__}."
            )

        enabled = state.meta.get("enabled")
        if not isinstance(enabled, dict):
            raise ValueError(
                "GateEventsByEnabledNode requires state.meta['enabled'] to be a dict. "
                f"Got {type(enabled).__name__}."
            )
        enabled_mask = enabled.get(self.enabled_key)
        if not isinstance(enabled_mask, torch.Tensor):
            raise ValueError(
                "GateEventsByEnabledNode enabled mask must be a torch.Tensor. "
                f"Got {type(enabled_mask).__name__}."
            )
        if enabled_mask.dtype != torch.bool:
            raise ValueError(
                f"GateEventsByEnabledNode enabled mask must be bool, got {enabled_mask.dtype}."
            )
        if enabled_mask.shape != (state.batch_size,):
            raise ValueError(
                "GateEventsByEnabledNode enabled mask must have shape [B]. "
                f"Got {enabled_mask.shape}, batch_size={state.batch_size}."
            )
        if enabled_mask.device != events.mask.device:
            raise ValueError(
                "GateEventsByEnabledNode enabled mask device mismatch. "
                f"mask.device={enabled_mask.device}, events.device={events.mask.device}."
            )

        gate = enabled_mask[:, None].expand(-1, events.mask.shape[1])  # [B, E]
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


class EventTrainNode(ProcessNode):
    """Generate a train of events with regular/irregular/missed spacing."""

    def __init__(
        self,
        *,
        seq_len: int,
        num_events: int | str,
        schema: EventSchema,
        mode: str,
        type_label_key: str | None,
        type_id: int | None,
        amplitude: SamplerLike[float],
        amplitude_param: str,
        missed_gap_factor: SamplerLike[float],
        out_key: str,
        centers_out_key: str,
    ) -> None:
        """Initialize an event train generator."""
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if isinstance(num_events, int) and num_events <= 0:
            raise ValueError(f"num_events must be positive, got {num_events}.")
        if isinstance(num_events, str) and not num_events:
            raise ValueError("num_events key must be non-empty.")
        if mode not in {"regular", "irregular", "missed_beat"}:
            raise ValueError(f"mode must be regular/irregular/missed_beat, got {mode}.")
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
        self.amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        self.amplitude_index = schema.param_id(amplitude_param)
        self.missed_gap_factor_sampler = sampler_from_value(
            missed_gap_factor, name="missed_gap_factor"
        )
        self.out_key = out_key
        self.centers_out_key = centers_out_key

    def _resolve_num_events(self, state: ProcessState) -> int:
        """Resolve num_events from a constant or state key."""
        if isinstance(self.num_events, str):
            if self.num_events not in state.data:
                raise ValueError(f"num_events key '{self.num_events}' not found in state.")
            value = state.data[self.num_events]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        "num_events must be a scalar when provided as a tensor. "
                        f"Got shape {value.shape}."
                    )
                raw_value = value.item()
            elif isinstance(value, (int, float)):
                raw_value = value
            else:
                raise TypeError(
                    "num_events must be an int or 0-dim tensor when provided by key. "
                    f"Got {type(value).__name__}."
                )
        else:
            raw_value = self.num_events

        if isinstance(raw_value, float) and not raw_value.is_integer():
            raise ValueError(f"num_events must be an integer, got {raw_value}.")
        num_events = int(raw_value)
        if num_events <= 0:
            raise ValueError(f"num_events must be positive, got {num_events}.")
        return num_events

    def _resolve_type_ids(self, state: ProcessState, num_events: int) -> torch.Tensor:
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
            type_ids = type_idx[:, None].expand(state.batch_size, num_events)  # [B, N]
            return type_ids.to(torch.int64)

        type_ids = torch.full(
            (state.batch_size, num_events),
            int(self.type_id),
            device=state.device,
            dtype=torch.int64,
        )  # [B, N]
        return type_ids

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Generate event centers and assemble an EventBatch."""
        self._record_seed(state, rng)

        num_events = self._resolve_num_events(state)

        base_interval = 1.0 / (num_events + 1)
        base_intervals = torch.full(
            (state.batch_size, num_events + 1),
            fill_value=base_interval,
            device=state.device,
            dtype=torch.float32,
        )  # [B, N+1]

        missed_indices = torch.full(
            (state.batch_size,),
            -1,
            device=state.device,
            dtype=torch.int64,
        )  # [B]

        if self.mode == "regular":
            intervals = base_intervals  # [B, N+1]
        elif self.mode == "irregular":
            random_intervals = torch.rand(
                (state.batch_size, num_events + 1),
                generator=rng,
                device=state.device,
            )  # [B, N+1]
            random_intervals = random_intervals / random_intervals.sum(dim=1, keepdim=True)  # [B, N+1]
            intervals = random_intervals
        else:
            missed_gap_factor = sampler_sample(
                sampler=self.missed_gap_factor_sampler,
                shape=(state.batch_size,),
                rng=rng,
                device=state.device,
                dtype=torch.float32,
                name="missed_gap_factor",
            )  # [B]
            if torch.any(missed_gap_factor <= 1):
                raise ValueError("missed_gap_factor must be > 1 for all samples.")
            missed_indices = torch.randint(
                0,
                num_events + 1,
                (state.batch_size,),
                generator=rng,
                device=state.device,
            )  # [B]
            missed_multipliers = torch.ones(
                (state.batch_size, num_events + 1),
                device=state.device,
            )  # [B, N+1]
            missed_multipliers.scatter_(
                1,
                missed_indices[:, None],
                missed_gap_factor[:, None],
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

        type_ids = self._resolve_type_ids(state, num_events)  # [B, N]

        amplitude_base = sampler_sample(
            sampler=self.amplitude_sampler,
            shape=(state.batch_size,),
            rng=rng,
            device=state.device,
            dtype=torch.float32,
            name="amplitude",
        )  # [B]
        amplitude = amplitude_base[:, None].expand(state.batch_size, num_events)  # [B, N]

        params = torch.zeros(
            (state.batch_size, num_events, len(self.schema.param_names)),
            device=state.device,
            dtype=torch.float32,
        )  # [B, N, P]
        params[:, :, self.amplitude_index] = amplitude

        mask = torch.ones(
            (state.batch_size, num_events),
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

        if self.mode != "missed_beat":
            missed_gap_factor = sampler_sample(
                sampler=self.missed_gap_factor_sampler,
                shape=(state.batch_size,),
                rng=rng,
                device=state.device,
                dtype=torch.float32,
                name="missed_gap_factor",
            )  # [B]
            if torch.any(missed_gap_factor <= 1):
                raise ValueError("missed_gap_factor must be > 1 for all samples.")

        samples_base = f"{SAMPLES_PREFIX}/EventTrainNode:{self.out_key}"
        state.data[f"{samples_base}/intervals"] = intervals
        state.data[f"{samples_base}/missed_indices"] = missed_indices
        state.data[f"{samples_base}/phase_offset"] = phase_offset
        state.data[f"{samples_base}/missed_gap_factor"] = missed_gap_factor
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

    def __init__(self, *, in_key: str, out_key: str, jitter_std: SamplerLike[float]) -> None:
        """Initialize a time jitter node."""
        super().__init__()
        if not in_key or not out_key:
            raise ValueError("in_key and out_key must be non-empty.")
        self.in_key = in_key
        self.out_key = out_key
        self.jitter_std_sampler = sampler_from_value(jitter_std, name="jitter_std")

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Apply time jitter to events."""
        self._record_seed(state, rng)
        events = state.data.get(self.in_key)
        if not isinstance(events, EventBatch):
            raise TypeError(
                f"TimeJitterNode expects EventBatch at '{self.in_key}', got {type(events).__name__}."
            )

        jitter_std = sampler_sample(
            sampler=self.jitter_std_sampler,
            shape=(state.batch_size,),
            rng=rng,
            device=events.times.device,
            dtype=torch.float32,
            name="jitter_std",
        )  # [B]
        if torch.any(jitter_std < 0):
            raise ValueError("jitter_std must be non-negative for all samples.")
        noise = torch.randn(
            events.times.shape,
            generator=rng,
            device=events.times.device,
        )  # [B, E]
        time_jitter = noise * jitter_std[:, None]  # [B, E]
        jittered = events.times + time_jitter  # [B, E]
        state.data[self.out_key] = EventBatch(
            times=jittered,
            type_ids=events.type_ids,
            params=events.params,
            mask=events.mask,
            schema=events.schema,
            meta=events.meta,
        )
        samples_base = f"{SAMPLES_PREFIX}/TimeJitterNode:{self.out_key}"
        state.data[f"{samples_base}/time_jitter"] = time_jitter
        state.data[f"{samples_base}/jitter_std"] = jitter_std
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
