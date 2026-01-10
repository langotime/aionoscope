from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..core.events import EventBatch, events_select
from ..core.rng import rng_make_generator, rng_split
from ..core.types import LatentState
from .base import Process


@dataclass
class ProcessState:
    """Mutable container passed between process ops."""

    batch_size: int
    device: torch.device
    data: dict[str, Any]
    y: dict[str, torch.Tensor]
    meta: dict[str, Any]


class ProcessOp(nn.Module):
    """Base class for process graph operations."""

    def _record_seed(self, state: ProcessState, rng: torch.Generator) -> None:
        """Append the current op seed to the trace."""
        trace = state.meta.setdefault("trace_seeds", [])
        trace.append(
            {
                "op": self.__class__.__name__,
                "seed": int(rng.initial_seed()),
            }
        )

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Apply this op to the process state."""
        raise NotImplementedError


GraphSpec = ProcessOp | list[ProcessOp]


def _normalize_graphspec(graph: GraphSpec) -> ProcessOp:
    """Normalize a graph spec into a concrete ProcessOp."""
    if isinstance(graph, ProcessOp):
        return graph
    if isinstance(graph, list):
        return Seq(graph)
    raise TypeError(f"GraphSpec must be ProcessOp or list[ProcessOp], got {type(graph).__name__}.")


def _clone_state(state: ProcessState) -> ProcessState:
    """Shallow-copy ProcessState with new data/y dicts."""
    return ProcessState(
        batch_size=state.batch_size,
        device=state.device,
        data=dict(state.data),
        y=dict(state.y),
        meta=state.meta,
    )


class ProcessNode(ProcessOp):
    """Leaf op that reads/writes ProcessState fields."""

    pass


class Seq(ProcessOp):
    """Sequential composition of ops."""

    def __init__(self, ops: list[ProcessOp]) -> None:
        """Create a sequential op from a list of ops."""
        super().__init__()
        if not ops:
            raise ValueError("Seq requires at least one op.")
        self.ops = nn.ModuleList([_normalize_graphspec(op) for op in ops])

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Run ops in order with split RNGs."""
        self._record_seed(state, rng)
        child_generators = rng_split(
            rng=rng,
            num_children=len(self.ops),
            device=state.device,
        )
        out = state
        for op, child_rng in zip(self.ops, child_generators, strict=True):
            out = op(out, rng=child_rng)
        return out


class Switch(ProcessOp):
    """Branch per-sample based on a label key."""

    def __init__(
        self,
        *,
        label_key: str,
        cases: dict[int, GraphSpec],
        default: GraphSpec | None = None,
    ) -> None:
        """Create a switch op based on a label key."""
        super().__init__()
        if not cases:
            raise ValueError("Switch requires at least one case.")
        self.label_key = label_key
        self.cases = {key: _normalize_graphspec(op) for key, op in cases.items()}
        self.default = _normalize_graphspec(default) if default is not None else None

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Execute branches and merge outputs by label mask."""
        self._record_seed(state, rng)

        if self.label_key in state.y:
            label = state.y[self.label_key]
        elif self.label_key in state.data:
            label = state.data[self.label_key]
        else:
            raise ValueError(
                f"Switch label_key '{self.label_key}' is missing from state.y and state.data."
            )
        if label.ndim != 1 or label.shape[0] != state.batch_size:
            raise ValueError(
                "Switch label must have shape [B]. "
                f"Got {label.shape}, batch_size={state.batch_size}."
            )

        label = label.to(torch.int64)  # [B]
        case_ids = list(self.cases.keys())
        masks = {case_id: label == case_id for case_id in case_ids}  # [B]
        covered = torch.zeros_like(label, dtype=torch.bool)  # [B]
        for mask in masks.values():
            covered = covered | mask

        if self.default is None and not torch.all(covered):
            missing = torch.nonzero(~covered, as_tuple=False).flatten()  # [M]
            raise ValueError(f"Switch has no default for labels at indices {missing.tolist()}.")

        child_generators = rng_split(
            rng=rng,
            num_children=len(self.cases) + (1 if self.default is not None else 0),
            device=state.device,
        )

        candidates: dict[int, ProcessState] = {}
        for (case_id, op), child_rng in zip(
            self.cases.items(),
            child_generators[: len(self.cases)],
            strict=True,
        ):
            branch_state = _clone_state(state)
            candidate = op(branch_state, rng=child_rng)
            for key in state.data:
                if key in candidate.data and candidate.data[key] is not state.data[key]:
                    raise ValueError(
                        f"Switch branch overwrote existing key '{key}'. "
                        "Write new keys or move the node outside Switch."
                    )
            candidates[case_id] = candidate

        base_state = None
        if self.default is not None:
            branch_state = _clone_state(state)
            base_state = self.default(
                branch_state,
                rng=child_generators[-1],
            )
        else:
            first_case_id = case_ids[0]
            base_state = candidates[first_case_id]

        base_keys = set(state.data.keys())
        new_keys = set(base_state.data.keys()) - base_keys
        for case_id, candidate in candidates.items():
            candidate_new_keys = set(candidate.data.keys()) - base_keys
            if candidate_new_keys != new_keys:
                raise ValueError(
                    "Switch branches must write the same new keys. "
                    f"Got {sorted(candidate_new_keys)} vs {sorted(new_keys)}."
                )

        merged_data = dict(state.data)
        for key in new_keys:
            base_val = base_state.data[key]
            merged_val = base_val
            for case_id, candidate in candidates.items():
                mask = masks[case_id]
                candidate_val = candidate.data[key]
                if isinstance(merged_val, torch.Tensor):
                    if candidate_val.shape != merged_val.shape:
                        raise ValueError(
                            f"Switch candidate '{key}' shape mismatch. "
                            f"{candidate_val.shape} vs {merged_val.shape}."
                        )
                    updated = merged_val.clone()
                    updated[mask] = candidate_val[mask]
                    merged_val = updated
                elif isinstance(merged_val, EventBatch):
                    merged_val = events_select(merged_val, candidate_val, mask)
                else:
                    raise TypeError(
                        f"Switch does not support merging type {type(merged_val).__name__}."
                    )
            merged_data[key] = merged_val

        return ProcessState(
            batch_size=state.batch_size,
            device=state.device,
            data=merged_data,
            y=dict(state.y),
            meta=state.meta,
        )


class Parallel(ProcessOp):
    """Execute multiple branches and namespace their outputs."""

    def __init__(self, *, branches: dict[str, GraphSpec]) -> None:
        """Create a parallel op with named branches."""
        super().__init__()
        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        self.branches = {name: _normalize_graphspec(op) for name, op in branches.items()}

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Run branches in parallel and prefix their new keys."""
        self._record_seed(state, rng)

        child_generators = rng_split(
            rng=rng,
            num_children=len(self.branches),
            device=state.device,
        )

        base_keys = set(state.data.keys())
        merged_data = dict(state.data)

        for (name, op), child_rng in zip(
            self.branches.items(),
            child_generators,
            strict=True,
        ):
            branch_state = _clone_state(state)
            candidate = op(branch_state, rng=child_rng)
            for key in state.data:
                if key in candidate.data and candidate.data[key] is not state.data[key]:
                    raise ValueError(
                        f"Parallel branch overwrote existing key '{key}'. "
                        "Write new keys or use Scope."
                    )
            new_keys = set(candidate.data.keys()) - base_keys
            for key in new_keys:
                merged_data[f"{name}.{key}"] = candidate.data[key]

        return ProcessState(
            batch_size=state.batch_size,
            device=state.device,
            data=merged_data,
            y=dict(state.y),
            meta=state.meta,
        )


class Scope(ProcessOp):
    """Prefix writes from a subgraph to avoid key collisions."""

    def __init__(self, *, prefix: str, op: GraphSpec) -> None:
        """Create a scope that prefixes new keys from a subgraph."""
        super().__init__()
        if not prefix:
            raise ValueError("Scope prefix must be non-empty.")
        self.prefix = prefix
        self.op = _normalize_graphspec(op)

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Run a subgraph and add a prefix to its new outputs."""
        self._record_seed(state, rng)
        branch_state = _clone_state(state)
        candidate = self.op(branch_state, rng=rng)
        base_keys = set(state.data.keys())
        new_keys = set(candidate.data.keys()) - base_keys
        merged_data = dict(state.data)
        for key in new_keys:
            merged_data[f"{self.prefix}.{key}"] = candidate.data[key]
        return ProcessState(
            batch_size=state.batch_size,
            device=state.device,
            data=merged_data,
            y=dict(state.y),
            meta=state.meta,
        )


class ProcessGraph(Process):
    """Process backed by a graph of ProcessOps."""

    def __init__(
        self,
        *,
        graph: GraphSpec,
        outputs: set[str],
        name: str,
        base_meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the graph-backed process."""
        super().__init__()
        if not name:
            raise ValueError("ProcessGraph name must be non-empty.")
        if not outputs:
            raise ValueError("ProcessGraph outputs must be non-empty.")
        allowed = {"latent", "events", "centers"}
        if not outputs.issubset(allowed):
            raise ValueError(f"ProcessGraph outputs must be in {sorted(allowed)}.")

        self.graph = _normalize_graphspec(graph)
        self.outputs = set(outputs)
        self.name = name
        self.base_meta = dict(base_meta or {})

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        """Run the graph and return a LatentState."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        generator, seed, _ = rng_make_generator(rng=rng, device=device)
        meta = {"process": self.name, "seed": seed, "trace_seeds": []}
        meta.update(self.base_meta)

        state = ProcessState(
            batch_size=batch_size,
            device=device,
            data={},
            y={},
            meta=meta,
        )
        state = self.graph(state, rng=generator)

        data = state.data
        if "latent" in self.outputs and "latent" not in data:
            raise ValueError("ProcessGraph outputs require 'latent' but it is missing.")
        if "events" in self.outputs and "events" not in data:
            raise ValueError("ProcessGraph outputs require 'events' but it is missing.")
        if "centers" in self.outputs and "centers" not in data:
            raise ValueError("ProcessGraph outputs require 'centers' but it is missing.")

        centers = data.get(
            "centers",
            torch.empty((batch_size, 0), device=device),
        )  # [B, N]
        latent = data.get("latent")
        events = data.get("events")

        return LatentState(
            centers=centers,
            latent=latent,
            events=events,
            y=state.y,
            meta=state.meta,
        )


class ProcessChain(ProcessGraph):
    """Linear process graph convenience wrapper."""

    def __init__(
        self,
        *,
        nodes: list[ProcessOp],
        outputs: set[str],
        name: str,
        base_meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a linear process graph."""
        super().__init__(
            graph=Seq(nodes),
            outputs=outputs,
            name=name,
            base_meta=base_meta,
        )
