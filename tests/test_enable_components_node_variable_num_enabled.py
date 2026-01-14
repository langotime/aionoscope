from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from toyts import EnableComponentsNode, Sampler
from toyts.processes.graph import ProcessState


@dataclass(frozen=True)
class _FixedIntSampler(Sampler):
    values: list[int]

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if shape != (len(self.values),):
            raise ValueError(f"_FixedIntSampler expected shape {(len(self.values),)}, got {shape}.")
        if dtype is not torch.int64:
            raise ValueError(f"_FixedIntSampler requires dtype=torch.int64, got {dtype}.")
        return torch.tensor(self.values, device=device, dtype=dtype)  # [B]

    def spec(self) -> dict[str, Any]:
        return {"kind": "fixed_int", "values": list(self.values)}


def test_enable_components_node_variable_num_enabled_per_sample() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2", "c3", "c4"]
    batch_size = 7

    k_values = [1, 2, 3, 1, 5, 2, 4]
    node = EnableComponentsNode(component_keys=component_keys, num_enabled=_FixedIntSampler(k_values))

    rng1 = torch.Generator(device=device).manual_seed(123)
    state1 = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out1 = node(state1, rng=rng1)

    enabled = out1.meta["enabled"]
    enabled_matrix = torch.stack([enabled[key] for key in component_keys], dim=1)  # [B, N]
    component_count = out1.y["component_count"]  # [B]

    assert component_count.shape == (batch_size,)
    assert component_count.dtype == torch.int64
    assert enabled_matrix.shape == (batch_size, len(component_keys))
    assert enabled_matrix.dtype == torch.bool
    torch.testing.assert_close(enabled_matrix.sum(dim=1).to(torch.int64), component_count)

    assert "component_id" not in out1.y

    rng2 = torch.Generator(device=device).manual_seed(123)
    state2 = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out2 = node(state2, rng=rng2)

    torch.testing.assert_close(out1.y["component_count"], out2.y["component_count"])
    enabled2 = out2.meta["enabled"]
    for key in component_keys:
        torch.testing.assert_close(enabled[key], enabled2[key])


def test_enable_components_node_emits_component_id_for_single_component_batches() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2"]
    batch_size = 8

    node = EnableComponentsNode(component_keys=component_keys, num_enabled=1)
    rng = torch.Generator(device=device).manual_seed(999)
    state = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out = node(state, rng=rng)

    component_id = out.y["component_id"]  # [B]
    component_count = out.y["component_count"]  # [B]

    assert component_id.shape == (batch_size,)
    assert component_id.dtype == torch.int64
    assert torch.all((component_id >= 0) & (component_id < len(component_keys)))

    torch.testing.assert_close(component_count, torch.ones((batch_size,), device=device, dtype=torch.int64))
    assert out.meta["label_names"]["component_id"] == component_keys

