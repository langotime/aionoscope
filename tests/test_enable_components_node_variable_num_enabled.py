from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import pytest

from toyts import CategoricalSampler, EnableComponentsNode, Sampler, WeightedPermutationSampler
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


def test_enable_components_node_component_id_sampler_is_deterministic() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2"]
    batch_size = 64

    sampler = CategoricalSampler(probs=[0.2, 0.7, 0.1])
    node = EnableComponentsNode(component_keys=component_keys, num_enabled=1, component_id=sampler)

    rng1 = torch.Generator(device=device).manual_seed(123)
    state1 = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out1 = node(state1, rng=rng1)

    rng2 = torch.Generator(device=device).manual_seed(123)
    state2 = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out2 = node(state2, rng=rng2)

    torch.testing.assert_close(out1.y["component_id"], out2.y["component_id"])
    enabled1 = out1.meta["enabled"]
    enabled2 = out2.meta["enabled"]
    for key in component_keys:
        torch.testing.assert_close(enabled1[key], enabled2[key])

    assert out1.meta["enabled_spec"]["component_id"] == sampler.spec()


def test_enable_components_node_component_id_sampler_respects_zero_probabilities() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2"]
    batch_size = 32

    sampler = CategoricalSampler(probs=[0.0, 1.0, 0.0])
    node = EnableComponentsNode(component_keys=component_keys, num_enabled=1, component_id=sampler)

    rng = torch.Generator(device=device).manual_seed(999)
    state = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out = node(state, rng=rng)

    component_id = out.y["component_id"]  # [B]
    assert torch.all(component_id == 1)

    enabled = out.meta["enabled"]
    assert torch.all(enabled["c1"])
    assert torch.all(~enabled["c0"])
    assert torch.all(~enabled["c2"])


def test_enable_components_node_component_id_sampler_requires_num_enabled_one() -> None:
    component_keys = ["c0", "c1"]
    sampler = CategoricalSampler(probs=[0.5, 0.5])

    with pytest.raises(ValueError, match="component_id is only supported when num_enabled is a constant 1"):
        EnableComponentsNode(component_keys=component_keys, num_enabled=2, component_id=sampler)

    fixed_ones = _FixedIntSampler([1, 1, 1])
    with pytest.raises(ValueError, match="component_id is only supported when num_enabled is a constant 1"):
        EnableComponentsNode(component_keys=component_keys, num_enabled=fixed_ones, component_id=sampler)


def test_enable_components_node_component_id_sampler_rejects_out_of_range_indices() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2"]
    batch_size = 3

    out_of_range = _FixedIntSampler([0, 3, 1])
    node = EnableComponentsNode(component_keys=component_keys, num_enabled=1, component_id=out_of_range)

    rng = torch.Generator(device=device).manual_seed(1)
    state = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    with pytest.raises(ValueError, match="sampled component_id out of range"):
        node(state, rng=rng)


def test_enable_components_node_component_order_sampler_produces_biased_k_hot_masks() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2", "c3"]
    batch_size = 64

    sampler = WeightedPermutationSampler(probs=[1.0, 1.0, 0.0, 0.0])
    node = EnableComponentsNode(component_keys=component_keys, num_enabled=2, component_order=sampler)

    rng = torch.Generator(device=device).manual_seed(123)
    state = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    out = node(state, rng=rng)

    enabled = out.meta["enabled"]
    enabled_matrix = torch.stack([enabled[key] for key in component_keys], dim=1)  # [B, N]

    assert enabled_matrix.dtype == torch.bool
    torch.testing.assert_close(
        enabled_matrix.sum(dim=1).to(torch.int64),
        torch.full((batch_size,), 2, device=device, dtype=torch.int64),
    )
    assert torch.all(~enabled["c2"])
    assert torch.all(~enabled["c3"])
    assert out.meta["enabled_spec"]["component_order"] == sampler.spec()


def test_enable_components_node_component_order_sampler_rejects_impossible_k() -> None:
    device = torch.device("cpu")
    component_keys = ["c0", "c1", "c2"]
    batch_size = 4

    sampler = WeightedPermutationSampler(probs=[1.0, 0.0, 0.0])
    node = EnableComponentsNode(component_keys=component_keys, num_enabled=2, component_order=sampler)

    rng = torch.Generator(device=device).manual_seed(1)
    state = ProcessState(batch_size=batch_size, device=device, data={}, y={}, meta={})
    with pytest.raises(ValueError, match="num_positive_probs"):
        node(state, rng=rng)


def test_enable_components_node_rejects_component_id_and_component_order_together() -> None:
    sampler = CategoricalSampler(probs=[0.5, 0.5])
    order = WeightedPermutationSampler(probs=[1.0, 1.0])
    with pytest.raises(ValueError, match="does not allow both component_id and component_order"):
        EnableComponentsNode(
            component_keys=["c0", "c1"],
            num_enabled=1,
            component_id=sampler,
            component_order=order,
        )
