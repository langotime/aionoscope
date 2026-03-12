from __future__ import annotations

import pytest
import torch

from aiono import (
    BernoulliSampler,
    CategoricalSampler,
    ChoiceSampler,
    ConstantSampler,
    LogUniformSampler,
    NormalSampler,
    RandIntSampler,
    UniformSampler,
    WeightedPermutationSampler,
)


def _sample_with_seed(
    sampler,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    device = torch.device("cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    samples = sampler.sample(
        shape=shape,
        rng=rng,
        device=device,
        dtype=dtype,
    )  # [*shape]
    return samples


def test_sampler_determinism_and_shapes() -> None:
    shape = (2, 3)

    uniform = _sample_with_seed(
        UniformSampler(0.0, 1.0),
        shape=shape,
        dtype=torch.float32,
        seed=123,
    )
    uniform_again = _sample_with_seed(
        UniformSampler(0.0, 1.0),
        shape=shape,
        dtype=torch.float32,
        seed=123,
    )
    torch.testing.assert_close(uniform, uniform_again)
    assert uniform.shape == shape
    assert uniform.dtype == torch.float32

    log_uniform = _sample_with_seed(
        LogUniformSampler(0.1, 2.0),
        shape=shape,
        dtype=torch.float32,
        seed=321,
    )
    assert log_uniform.shape == shape
    assert log_uniform.dtype == torch.float32

    normal = _sample_with_seed(
        NormalSampler(0.0, 1.0),
        shape=shape,
        dtype=torch.float32,
        seed=55,
    )
    assert normal.shape == shape
    assert normal.dtype == torch.float32

    randint = _sample_with_seed(
        RandIntSampler(0, 5),
        shape=shape,
        dtype=torch.int64,
        seed=9,
    )
    assert randint.shape == shape
    assert randint.dtype == torch.int64

    bernoulli = _sample_with_seed(
        BernoulliSampler(0.25),
        shape=shape,
        dtype=torch.bool,
        seed=9,
    )
    assert bernoulli.shape == shape
    assert bernoulli.dtype == torch.bool

    categorical = _sample_with_seed(
        CategoricalSampler([0.2, 0.3, 0.5]),
        shape=shape,
        dtype=torch.int64,
        seed=7,
    )
    assert categorical.shape == shape
    assert categorical.dtype == torch.int64

    choice = _sample_with_seed(
        ChoiceSampler(["a", "b", "c"]),
        shape=shape,
        dtype=torch.int64,
        seed=7,
    )
    assert choice.shape == shape
    assert choice.dtype == torch.int64

    const = _sample_with_seed(
        ConstantSampler(3.0),
        shape=shape,
        dtype=torch.float32,
        seed=1,
    )
    assert const.shape == shape
    assert const.dtype == torch.float32
    assert torch.all(const == 3.0)


def test_categorical_sampler_rejects_non_finite_probs() -> None:
    with pytest.raises(ValueError, match="finite probs"):
        CategoricalSampler([float("nan"), 1.0])
    with pytest.raises(ValueError, match="finite probs"):
        CategoricalSampler([float("inf"), 1.0])


def test_weighted_permutation_sampler_is_deterministic_and_is_a_permutation() -> None:
    sampler = WeightedPermutationSampler(probs=[1.0, 2.0, 0.5, 3.0])
    batch_size = 8
    num_classes = 4

    order1 = _sample_with_seed(
        sampler,
        shape=(batch_size, num_classes),
        dtype=torch.int64,
        seed=123,
    )
    order2 = _sample_with_seed(
        sampler,
        shape=(batch_size, num_classes),
        dtype=torch.int64,
        seed=123,
    )
    torch.testing.assert_close(order1, order2)

    expected = torch.arange(num_classes, dtype=torch.int64)[None, :].expand(batch_size, -1)  # [B, N]
    sorted_order = order1.sort(dim=1).values  # [B, N]
    torch.testing.assert_close(sorted_order, expected)


def test_weighted_permutation_sampler_places_zero_prob_last() -> None:
    sampler = WeightedPermutationSampler(probs=[1.0, 0.0, 2.0])
    order = _sample_with_seed(
        sampler,
        shape=(64, 3),
        dtype=torch.int64,
        seed=7,
    )
    assert torch.all(order[:, -1] == 1)


def test_weighted_permutation_sampler_validates_probs() -> None:
    with pytest.raises(ValueError, match="non-empty probs"):
        WeightedPermutationSampler(probs=[])
    with pytest.raises(ValueError, match="finite probs"):
        WeightedPermutationSampler(probs=[float("nan"), 1.0])
    with pytest.raises(ValueError, match="finite probs"):
        WeightedPermutationSampler(probs=[float("inf"), 1.0])
    with pytest.raises(ValueError, match="non-negative probs"):
        WeightedPermutationSampler(probs=[1.0, -1.0])
    with pytest.raises(ValueError, match="sum > 0"):
        WeightedPermutationSampler(probs=[0.0, 0.0])
