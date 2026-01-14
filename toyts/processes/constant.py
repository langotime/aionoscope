from __future__ import annotations

import torch

from ..core.samplers import ConstantSampler, SamplerLike, sampler_from_value, sampler_sample
from ..core.utils import SAMPLES_PREFIX
from .base import Process
from .graph import ProcessGraph, ProcessNode, ProcessState


class ConstantLatentNode(ProcessNode):
    """Write a constant latent signal into state.data.

    Output latent has shape [B, K, L] where K is `channels`.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        channels: int,
        value: SamplerLike[float],
        out_key: str,
        enabled_key: str | None = None,
    ) -> None:
        """Initialize a constant latent generator.

        Args:
            seq_len: Sequence length L.
            channels: Number of latent components K.
            value: Sampler for the constant value per sample.
            out_key: Destination key in state.data (typically "latent").
            enabled_key: Optional key under state.meta["enabled"] to gate the component
                per sample (bool [B]); disabled samples get value 0.0.
        """
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if not out_key:
            raise ValueError("out_key must be non-empty.")
        if enabled_key is not None and not enabled_key:
            raise ValueError("enabled_key must be non-empty when provided.")

        self.seq_len = seq_len
        self.channels = channels
        self.value_sampler = sampler_from_value(value, name="value")
        self.value = value
        self.out_key = out_key
        self.enabled_key = enabled_key

        if isinstance(self.value_sampler, ConstantSampler):
            value_scalar = float(self.value_sampler.value)
            if not torch.isfinite(torch.tensor(value_scalar)):
                raise ValueError(f"value must be finite, got {value_scalar}.")

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Sample a constant per sample and broadcast to [B, K, L]."""
        self._record_seed(state, rng)
        if state.device.type != rng.device.type:
            raise ValueError(
                "ConstantLatentNode rng device does not match state.device. "
                f"rng.device={rng.device}, state.device={state.device}."
            )

        batch_size = state.batch_size
        value = sampler_sample(
            sampler=self.value_sampler,
            shape=(batch_size,),
            rng=rng,
            device=state.device,
            dtype=torch.float32,
            name="value",
        )  # [B]
        if not torch.all(torch.isfinite(value)):
            raise ValueError("ConstantLatentNode value must be finite for all samples.")

        if self.enabled_key is not None:
            enabled = state.meta.get("enabled")
            if not isinstance(enabled, dict):
                raise ValueError(
                    "ConstantLatentNode requires state.meta['enabled'] to be a dict when enabled_key "
                    f"is set. Got {type(enabled).__name__}."
                )
            enabled_mask = enabled.get(self.enabled_key)
            if not isinstance(enabled_mask, torch.Tensor):
                raise ValueError(
                    "ConstantLatentNode enabled mask must be a torch.Tensor. "
                    f"Got {type(enabled_mask).__name__}."
                )
            if enabled_mask.dtype != torch.bool:
                raise ValueError(
                    f"ConstantLatentNode enabled mask must be bool, got {enabled_mask.dtype}."
                )
            if enabled_mask.shape != (batch_size,):
                raise ValueError(
                    "ConstantLatentNode enabled mask must have shape [B]. "
                    f"Got {enabled_mask.shape}, batch_size={batch_size}."
                )
            if enabled_mask.device != value.device:
                raise ValueError(
                    "ConstantLatentNode enabled mask device mismatch. "
                    f"mask.device={enabled_mask.device}, value.device={value.device}."
                )
            value = value * enabled_mask.to(dtype=value.dtype)  # [B]

        latent = value[:, None, None].expand(batch_size, self.channels, self.seq_len)  # [B, K, L]
        state.data[self.out_key] = latent

        samples_base = f"{SAMPLES_PREFIX}/ConstantLatentNode:{self.out_key}"
        state.data[f"{samples_base}/value"] = value
        return state


class ConstantProcess(ProcessGraph):
    """Generate a constant latent signal.

    This is the canonical "constant baseline" process used for additive component
    views (trend/periodic/noise), with `sample_rate_hz` stored in process meta.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        value: SamplerLike[float],
        channels: int = 1,
    ) -> None:
        """Initialize the constant process.

        Args:
            seq_len: Sequence length L.
            sample_rate_hz: Sampling rate in Hz (used by frequency-aware views).
            value: Sampler for the constant value per sample.
            channels: Number of latent components K.
        """
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        super().__init__(
            graph=[
                ConstantLatentNode(
                    seq_len=seq_len,
                    channels=channels,
                    value=value,
                    out_key="latent",
                )
            ],
            outputs={"latent"},
            name="ConstantProcess",
            base_meta={
                "seq_len": seq_len,
                "sample_rate_hz": float(sample_rate_hz),
                "channels": channels,
            },
        )
