from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..core.samplers import (
    ConstantSampler,
    SamplerLike,
    sampler_from_value,
    sampler_sample,
    sampler_sample_scalar,
)
from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_extract_process_meta
from .base import View


class ECGLeadsView(View):
    """A view that simulates ECG leads by mixing latent components.

    This view transforms a `LatentState` into an `Observation` by performing a
    linear combination of the latent components `K` to produce a multi-channel
    signal `C` (the "leads").

    The transformation involves:
    1.  **Mixing**: A base mixing matrix `A0` of shape `[C, K]` is used.
    2.  **Jitter**: Optional random noise is added to `A0` for each sample in
        the batch, creating a batch-specific mixing matrix `A`.
    3.  **Delay**: Each channel `C` can be randomly time-shifted by a maximum
        delay, simulating conduction delays.

    Args:
        A0: The base mixing matrix or a sampler for it. Accepts either a
            `torch.Tensor` of shape `[C, K]` or `[B, C, K]`, or a callable
            `A0(batch_size, generator, device)` returning `[B, C, K]`.
        jitter_std: Sampler for the mixing-matrix jitter scale.
        max_delay: Sampler for the maximum integer delay (per batch).
    """

    def __init__(
        self,
        *,
        A0: torch.Tensor
        | Callable[[int, torch.Generator, torch.device], torch.Tensor],
        jitter_std: SamplerLike[float],
        max_delay: SamplerLike[int],
    ) -> None:
        """Initialize ECG lead mixing parameters."""
        super().__init__()

        self._A0_callable: (
            Callable[[int, torch.Generator, torch.device], torch.Tensor] | None
        ) = None

        if callable(A0):
            self._A0_callable = A0
            self.A0 = None
        elif isinstance(A0, torch.Tensor):
            if A0.ndim not in (2, 3):
                raise ValueError(
                    "A0 must have shape [C, K] or [B, C, K]. "
                    f"Got {A0.shape}."
                )
            self.register_buffer("A0", A0.float())
        else:
            raise TypeError(
                "A0 must be a torch.Tensor or a callable returning a torch.Tensor. "
                f"Got {type(A0).__name__}."
            )
        jitter_std_sampler = sampler_from_value(jitter_std, name="jitter_std")
        max_delay_sampler = sampler_from_value(max_delay, name="max_delay")
        if isinstance(jitter_std_sampler, ConstantSampler) and jitter_std_sampler.value < 0:
            raise ValueError(
                f"jitter_std must be non-negative, got {jitter_std_sampler.value}."
            )
        if isinstance(max_delay_sampler, ConstantSampler):
            if isinstance(max_delay_sampler.value, bool) or not isinstance(
                max_delay_sampler.value, int
            ):
                raise ValueError("max_delay must be an integer.")
            if max_delay_sampler.value < 0:
                raise ValueError(
                    f"max_delay must be non-negative, got {max_delay_sampler.value}."
                )

        self.jitter_std = jitter_std
        self.max_delay = max_delay
        self.jitter_std_sampler = jitter_std_sampler
        self.max_delay_sampler = max_delay_sampler

    def forward(
        self,
        input_state: LatentState | Observation,
        *,
        rng: torch.Generator | None = None,
    ) -> Observation:
        """Apply the ECG leads simulation.

        This method expects a `LatentState` with a non-None `latent` tensor.

        Args:
            input_state: A `LatentState` object.
            rng: An optional `torch.Generator` for reproducibility.

        Returns:
            An `Observation` object where `x` is the mixed and delayed
            multi-channel signal of shape `[B, C, L]`.
        """
        if isinstance(input_state, LatentState):
            if input_state.latent is None:
                raise ValueError("ECGLeadsView requires LatentState.latent to be present.")
            latent_signal = input_state.latent  # [B, K, L]
            labels = input_state.y
            process_meta = input_state.meta
        elif isinstance(input_state, Observation):
            latent_signal = input_state.x  # [B, K, L]
            if latent_signal.ndim != 3:
                raise ValueError(
                    "ECGLeadsView expects Observation.x with shape [B, K, L]. "
                    f"Got {latent_signal.shape}."
                )
            labels = input_state.y
            process_meta = utils_extract_process_meta(input_state.meta)
        else:
            raise TypeError(
                f"ECGLeadsView expects LatentState or Observation, got {type(input_state).__name__}."
            )

        batch_size, latent_channels, seq_len = latent_signal.shape
        device = latent_signal.device

        generator, seed, _ = rng_make_generator(rng=rng, device=device)

        # --- Resolve base mixing matrix ---
        if self._A0_callable is None:
            A0_source = self.A0  # [C, K] or [B, C, K]
            if A0_source.ndim == 2:
                if A0_source.shape[1] != latent_channels:
                    raise ValueError(
                        "A0 has incompatible latent channels. "
                        f"A0.shape={A0_source.shape}, latent_channels={latent_channels}."
                    )
                base_A0 = A0_source[None, :, :].expand(batch_size, -1, -1)  # [B, C, K]
                meta_A0 = A0_source  # [C, K]
            elif A0_source.ndim == 3:
                if A0_source.shape[0] != batch_size:
                    raise ValueError(
                        "A0 has incompatible batch size. "
                        f"A0.shape={A0_source.shape}, batch_size={batch_size}."
                    )
                if A0_source.shape[2] != latent_channels:
                    raise ValueError(
                        "A0 has incompatible latent channels. "
                        f"A0.shape={A0_source.shape}, latent_channels={latent_channels}."
                    )
                base_A0 = A0_source  # [B, C, K]
                meta_A0 = A0_source  # [B, C, K]
            else:
                raise ValueError(
                    "A0 must have shape [C, K] or [B, C, K]. "
                    f"Got {A0_source.shape}."
                )
        else:
            A0_source = self._A0_callable(batch_size, generator, device)  # [B, C, K]
            if not isinstance(A0_source, torch.Tensor):
                raise TypeError(
                    "A0 callable must return a torch.Tensor. "
                    f"Got {type(A0_source).__name__}."
                )
            if A0_source.ndim != 3:
                raise ValueError(
                    "A0 callable must return a tensor of shape [B, C, K]. "
                    f"Got {A0_source.shape}."
                )
            if A0_source.shape[0] != batch_size:
                raise ValueError(
                    "A0 callable returned incompatible batch size. "
                    f"A0.shape={A0_source.shape}, batch_size={batch_size}."
                )
            if A0_source.shape[2] != latent_channels:
                raise ValueError(
                    "A0 callable returned incompatible latent channels. "
                    f"A0.shape={A0_source.shape}, latent_channels={latent_channels}."
                )
            base_A0 = A0_source.float()  # [B, C, K]
            meta_A0 = base_A0  # [B, C, K]

        jitter_std = sampler_sample(
            sampler=self.jitter_std_sampler,
            shape=(batch_size,),
            rng=generator,
            device=device,
            dtype=torch.float32,
            name="jitter_std",
        )  # [B]
        if torch.any(jitter_std < 0):
            raise ValueError("jitter_std must be non-negative for all samples.")

        max_delay_samples, max_delay_value = sampler_sample_scalar(
            sampler=self.max_delay_sampler,
            rng=generator,
            device=device,
            dtype=torch.int64,
            name="max_delay",
        )  # [1]
        if isinstance(max_delay_value, bool):
            raise ValueError("max_delay must be an integer.")
        max_delay = int(max_delay_value)
        if max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {max_delay}.")
        max_delay_tensor = max_delay_samples.expand(batch_size)  # [B]

        # --- Apply Mixing Matrix Jitter ---
        if torch.any(jitter_std > 0):
            jitter = torch.randn(
                (batch_size, base_A0.shape[1], latent_channels),
                generator=generator,
                device=device,
            )  # [B, C, K]
            mixing_matrix = base_A0 + jitter * jitter_std[:, None, None]  # [B, C, K]
        else:
            mixing_matrix = base_A0  # [B, C, K]

        # --- Mix latent components into observed channels ---
        observed_signal = torch.einsum(
            "bck,bkl->bcl",
            mixing_matrix,
            latent_signal,
        )  # [B, C, L]

        # --- Apply Channel-wise Delays ---
        delays = torch.zeros((batch_size, base_A0.shape[1]), device=device, dtype=torch.int64)  # [B, C]
        if max_delay > 0:
            # Sample random delays for each channel in the batch
            delays = torch.randint(
                -max_delay,
                max_delay + 1,
                (batch_size, base_A0.shape[1]),
                generator=generator,
                device=device,
            )  # [B, C]

            # Create shifted time indices and gather
            time_idx = torch.arange(seq_len, device=device)  # [L]
            shifted_idx = (time_idx[None, None, :] - delays[:, :, None]).clamp(0, seq_len - 1)  # [B, C, L]
            shifted_idx = shifted_idx.to(torch.int64)  # [B, C, L]
            observed_signal = torch.gather(observed_signal, dim=2, index=shifted_idx)  # [B, C, L]

        samples = {
            "jitter_std": jitter_std,
            "max_delay": max_delay_tensor,
        }
        spec = {
            "jitter_std": self.jitter_std_sampler.spec(),
            "max_delay": self.max_delay_sampler.spec(),
        }
        meta = {
            "view": "ECGLeadsView",
            "seed": seed,
            "A0": meta_A0,
            "A": mixing_matrix,
            "jitter_std": jitter_std,
            "max_delay": max_delay_tensor,
            "delays": delays,
            "samples": samples,
            "spec": spec,
            "process": process_meta,
        }

        return Observation(x=observed_signal, y=labels, meta=meta)
