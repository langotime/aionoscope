from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..core.rng import rng_make_generator
from ..core.types import LatentState, Observation
from ..core.utils import utils_require_latent
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
        jitter_std: The standard deviation of the Gaussian noise added to the
            mixing matrix for each sample. If 0, no jitter is applied.
        max_delay: The maximum integer delay (positive or negative) to be
            applied independently to each channel. If 0, no delay is applied.
    """

    def __init__(
        self,
        *,
        A0: torch.Tensor
        | Callable[[int, torch.Generator, torch.device], torch.Tensor],
        jitter_std: float,
        max_delay: int,
    ) -> None:
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
        if jitter_std < 0:
            raise ValueError(f"jitter_std must be non-negative, got {jitter_std}.")
        if max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {max_delay}.")

        self.jitter_std = jitter_std
        self.max_delay = max_delay

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
        latent_state = utils_require_latent(input_state, name="ECGLeadsView")
        if latent_state.latent is None:
            raise ValueError("ECGLeadsView requires LatentState.latent to be present.")

        latent_signal = latent_state.latent  # [B, K, L]
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

        # --- Apply Mixing Matrix Jitter ---
        if self.jitter_std > 0:
            jitter = torch.randn(
                (batch_size, base_A0.shape[1], latent_channels),
                generator=generator,
                device=device,
            )  # [B, C, K]
            mixing_matrix = base_A0 + jitter * self.jitter_std  # [B, C, K]
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
        if self.max_delay > 0:
            # Sample random delays for each channel in the batch
            delays = torch.randint(
                -self.max_delay,
                self.max_delay + 1,
                (batch_size, base_A0.shape[1]),
                generator=generator,
                device=device,
            )  # [B, C]

            # Create shifted time indices and gather
            time_idx = torch.arange(seq_len, device=device)  # [L]
            shifted_idx = (time_idx[None, None, :] - delays[:, :, None]).clamp(0, seq_len - 1)  # [B, C, L]
            shifted_idx = shifted_idx.to(torch.int64)  # [B, C, L]
            observed_signal = torch.gather(observed_signal, dim=2, index=shifted_idx)  # [B, C, L]

        meta = {
            "view": "ECGLeadsView",
            "seed": seed,
            "A0": meta_A0,
            "A": mixing_matrix,
            "jitter_std": self.jitter_std,
            "max_delay": self.max_delay,
            "delays": delays,
            "process": latent_state.meta,
        }

        return Observation(x=observed_signal, y=latent_state.y, meta=meta)
