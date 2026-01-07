from __future__ import annotations

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
        A0: The base mixing matrix `[C, K]` that maps latent components to
            observed channels.
        jitter_std: The standard deviation of the Gaussian noise added to the
            mixing matrix for each sample. If 0, no jitter is applied.
        max_delay: The maximum integer delay (positive or negative) to be
            applied independently to each channel. If 0, no delay is applied.
    """

    def __init__(
        self,
        *,
        A0: torch.Tensor,
        jitter_std: float,
        max_delay: int,
    ) -> None:
        super().__init__()

        if A0.ndim != 2:
            raise ValueError(f"A0 must be 2D [C, K], got {A0.shape}.")
        if jitter_std < 0:
            raise ValueError(f"jitter_std must be non-negative, got {jitter_std}.")
        if max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {max_delay}.")

        self.register_buffer("A0", A0.float())
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

        if self.A0.shape[1] != latent_channels:
            raise ValueError(
                "A0 has incompatible latent channels. "
                f"A0.shape={self.A0.shape}, latent_channels={latent_channels}."
            )

        generator, seed, _ = rng_make_generator(rng=rng, device=device)

        # --- Apply Mixing Matrix Jitter ---
        if self.jitter_std > 0:
            jitter = torch.randn(
                (batch_size, self.A0.shape[0], latent_channels),
                generator=generator,
                device=device,
            )  # [B, C, K]
            mixing_matrix = self.A0[None, :, :] + jitter * self.jitter_std  # [B, C, K]
        else:
            mixing_matrix = self.A0[None, :, :].expand(batch_size, -1, -1)  # [B, C, K]

        # --- Mix latent components into observed channels ---
        observed_signal = torch.einsum(
            "bck,bkl->bcl",
            mixing_matrix,
            latent_signal,
        )  # [B, C, L]

        # --- Apply Channel-wise Delays ---
        delays = torch.zeros((batch_size, self.A0.shape[0]), device=device, dtype=torch.int64)  # [B, C]
        if self.max_delay > 0:
            # Sample random delays for each channel in the batch
            delays = torch.randint(
                -self.max_delay,
                self.max_delay + 1,
                (batch_size, self.A0.shape[0]),
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
            "A0": self.A0,
            "A": mixing_matrix,
            "jitter_std": self.jitter_std,
            "max_delay": self.max_delay,
            "delays": delays,
            "process": latent_state.meta,
        }

        return Observation(x=observed_signal, y=latent_state.y, meta=meta)
