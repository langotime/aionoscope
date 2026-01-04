from __future__ import annotations

import torch
from torch import nn

from ..core.rng import rng_make_generator
from ..core.types import LatentState
from ..kernels.morph import morph_dog, morph_gaussian, morph_laplace


class PulseTrainProcess(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        num_pulses: int,
        rhythm_classes: list[str],
        shape_classes: list[str],
        latent_mode: str,
        amplitude: float = 1.0,
        missed_gap_factor: float = 2.5,
    ) -> None:
        super().__init__()

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if num_pulses <= 0:
            raise ValueError(f"num_pulses must be positive, got {num_pulses}.")
        if not rhythm_classes:
            raise ValueError("rhythm_classes must be non-empty.")
        if not shape_classes:
            raise ValueError("shape_classes must be non-empty.")
        if latent_mode != "pqrst3":
            raise ValueError("Only latent_mode='pqrst3' is supported in MVP.")

        required_rhythms = {"regular", "irregular", "missed_beat"}
        missing_rhythms = required_rhythms - set(rhythm_classes)
        if missing_rhythms:
            raise ValueError(
                "rhythm_classes must include regular, irregular, missed_beat. "
                f"Missing: {sorted(missing_rhythms)}."
            )

        required_shapes = {"gaussian", "sharp_laplace", "biphasic_dog"}
        missing_shapes = required_shapes - set(shape_classes)
        if missing_shapes:
            raise ValueError(
                "shape_classes must include gaussian, sharp_laplace, biphasic_dog. "
                f"Missing: {sorted(missing_shapes)}."
            )

        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}.")
        if missed_gap_factor <= 1:
            raise ValueError(
                f"missed_gap_factor must be >1 to create a pause, got {missed_gap_factor}."
            )

        self.seq_len = seq_len
        self.num_pulses = num_pulses
        self.rhythm_classes = list(rhythm_classes)
        self.shape_classes = list(shape_classes)
        self.latent_mode = latent_mode
        self.amplitude = amplitude
        self.missed_gap_factor = missed_gap_factor

        self._regular_index = self.rhythm_classes.index("regular")
        self._irregular_index = self.rhythm_classes.index("irregular")
        self._missed_index = self.rhythm_classes.index("missed_beat")

        self._gaussian_index = self.shape_classes.index("gaussian")
        self._laplace_index = self.shape_classes.index("sharp_laplace")
        self._dog_index = self.shape_classes.index("biphasic_dog")

        self._offset_fractions = [-0.2, 0.0, 0.25]
        self._sigma_fractions = [0.08, 0.04, 0.1]
        self._component_weights = [0.5, 1.0, 0.45]
        self._laplace_scale = 0.6
        self._dog_sigma1 = 0.6
        self._dog_sigma2 = 1.4
        self._dog_alpha = 0.8

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        generator, seed, _ = rng_make_generator(rng=rng, device=device)

        shape_idx = torch.randint(
            0,
            len(self.shape_classes),
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        rhythm_idx = torch.randint(
            0,
            len(self.rhythm_classes),
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]

        base_interval = 1.0 / (self.num_pulses + 1)
        base_intervals = torch.full(
            (batch_size, self.num_pulses + 1),
            fill_value=base_interval,
            device=device,
        )  # [B, N+1]
        random_intervals = torch.rand(
            (batch_size, self.num_pulses + 1),
            generator=generator,
            device=device,
        )  # [B, N+1]
        random_intervals = random_intervals / random_intervals.sum(dim=1, keepdim=True)  # [B, N+1]

        missed_indices = torch.randint(
            0,
            self.num_pulses + 1,
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        missed_multipliers = torch.ones(
            (batch_size, self.num_pulses + 1),
            device=device,
        )  # [B, N+1]
        missed_multipliers.scatter_(
            1,
            missed_indices[:, None],
            self.missed_gap_factor,
        )
        missed_intervals = base_intervals * missed_multipliers  # [B, N+1]
        missed_intervals = missed_intervals / missed_intervals.sum(dim=1, keepdim=True)  # [B, N+1]

        intervals = base_intervals  # [B, N+1]
        irregular_mask = (rhythm_idx == self._irregular_index)  # [B]
        missed_mask = (rhythm_idx == self._missed_index)  # [B]
        intervals = torch.where(irregular_mask[:, None], random_intervals, intervals)  # [B, N+1]
        intervals = torch.where(missed_mask[:, None], missed_intervals, intervals)  # [B, N+1]

        centers_normalized = intervals.cumsum(dim=1)[:, :-1]  # [B, N]
        centers = centers_normalized * (self.seq_len - 1)  # [B, N]

        time_grid = torch.linspace(
            0,
            self.seq_len - 1,
            steps=self.seq_len,
            device=device,
        )  # [L]
        spacing = (self.seq_len - 1) / (self.num_pulses + 1)

        offsets = torch.tensor(self._offset_fractions, device=device) * spacing  # [K]
        sigmas = torch.tensor(self._sigma_fractions, device=device) * spacing  # [K]
        weights = torch.tensor(self._component_weights, device=device)  # [K]

        centers_k = centers[:, :, None] + offsets[None, None, :]  # [B, N, K]
        relative = time_grid[None, None, None, :] - centers_k[..., None]  # [B, N, K, L]

        sigma_grid = sigmas[None, None, :, None]  # [1, 1, K, 1]
        gaussian = morph_gaussian(relative_t=relative, sigma=sigma_grid)  # [B, N, K, L]

        qrs_relative = relative[:, :, 1:2, :]  # [B, N, 1, L]
        qrs_gaussian = gaussian[:, :, 1:2, :]  # [B, N, 1, L]

        laplace_scale = sigmas[1] * self._laplace_scale  # []
        qrs_laplace = morph_laplace(relative_t=qrs_relative, scale=laplace_scale)  # [B, N, 1, L]

        dog_sigma1 = sigmas[1] * self._dog_sigma1  # []
        dog_sigma2 = sigmas[1] * self._dog_sigma2  # []
        dog_alpha = torch.tensor(self._dog_alpha, device=device)  # []
        qrs_dog = morph_dog(
            relative_t=qrs_relative,
            sigma1=dog_sigma1,
            sigma2=dog_sigma2,
            alpha=dog_alpha,
        )  # [B, N, 1, L]

        laplace_mask = (shape_idx == self._laplace_index)[:, None, None, None]  # [B, 1, 1, 1]
        dog_mask = (shape_idx == self._dog_index)[:, None, None, None]  # [B, 1, 1, 1]

        qrs = qrs_gaussian  # [B, N, 1, L]
        qrs = torch.where(laplace_mask, qrs_laplace, qrs)  # [B, N, 1, L]
        qrs = torch.where(dog_mask, qrs_dog, qrs)  # [B, N, 1, L]

        components = gaussian  # [B, N, K, L]
        components[:, :, 1:2, :] = qrs
        components = components * weights[None, None, :, None]  # [B, N, K, L]

        latent = components.sum(dim=1)  # [B, K, L]
        latent = latent - latent.mean(dim=-1, keepdim=True)  # [B, K, L]

        energy = latent.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # [B, 1, 1]
        if torch.any(energy <= 0):
            raise ValueError("Latent energy must be positive for normalization.")
        latent = latent / energy * self.amplitude  # [B, K, L]

        y = {
            "shape": shape_idx,
            "rhythm": rhythm_idx,
        }
        meta = {
            "process": "PulseTrainProcess",
            "seed": seed,
            "seq_len": self.seq_len,
            "num_pulses": self.num_pulses,
            "latent_mode": self.latent_mode,
            "shape_names": self.shape_classes,
            "rhythm_names": self.rhythm_classes,
            "spacing_samples": spacing,
        }

        return LatentState(centers=centers, latent=latent, y=y, meta=meta)
