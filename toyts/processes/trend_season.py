from __future__ import annotations

import math

import torch
from torch import nn

from ..core.rng import rng_make_generator
from ..core.types import LatentState


class TrendSeasonAnomalyProcess(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        components: int,
        regime_classes: list[str] | None = None,
        anomaly_classes: list[str] | None = None,
        slope_max: float = 0.8,
        season_amp: float = 1.0,
        spiky_boost: float = 2.0,
        season_freq_min: float = 1.0,
        season_freq_max: float = 4.0,
        anomaly_scale: float = 1.5,
    ) -> None:
        super().__init__()

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if components < 3:
            raise ValueError("components must be >= 3 (trend, season, anomaly).")
        if slope_max <= 0:
            raise ValueError(f"slope_max must be positive, got {slope_max}.")
        if season_amp <= 0:
            raise ValueError(f"season_amp must be positive, got {season_amp}.")
        if season_freq_min <= 0 or season_freq_max <= 0:
            raise ValueError("season_freq_min/max must be positive.")
        if season_freq_max < season_freq_min:
            raise ValueError("season_freq_max must be >= season_freq_min.")
        if anomaly_scale <= 0:
            raise ValueError(f"anomaly_scale must be positive, got {anomaly_scale}.")

        self.seq_len = seq_len
        self.components = components
        self.regime_classes = regime_classes or ["steady", "ramping", "spiky"]
        self.anomaly_classes = anomaly_classes or ["none", "drop", "spike"]
        self.slope_max = slope_max
        self.season_amp = season_amp
        self.spiky_boost = spiky_boost
        self.season_freq_min = season_freq_min
        self.season_freq_max = season_freq_max
        self.anomaly_scale = anomaly_scale

        if not self.regime_classes:
            raise ValueError("regime_classes must be non-empty.")
        if not self.anomaly_classes:
            raise ValueError("anomaly_classes must be non-empty.")

        self._steady_index = self.regime_classes.index("steady")
        self._ramping_index = self.regime_classes.index("ramping")
        self._spiky_index = self.regime_classes.index("spiky")

        self._none_index = self.anomaly_classes.index("none")
        self._drop_index = self.anomaly_classes.index("drop")
        self._spike_index = self.anomaly_classes.index("spike")

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

        regime_idx = torch.randint(
            0,
            len(self.regime_classes),
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        anomaly_idx = torch.randint(
            0,
            len(self.anomaly_classes),
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]

        time_grid = torch.linspace(0, 1, steps=self.seq_len, device=device)  # [L]
        time_grid = time_grid[None, None, :]  # [1, 1, L]

        slope = torch.zeros(batch_size, device=device)  # [B]
        ramp_mask = (regime_idx == self._ramping_index)  # [B]
        slope_noise = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        slope_noise = (slope_noise * 2.0 - 1.0) * self.slope_max  # [B]
        slope = torch.where(ramp_mask, slope_noise, slope)  # [B]

        offset = torch.randn(
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        trend = (slope[:, None, None] * (time_grid - 0.5)) + offset[:, None, None]  # [B, 1, L]

        freq = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        freq = self.season_freq_min + (self.season_freq_max - self.season_freq_min) * freq  # [B]
        phase = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        phase = phase * (2.0 * math.pi)  # [B]

        spiky_mask = (regime_idx == self._spiky_index)  # [B]
        season_amp = torch.full((batch_size,), self.season_amp, device=device)  # [B]
        season_amp = torch.where(spiky_mask, season_amp * self.spiky_boost, season_amp)  # [B]

        season = season_amp[:, None, None] * torch.sin(
            2.0 * math.pi * freq[:, None, None] * time_grid + phase[:, None, None]
        )  # [B, 1, L]

        anomaly_amp = torch.zeros(batch_size, device=device)  # [B]
        spike_mask = (anomaly_idx == self._spike_index)  # [B]
        drop_mask = (anomaly_idx == self._drop_index)  # [B]

        anomaly_noise = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        anomaly_noise = (0.5 + anomaly_noise) * self.anomaly_scale  # [B]
        anomaly_amp = torch.where(spike_mask, anomaly_noise, anomaly_amp)  # [B]
        anomaly_amp = torch.where(drop_mask, -anomaly_noise, anomaly_amp)  # [B]

        anomaly_center = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
        )  # [B]
        anomaly_center = anomaly_center[:, None, None]  # [B, 1, 1]
        anomaly_sigma = torch.full((batch_size, 1, 1), 0.03, device=device)  # [B, 1, 1]

        anomaly = anomaly_amp[:, None, None] * torch.exp(
            -0.5 * ((time_grid - anomaly_center) / anomaly_sigma).pow(2)
        )  # [B, 1, L]

        latent = torch.zeros(
            (batch_size, self.components, self.seq_len),
            device=device,
        )  # [B, K, L]
        latent[:, 0:1, :] = trend
        latent[:, 1:2, :] = season
        latent[:, 2:3, :] = anomaly

        if self.components > 3:
            extra_noise = torch.randn(
                (batch_size, self.components - 3, self.seq_len),
                generator=generator,
                device=device,
            )  # [B, K-3, L]
            latent[:, 3:, :] = 0.05 * extra_noise

        centers = torch.empty((batch_size, 0), device=device)  # [B, 0]

        y = {
            "regime": regime_idx,
            "anomaly_type": anomaly_idx,
        }
        meta = {
            "process": "TrendSeasonAnomalyProcess",
            "seed": seed,
            "seq_len": self.seq_len,
            "components": self.components,
            "regime_names": self.regime_classes,
            "anomaly_names": self.anomaly_classes,
            "season_freq_min": self.season_freq_min,
            "season_freq_max": self.season_freq_max,
        }

        return LatentState(centers=centers, latent=latent, y=y, meta=meta)
