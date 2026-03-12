from __future__ import annotations

import math

import torch
from torch import nn

from ..core.events import EventSchema
from ..core.samplers import (
    ConstantSampler,
    Sampler,
    SamplerLike,
    sampler_from_value,
    sampler_sample_scalar,
)
from ..core.types import LatentState
from ..core.utils import SAMPLES_PREFIX
from .graph import ProcessGraph, ProcessNode, ProcessState, Switch
from .nodes import EventTrainNode, SampleLabelsNode


class _PulseTrainParamsNode(ProcessNode):
    """Sample pulse train parameters that affect event count."""

    def __init__(
        self,
        *,
        seq_len: int,
        sample_rate_hz: float,
        frequency_sampler: Sampler,
        num_events_key: str,
    ) -> None:
        """Initialize the parameter sampler node."""
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        if not num_events_key:
            raise ValueError("num_events_key must be non-empty.")

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = (seq_len - 1) / sample_rate_hz
        self.frequency_sampler = frequency_sampler
        self.num_events_key = num_events_key

    def forward(self, state: ProcessState, *, rng: torch.Generator) -> ProcessState:
        """Sample frequency and derive the number of events."""
        self._record_seed(state, rng)
        frequency_samples, frequency_value = sampler_sample_scalar(
            sampler=self.frequency_sampler,
            rng=rng,
            device=state.device,
            dtype=torch.float32,
            name="frequency_hz",
        )  # [1]
        if frequency_value <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_value}.")
        frequency_hz = frequency_samples.expand(state.batch_size)  # [B]

        expected_pulses = frequency_value * self.duration_sec
        num_pulses = int(math.floor(expected_pulses + 0.5))
        if num_pulses <= 0:
            raise ValueError(
                "frequency_hz and sample_rate_hz yield fewer than 1 pulse "
                f"for seq_len={self.seq_len}. "
                f"Got duration_sec={self.duration_sec:.6f}, expected_pulses={expected_pulses:.6f}."
            )

        spacing = (self.seq_len - 1) / (num_pulses + 1)
        num_pulses_tensor = torch.full(
            (state.batch_size,),
            num_pulses,
            device=state.device,
            dtype=torch.int64,
        )  # [B]
        spacing_tensor = torch.full(
            (state.batch_size,),
            spacing,
            device=state.device,
            dtype=torch.float32,
        )  # [B]

        state.data[self.num_events_key] = num_pulses

        samples_base = f"{SAMPLES_PREFIX}/PulseTrainProcess"
        state.data[f"{samples_base}/frequency_hz"] = frequency_hz
        state.data[f"{samples_base}/num_pulses"] = num_pulses_tensor
        state.data[f"{samples_base}/spacing_samples"] = spacing_tensor
        return state


class PulseTrainProcess(nn.Module):
    """A process that generates a train of pulses as an event stream.

    This module samples rhythm and shape labels, generates event centers, and
    emits an `EventBatch` where the event type encodes the pulse shape.

    Args:
        seq_len: The length of the generated sequence `L`.
        frequency_hz: Sampler for the pulse frequency in Hz (per batch). The
            actual pulse count is `round(frequency_hz * duration_sec)`.
        sample_rate_hz: The sampling rate used to interpret `seq_len` in seconds.
        rhythm_classes: A list of rhythm types to sample from. Must include
            "regular", "irregular", and "missed_beat".
        shape_classes: A list of pulse shape types to sample from. Must include
            "gaussian", "sharp_laplace", and "biphasic_dog".
        latent_mode: The structure of the latent components. Currently, only
            "pqrst3" is supported.
        amplitude: Sampler for per-event amplitudes (stored in event params).
        missed_gap_factor: In the "missed_beat" rhythm, the factor by which the
            interval is increased to simulate a pause. Must be > 1.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        frequency_hz: SamplerLike[float],
        sample_rate_hz: float,
        rhythm_classes: list[str],
        shape_classes: list[str],
        latent_mode: str,
        amplitude: SamplerLike[float] = 1.0,
        missed_gap_factor: SamplerLike[float] = 2.5,
    ) -> None:
        """Initialize a pulse-train event process."""
        super().__init__()

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        if not rhythm_classes:
            raise ValueError("rhythm_classes must be non-empty.")
        if not shape_classes:
            raise ValueError("shape_classes must be non-empty.")
        if latent_mode != "pqrst3":
            raise ValueError("Only latent_mode='pqrst3' is supported in MVP.")
        frequency_sampler = sampler_from_value(frequency_hz, name="frequency_hz")
        amplitude_sampler = sampler_from_value(amplitude, name="amplitude")
        missed_gap_factor_sampler = sampler_from_value(
            missed_gap_factor, name="missed_gap_factor"
        )
        if isinstance(frequency_sampler, ConstantSampler):
            if frequency_sampler.value <= 0:
                raise ValueError(
                    f"frequency_hz must be positive, got {frequency_sampler.value}."
                )
        if isinstance(amplitude_sampler, ConstantSampler):
            if amplitude_sampler.value <= 0:
                raise ValueError(
                    f"amplitude must be positive, got {amplitude_sampler.value}."
                )
        if isinstance(missed_gap_factor_sampler, ConstantSampler):
            if missed_gap_factor_sampler.value <= 1:
                raise ValueError(
                    "missed_gap_factor must be >1 to create a pause, got "
                    f"{missed_gap_factor_sampler.value}."
                )

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

        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = (seq_len - 1) / sample_rate_hz
        self.frequency_sampler = frequency_sampler
        self.amplitude_sampler = amplitude_sampler
        self.missed_gap_factor_sampler = missed_gap_factor_sampler

        self.frequency_hz: float | None = None
        self.num_pulses: int | None = None
        if isinstance(self.frequency_sampler, ConstantSampler):
            self.frequency_hz = float(self.frequency_sampler.value)
            expected_pulses = self.frequency_hz * self.duration_sec
            num_pulses = int(math.floor(expected_pulses + 0.5))
            if num_pulses <= 0:
                raise ValueError(
                    "frequency_hz and sample_rate_hz yield fewer than 1 pulse "
                    f"for seq_len={seq_len}. "
                    f"Got duration_sec={self.duration_sec:.6f}, expected_pulses={expected_pulses:.6f}."
                )
            self.num_pulses = num_pulses
        self.rhythm_classes = list(rhythm_classes)
        self.shape_classes = list(shape_classes)
        self.latent_mode = latent_mode
        self.amplitude = amplitude
        self.missed_gap_factor = missed_gap_factor

        self._regular_index = self.rhythm_classes.index("regular")
        self._irregular_index = self.rhythm_classes.index("irregular")
        self._missed_index = self.rhythm_classes.index("missed_beat")

        self.schema = EventSchema(
            type_names=self.shape_classes,
            param_names=["amplitude"],
            time_unit="samples",
        )
        base_meta = {
            "seq_len": self.seq_len,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_sec": self.duration_sec,
            "latent_mode": self.latent_mode,
            "shape_names": self.shape_classes,
            "rhythm_names": self.rhythm_classes,
        }
        if self.frequency_hz is not None and self.num_pulses is not None:
            spacing = (self.seq_len - 1) / (self.num_pulses + 1)
            base_meta["frequency_hz"] = self.frequency_hz
            base_meta["num_pulses"] = self.num_pulses
            base_meta["spacing_samples"] = spacing
        if isinstance(self.amplitude_sampler, ConstantSampler):
            base_meta["amplitude"] = float(self.amplitude_sampler.value)
        if isinstance(self.missed_gap_factor_sampler, ConstantSampler):
            base_meta["missed_gap_factor"] = float(self.missed_gap_factor_sampler.value)

        self._graph = ProcessGraph(
            name="PulseTrainProcess",
            outputs={"events", "centers"},
            base_meta=base_meta,
            graph=[
                _PulseTrainParamsNode(
                    seq_len=self.seq_len,
                    sample_rate_hz=self.sample_rate_hz,
                    frequency_sampler=self.frequency_sampler,
                    num_events_key="pulse.num_events",
                ),
                SampleLabelsNode(
                    labels={
                        "shape": self.shape_classes,
                        "rhythm": self.rhythm_classes,
                    }
                ),
                Switch(
                    label_key="rhythm",
                    cases={
                        self._regular_index: EventTrainNode(
                            seq_len=self.seq_len,
                            num_events="pulse.num_events",
                            schema=self.schema,
                            mode="regular",
                            type_label_key="shape",
                            type_id=None,
                            amplitude=self.amplitude_sampler,
                            amplitude_param="amplitude",
                            missed_gap_factor=self.missed_gap_factor_sampler,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                        self._irregular_index: EventTrainNode(
                            seq_len=self.seq_len,
                            num_events="pulse.num_events",
                            schema=self.schema,
                            mode="irregular",
                            type_label_key="shape",
                            type_id=None,
                            amplitude=self.amplitude_sampler,
                            amplitude_param="amplitude",
                            missed_gap_factor=self.missed_gap_factor_sampler,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                        self._missed_index: EventTrainNode(
                            seq_len=self.seq_len,
                            num_events="pulse.num_events",
                            schema=self.schema,
                            mode="missed_beat",
                            type_label_key="shape",
                            type_id=None,
                            amplitude=self.amplitude_sampler,
                            amplitude_param="amplitude",
                            missed_gap_factor=self.missed_gap_factor_sampler,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                    },
                ),
            ],
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rng: torch.Generator | None = None,
    ) -> LatentState:
        """Generate pulse-train events and return the latent state."""
        return self._graph(batch_size, device, rng=rng)
