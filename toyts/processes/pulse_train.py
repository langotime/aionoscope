from __future__ import annotations

import math

import torch
from torch import nn

from ..core.events import EventSchema
from ..core.types import LatentState
from .graph import ProcessGraph, Switch
from .nodes import EventTrainNode, SampleLabelsNode


class PulseTrainProcess(nn.Module):
    """A process that generates a train of pulses as an event stream.

    This module samples rhythm and shape labels, generates event centers, and
    emits an `EventBatch` where the event type encodes the pulse shape.

    Args:
        seq_len: The length of the generated sequence `L`.
        frequency_hz: The pulse frequency in Hz. The actual pulse count is
            `round(frequency_hz * duration_sec)`.
        sample_rate_hz: The sampling rate used to interpret `seq_len` in seconds.
        rhythm_classes: A list of rhythm types to sample from. Must include
            "regular", "irregular", and "missed_beat".
        shape_classes: A list of pulse shape types to sample from. Must include
            "gaussian", "sharp_laplace", and "biphasic_dog".
        latent_mode: The structure of the latent components. Currently, only
            "pqrst3" is supported.
        amplitude: The per-event amplitude (stored in event params).
        missed_gap_factor: In the "missed_beat" rhythm, the factor by which the
            interval is increased to simulate a pause. Must be > 1.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        frequency_hz: float,
        sample_rate_hz: float,
        rhythm_classes: list[str],
        shape_classes: list[str],
        latent_mode: str,
        amplitude: float = 1.0,
        missed_gap_factor: float = 2.5,
    ) -> None:
        """Initialize a pulse-train event process."""
        super().__init__()

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        if frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz}.")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}.")
        if not rhythm_classes:
            raise ValueError("rhythm_classes must be non-empty.")
        if not shape_classes:
            raise ValueError("shape_classes must be non-empty.")
        if latent_mode != "pqrst3":
            raise ValueError("Only latent_mode='pqrst3' is supported in MVP.")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}.")
        if missed_gap_factor <= 1:
            raise ValueError(
                f"missed_gap_factor must be >1 to create a pause, got {missed_gap_factor}."
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
        duration_sec = (seq_len - 1) / sample_rate_hz
        expected_pulses = frequency_hz * duration_sec
        num_pulses = int(math.floor(expected_pulses + 0.5))
        if num_pulses <= 0:
            raise ValueError(
                "frequency_hz and sample_rate_hz yield fewer than 1 pulse "
                f"for seq_len={seq_len}. "
                f"Got duration_sec={duration_sec:.6f}, expected_pulses={expected_pulses:.6f}."
            )

        self.frequency_hz = frequency_hz
        self.sample_rate_hz = sample_rate_hz
        self.duration_sec = duration_sec
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
        spacing = (self.seq_len - 1) / (self.num_pulses + 1)

        base_meta = {
            "seq_len": self.seq_len,
            "frequency_hz": self.frequency_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_sec": self.duration_sec,
            "num_pulses": self.num_pulses,
            "latent_mode": self.latent_mode,
            "shape_names": self.shape_classes,
            "rhythm_names": self.rhythm_classes,
            "amplitude": self.amplitude,
            "missed_gap_factor": self.missed_gap_factor,
            "spacing_samples": spacing,
        }

        self._graph = ProcessGraph(
            name="PulseTrainProcess",
            outputs={"events", "centers"},
            base_meta=base_meta,
            graph=[
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
                            num_events=self.num_pulses,
                            schema=self.schema,
                            mode="regular",
                            type_label_key="shape",
                            type_id=None,
                            amplitude_min=self.amplitude,
                            amplitude_max=self.amplitude,
                            amplitude_param="amplitude",
                            missed_gap_factor=self.missed_gap_factor,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                        self._irregular_index: EventTrainNode(
                            seq_len=self.seq_len,
                            num_events=self.num_pulses,
                            schema=self.schema,
                            mode="irregular",
                            type_label_key="shape",
                            type_id=None,
                            amplitude_min=self.amplitude,
                            amplitude_max=self.amplitude,
                            amplitude_param="amplitude",
                            missed_gap_factor=self.missed_gap_factor,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                        self._missed_index: EventTrainNode(
                            seq_len=self.seq_len,
                            num_events=self.num_pulses,
                            schema=self.schema,
                            mode="missed_beat",
                            type_label_key="shape",
                            type_id=None,
                            amplitude_min=self.amplitude,
                            amplitude_max=self.amplitude,
                            amplitude_param="amplitude",
                            missed_gap_factor=self.missed_gap_factor,
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
