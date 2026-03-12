from __future__ import annotations

import torch

from aiono import EventSchema, ProcessGraph, SingleEventNode, TimeJitterNode


def test_time_jitter_samples_meta() -> None:
    device = torch.device("cpu")

    schema = EventSchema(
        type_names=["pulse"],
        param_names=["amplitude"],
        time_unit="samples",
    )

    process = ProcessGraph(
        name="TimeJitterMeta",
        outputs={"events"},
        base_meta={"seq_len": 16},
        graph=[
            SingleEventNode(
                seq_len=16,
                schema=schema,
                type_name="pulse",
                time_min=0,
                time_max=0,
                amplitude=1.0,
                amplitude_param="amplitude",
                out_key="events",
            ),
            TimeJitterNode(
                in_key="events",
                out_key="events",
                jitter_std=0.2,
            ),
        ],
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(77)

    batch_size = 2
    state = process(batch_size=batch_size, device=device, rng=generator)

    samples = state.meta["samples"]
    node_samples = samples["TimeJitterNode:events"]
    time_jitter = node_samples["time_jitter"]  # [B, E]

    assert time_jitter.shape == state.events.times.shape
    torch.testing.assert_close(state.events.times, time_jitter)
