from __future__ import annotations

import math

import torch

from toyts import EventBatch, EventRenderView, EventSchema, LatentState


def _make_state(*, events: EventBatch, seq_len: int, sample_rate_hz: float) -> LatentState:
    batch_size = events.times.shape[0]
    device = events.times.device
    latent = torch.zeros((batch_size, 1, seq_len), device=device, dtype=torch.float32)  # [B, 1, L]
    centers = torch.empty((batch_size, 0), device=device)  # [B, 0]
    return LatentState(
        centers=centers,
        latent=latent,
        events=events,
        y={},
        meta={"seq_len": seq_len, "sample_rate_hz": sample_rate_hz},
    )


def test_event_render_view_spikes_sum() -> None:
    device = torch.device("cpu")
    seq_len = 8

    schema = EventSchema(
        type_names=["spike"],
        param_names=["amplitude"],
        time_unit="samples",
    )
    spike_id = schema.type_id("spike")
    amp_idx = schema.param_id("amplitude")

    times = torch.tensor([[2.0, 2.0]], device=device)  # [B=1, E=2]
    type_ids = torch.tensor([[spike_id, spike_id]], device=device, dtype=torch.int64)  # [1, 2]
    params = torch.zeros((1, 2, 1), device=device, dtype=torch.float32)  # [1, 2, P=1]
    params[0, 0, amp_idx] = 1.0
    params[0, 1, amp_idx] = 2.0
    mask = torch.tensor([[True, True]], device=device)  # [1, 2]

    events = EventBatch(times=times, type_ids=type_ids, params=params, mask=mask, schema=schema, meta={})
    state = _make_state(events=events, seq_len=seq_len, sample_rate_hz=10.0)

    view = EventRenderView(
        seq_len=seq_len,
        amplitude_param="amplitude",
        rounding="nearest",
    )
    obs = view(state)
    expected = torch.zeros((1, 1, seq_len), device=device)  # [1, 1, L]
    expected[0, 0, 2] = 3.0
    torch.testing.assert_close(obs.x, expected)


def test_event_render_view_level_change() -> None:
    device = torch.device("cpu")
    seq_len = 8

    schema = EventSchema(
        type_names=["level_change"],
        param_names=["amplitude"],
        time_unit="samples",
    )
    level_id = schema.type_id("level_change")
    amp_idx = schema.param_id("amplitude")

    times = torch.tensor([[3.0]], device=device)  # [1, 1]
    type_ids = torch.tensor([[level_id]], device=device, dtype=torch.int64)  # [1, 1]
    params = torch.zeros((1, 1, 1), device=device, dtype=torch.float32)  # [1, 1, 1]
    params[0, 0, amp_idx] = 0.5
    mask = torch.tensor([[True]], device=device)  # [1, 1]

    events = EventBatch(times=times, type_ids=type_ids, params=params, mask=mask, schema=schema, meta={})
    state = _make_state(events=events, seq_len=seq_len, sample_rate_hz=10.0)

    view = EventRenderView(
        seq_len=seq_len,
        amplitude_param="amplitude",
        rounding="nearest",
    )
    obs = view(state)
    expected = torch.zeros((1, 1, seq_len), device=device)  # [1, 1, L]
    expected[0, 0, 3:] = 0.5
    torch.testing.assert_close(obs.x, expected)


def test_event_render_view_gaussian_bump() -> None:
    device = torch.device("cpu")
    seq_len = 9
    sample_rate_hz = 10.0

    schema = EventSchema(
        type_names=["gaussian"],
        param_names=["amplitude", "sigma_sec"],
        time_unit="samples",
    )
    gauss_id = schema.type_id("gaussian")
    amp_idx = schema.param_id("amplitude")
    sigma_idx = schema.param_id("sigma_sec")

    times = torch.tensor([[4.0]], device=device)  # [1, 1]
    type_ids = torch.tensor([[gauss_id]], device=device, dtype=torch.int64)  # [1, 1]
    params = torch.zeros((1, 1, 2), device=device, dtype=torch.float32)  # [1, 1, 2]
    params[0, 0, amp_idx] = 1.0
    params[0, 0, sigma_idx] = 0.1
    mask = torch.tensor([[True]], device=device)  # [1, 1]

    events = EventBatch(times=times, type_ids=type_ids, params=params, mask=mask, schema=schema, meta={})
    state = _make_state(events=events, seq_len=seq_len, sample_rate_hz=sample_rate_hz)

    view = EventRenderView(
        seq_len=seq_len,
        amplitude_param="amplitude",
        rounding="nearest",
        sigma_sec_param="sigma_sec",
    )
    obs = view(state)

    sigma_samples = 0.1 * sample_rate_hz
    expected = torch.zeros((1, 1, seq_len), device=device)  # [1, 1, L]
    for t in range(seq_len):
        expected[0, 0, t] = math.exp(-0.5 * ((t - 4.0) / sigma_samples) ** 2)
    torch.testing.assert_close(obs.x, expected, rtol=1e-5, atol=1e-5)


def test_event_render_view_rect_pulse_and_decay_and_ringdown() -> None:
    device = torch.device("cpu")
    seq_len = 8
    sample_rate_hz = 1.0

    schema = EventSchema(
        type_names=["rect_pulse", "exp_decay", "ringdown"],
        param_names=["amplitude", "duration_sec", "tau_sec", "frequency_hz", "phase"],
        time_unit="samples",
    )
    amp_idx = schema.param_id("amplitude")
    dur_idx = schema.param_id("duration_sec")
    tau_idx = schema.param_id("tau_sec")
    freq_idx = schema.param_id("frequency_hz")
    phase_idx = schema.param_id("phase")

    rect_id = schema.type_id("rect_pulse")
    decay_id = schema.type_id("exp_decay")
    ring_id = schema.type_id("ringdown")

    times = torch.tensor([[2.0, 2.0, 1.0]], device=device)  # [1, 3]
    type_ids = torch.tensor([[rect_id, decay_id, ring_id]], device=device, dtype=torch.int64)  # [1, 3]
    params = torch.zeros((1, 3, 5), device=device, dtype=torch.float32)  # [1, 3, 5]

    # rect_pulse: amplitude 2.0, duration 2 sec -> samples=2 (sample_rate_hz=1)
    params[0, 0, amp_idx] = 2.0
    params[0, 0, dur_idx] = 2.0

    # exp_decay: amplitude 1.0, tau=1.0
    params[0, 1, amp_idx] = 1.0
    params[0, 1, tau_idx] = 1.0

    # ringdown: amplitude 1.0, tau=1.0, freq=0.25 Hz, phase=0
    params[0, 2, amp_idx] = 1.0
    params[0, 2, tau_idx] = 1.0
    params[0, 2, freq_idx] = 0.25
    params[0, 2, phase_idx] = 0.0

    mask = torch.tensor([[True, True, True]], device=device)  # [1, 3]
    events = EventBatch(times=times, type_ids=type_ids, params=params, mask=mask, schema=schema, meta={})
    state = _make_state(events=events, seq_len=seq_len, sample_rate_hz=sample_rate_hz)

    view = EventRenderView(
        seq_len=seq_len,
        amplitude_param="amplitude",
        rounding="nearest",
        duration_sec_param="duration_sec",
        tau_sec_param="tau_sec",
        frequency_hz_param="frequency_hz",
        phase_param="phase",
    )
    obs = view(state)

    expected = torch.zeros((1, 1, seq_len), device=device)  # [1, 1, L]

    # rect_pulse contributes 2.0 at t=2,3
    expected[0, 0, 2:4] += 2.0

    # exp_decay from t=2: exp(-(t-2)/1)
    for t in range(seq_len):
        if t >= 2:
            expected[0, 0, t] += math.exp(-(t - 2))

    # ringdown from t=1: exp(-(t-1)) * sin(2*pi*0.25*(t-1))
    for t in range(seq_len):
        if t >= 1:
            dt = t - 1
            expected[0, 0, t] += math.exp(-dt) * math.sin(2.0 * math.pi * 0.25 * dt)

    torch.testing.assert_close(obs.x, expected, rtol=1e-5, atol=1e-5)

