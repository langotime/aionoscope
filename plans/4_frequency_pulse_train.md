# Plan: Frequency-based pulse train

1. Update `PulseTrainProcess` to accept `frequency_hz` and `sample_rate_hz`, derive pulse count, and add random phase offset.
2. Replace `num_pulses` usage across tests, scripts, examples, and docs with the new frequency API.
3. Run `uv run pytest` and address any failures.
