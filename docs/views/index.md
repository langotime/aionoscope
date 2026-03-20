# Views

- [docs/index.md](../index.md): return to the docs map.

This page is the public observation-model catalog for fast navigation.

## Event And Rendering Views

- `EventStreamView`: expose events directly.
- `EventImpulseView`: convert event batches into impulse trains.
- `KernelConvView`: render impulses into dense latent components or channels.
- `EventRenderView`: direct additive rendering for small event families.
- `ECGLeadsView`: latent-to-lead mixing for ECG-style observations.

## Signal Modification Views

- Noise: `GaussianNoiseView`, `UniformNoiseView`, `RandomWalkNoiseView`, `BaselineWanderView`, `ColoredNoiseView`, `LaplaceNoiseView`.
- Periodic: `SineWaveView`, `SawtoothWaveView`, `SquareWaveView`, `TriangleWaveView`, `ChirpView`, `DampedSineWaveView`.
- Trends: `LinearTrendView`, `QuadraticTrendView`, `LogTrendView`, `ExponentialTrendView`, `PiecewiseLinearTrendView`, `SigmoidTrendView`.
- Units and scaling: `UnitsAbsoluteView`, `UnitsPercentOfCapacityView`, `NormalizeView`, `ClippingView`.
- Sampling and missingness: `SamplingAggregationView`, `MissingnessView`.

Use [ARCHITECTURE.md](../../ARCHITECTURE.md) for boundary rules and [DOCUMENTATION.md](../../DOCUMENTATION.md) for example-oriented guidance.
