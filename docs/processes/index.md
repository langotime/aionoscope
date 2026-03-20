# Processes

- [docs/index.md](../index.md): return to the docs map.

This page is the public process catalog for fast navigation.

## Main Process APIs

- `ProcessGraph` and `ProcessChain`: graph-based latent execution models.
- `PulseTrainProcess`: ECG-like event process with rhythm and morphology structure.
- `TrendSeasonAnomalyProcess`: trend/season/anomaly generator for metric-like signals.
- `ECGProcess`: PTB-XL-inspired ECG event generator with SCP labels.
- `CurriculumProcess`: staged process selection for curriculum-style generation.
- `ConstantProcess`: simple constant latent baseline.

## Main Process Nodes

- Event generators: `SingleEventNode`, `EventTrainNode`.
- Event transforms: `TimeShiftNode`, `TimeJitterNode`, `MapTypeNode`, `DedupeEventsNode`.
- Event composition: `UnionEventsNode`, `GateEventsNode`, `GateEventsByEnabledNode`.
- Label nodes: `SampleLabelNode`, `SampleLabelsNode`, `SampleMultiLabelNode`, `SetLabelsNode`.
- Mixture control: `EnableComponentsNode`.

Use [ARCHITECTURE.md](../../ARCHITECTURE.md) for extension rules and [DOCUMENTATION.md](../../DOCUMENTATION.md) for runnable usage patterns.
