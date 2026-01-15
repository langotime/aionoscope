from __future__ import annotations

from pathlib import Path

# The ToyTS library lives under the nested `toyts/toyts/` directory in this monorepo.
# We extend the package search path so imports like `toyts.core.*` work without
# introducing duplicate module namespaces (which breaks isinstance checks).
_NESTED_PACKAGE_DIR = Path(__file__).resolve().parent / "toyts"
if not _NESTED_PACKAGE_DIR.is_dir():
    raise ImportError(
        "Failed to import ToyTS: expected nested sources at "
        f"{_NESTED_PACKAGE_DIR}. Ensure you're running from the monorepo checkout."
    )
__path__.append(str(_NESTED_PACKAGE_DIR))

from .core.curriculum import (  # noqa: E402
    CurriculumSchedule,
    curriculum_sample_stage_id,
    curriculum_stage_histogram,
)
from .core.events import EventBatch, EventSchema  # noqa: E402
from .core.pipeline import SynthPipeline  # noqa: E402
from .core.samplers import (  # noqa: E402
    BernoulliSampler,
    CategoricalSampler,
    ChoiceSampler,
    ConstantSampler,
    LogUniformSampler,
    NormalSampler,
    RandIntSampler,
    Sampler,
    SamplerLike,
    UniformSampler,
)
from .core.types import LatentState, Observation  # noqa: E402
from .kernels.pqrst import make_pqrst_kernel_bank, pqrst_kernel_size  # noqa: E402
from .processes.curriculum import CurriculumProcess  # noqa: E402
from .processes.constant import ConstantLatentNode, ConstantProcess  # noqa: E402
from .processes.pulse_train import PulseTrainProcess  # noqa: E402
from .processes.graph import (  # noqa: E402
    Parallel,
    ProcessChain,
    ProcessGraph,
    ProcessOp,
    ProcessState,
    Scope,
    Seq,
    Switch,
)
from .processes.nodes import (  # noqa: E402
    DedupeEventsNode,
    EnableComponentsNode,
    EventTrainNode,
    GateEventsByEnabledNode,
    GateEventsNode,
    MapTypeNode,
    SampleLabelsNode,
    SetLabelsNode,
    SingleEventNode,
    TimeJitterNode,
    TimeShiftNode,
    UnionEventsNode,
)
from .processes.trend_season import TrendSeasonAnomalyProcess  # noqa: E402
from .views.ecg_leads import ECGLeadsView  # noqa: E402
from .views.events import EventImpulseView, EventStreamView, KernelConvView  # noqa: E402
from .views.events_basic import EventRenderView  # noqa: E402
from .views.noise import (  # noqa: E402
    BaselineWanderView,
    BrownNoiseView,
    ColoredNoiseView,
    GaussianNoiseView,
    LaplaceNoiseView,
    RandomWalkNoiseView,
    UniformNoiseView,
)
from .views.periodic import (  # noqa: E402
    ChirpView,
    DampedSineWaveView,
    SawtoothWaveView,
    SineWaveView,
    SquareWaveView,
    TriangleWaveView,
)
from .views.trend import (  # noqa: E402
    ExponentialTrendView,
    LinearTrendView,
    LogTrendView,
    PiecewiseLinearTrendView,
    QuadraticTrendView,
    SigmoidTrendView,
)
from .views.units import (  # noqa: E402
    ClippingView,
    NormalizeView,
    UnitsAbsoluteView,
    UnitsPercentOfCapacityView,
)
from .views.sampling import SamplingAggregationView  # noqa: E402
from .views.missingness import MissingnessView  # noqa: E402
from .views.base import View, ViewChain  # noqa: E402
from .processes.base import Process  # noqa: E402

__all__ = [
    "BaselineWanderView",
    "BrownNoiseView",
    "BernoulliSampler",
    "CategoricalSampler",
    "ChirpView",
    "ChoiceSampler",
    "ClippingView",
    "ColoredNoiseView",
    "ConstantLatentNode",
    "ConstantProcess",
    "ConstantSampler",
    "CurriculumProcess",
    "CurriculumSchedule",
    "DampedSineWaveView",
    "ECGLeadsView",
    "DedupeEventsNode",
    "EnableComponentsNode",
    "ExponentialTrendView",
    "EventBatch",
    "EventImpulseView",
    "EventRenderView",
    "EventSchema",
    "EventStreamView",
    "EventTrainNode",
    "GaussianNoiseView",
    "GateEventsByEnabledNode",
    "GateEventsNode",
    "KernelConvView",
    "LaplaceNoiseView",
    "LinearTrendView",
    "LogUniformSampler",
    "LogTrendView",
    "LatentState",
    "MissingnessView",
    "MapTypeNode",
    "NormalSampler",
    "Parallel",
    "NormalizeView",
    "Observation",
    "PiecewiseLinearTrendView",
    "Process",
    "ProcessChain",
    "ProcessGraph",
    "ProcessOp",
    "ProcessState",
    "PulseTrainProcess",
    "QuadraticTrendView",
    "RandIntSampler",
    "RandomWalkNoiseView",
    "SawtoothWaveView",
    "Sampler",
    "SamplerLike",
    "SampleLabelsNode",
    "SetLabelsNode",
    "SigmoidTrendView",
    "SineWaveView",
    "SquareWaveView",
    "Scope",
    "Seq",
    "SamplingAggregationView",
    "SingleEventNode",
    "SynthPipeline",
    "Switch",
    "TimeJitterNode",
    "TimeShiftNode",
    "TrendSeasonAnomalyProcess",
    "TriangleWaveView",
    "UnionEventsNode",
    "UniformNoiseView",
    "UnitsAbsoluteView",
    "UnitsPercentOfCapacityView",
    "UniformSampler",
    "View",
    "ViewChain",
    "make_pqrst_kernel_bank",
    "curriculum_sample_stage_id",
    "curriculum_stage_histogram",
    "pqrst_kernel_size",
]
