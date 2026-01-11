from __future__ import annotations

from pathlib import Path

import torch

from toyts.core.curriculum import CurriculumSchedule, curriculum_stage_histogram
from toyts.core.events import EventSchema
from toyts.core.pipeline import SynthPipeline
from toyts.core.utils import utils_make_canonical_A0
from toyts.kernels.pqrst import make_pqrst_kernel_bank, pqrst_kernel_size
from toyts.processes.curriculum import CurriculumProcess
from toyts.processes.graph import ProcessGraph, Switch
from toyts.processes.nodes import EventTrainNode, SampleLabelsNode, SetLabelsNode, SingleEventNode
from toyts.views.ecg_leads import ECGLeadsView
from toyts.views.events import EventImpulseView, EventStreamView, KernelConvView
from toyts.views.missingness import MissingnessView
from toyts.views.noise import BaselineWanderView, NoiseView, NormalizeView
from toyts.views.sampling import SamplingAggregationView


def main() -> None:
    """Demonstrate curriculum-style process selection with increasing complexity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 1024
    num_events = 8
    shape_names = ["gaussian", "sharp_laplace", "biphasic_dog"]
    rhythm_names = ["regular", "irregular", "missed_beat"]

    schema = EventSchema(
        type_names=shape_names,
        param_names=["amplitude"],
        time_unit="samples",
    )

    spacing = (seq_len - 1) / (num_events + 1)
    kernel_size = pqrst_kernel_size(spacing=spacing, support_sigma=6.0)
    kernels = make_pqrst_kernel_bank(
        shape_names=shape_names,
        spacing=spacing,
        kernel_size=kernel_size,
        device=device,
    )  # [K=3, T=3, W]
    padding = kernel_size // 2

    A0 = utils_make_canonical_A0(num_leads=1, num_latent=3)  # [C=1, K=3]

    def event_head() -> list[torch.nn.Module]:
        return [
            EventImpulseView(
                seq_len=seq_len,
                amplitude_param="amplitude",
                rounding="nearest",
            ),
            KernelConvView(kernels=kernels, padding=padding),
        ]

    views: dict[str, torch.nn.Module] = {
        "events": EventStreamView(),
        "clean": torch.nn.Sequential(
            *event_head(),
            ECGLeadsView(A0=A0, jitter_std=0.0, max_delay=0),
        ),
        "view1": torch.nn.Sequential(
            *event_head(),
            ECGLeadsView(A0=A0, jitter_std=0.05, max_delay=2),
            NoiseView(noise_std=0.1),
            BaselineWanderView(amplitude_std=0.2, freq_min=0.1, freq_max=0.3),
            SamplingAggregationView(mode="downsample", stride=2),
            NormalizeView(),
        ),
        "view2": torch.nn.Sequential(
            *event_head(),
            ECGLeadsView(A0=A0, jitter_std=0.1, max_delay=4),
            NoiseView(noise_std=0.15),
            BaselineWanderView(amplitude_std=0.4, freq_min=0.05, freq_max=0.2),
            MissingnessView(dropout_prob=0.05, gap_prob=0.1, gap_length=50, hold_prob=0.01),
            NormalizeView(),
        ),
    }

    # ---------------------------------------------------------------------
    # Stage-by-stage: build increasingly complex processes.
    # ---------------------------------------------------------------------
    stage_processes: dict[str, torch.nn.Module] = {
        "stage0_single_event": ProcessGraph(
            name="stage0_single_event",
            outputs={"events"},
            base_meta={"seq_len": seq_len, "num_events": 1},
            graph=[
                SetLabelsNode(
                    labels={
                        "shape": (shape_names, 0),
                        "rhythm": (rhythm_names, 0),
                    }
                ),
                SingleEventNode(
                    seq_len=seq_len,
                    schema=schema,
                    type_name="gaussian",
                    time_min=seq_len // 4,
                    time_max=seq_len - seq_len // 4 - 1,
                    amplitude=1.0,
                    amplitude_param="amplitude",
                    out_key="events",
                ),
            ],
        ),
        "stage1_regular_train": ProcessGraph(
            name="stage1_regular_train",
            outputs={"events"},
            base_meta={"seq_len": seq_len, "num_events": num_events},
            graph=[
                SetLabelsNode(
                    labels={
                        "shape": (shape_names, 0),
                        "rhythm": (rhythm_names, 0),
                    }
                ),
                EventTrainNode(
                    seq_len=seq_len,
                    num_events=num_events,
                    schema=schema,
                    mode="regular",
                    type_label_key=None,
                    type_id=schema.type_id("gaussian"),
                    amplitude=1.0,
                    amplitude_param="amplitude",
                    missed_gap_factor=2.5,
                    out_key="events",
                    centers_out_key="centers",
                ),
            ],
        ),
        "stage2_irregular_train": ProcessGraph(
            name="stage2_irregular_train",
            outputs={"events"},
            base_meta={"seq_len": seq_len, "num_events": num_events},
            graph=[
                SetLabelsNode(
                    labels={
                        "shape": (shape_names, 0),
                        "rhythm": (rhythm_names, 1),
                    }
                ),
                EventTrainNode(
                    seq_len=seq_len,
                    num_events=num_events,
                    schema=schema,
                    mode="irregular",
                    type_label_key=None,
                    type_id=schema.type_id("gaussian"),
                    amplitude=1.0,
                    amplitude_param="amplitude",
                    missed_gap_factor=2.5,
                    out_key="events",
                    centers_out_key="centers",
                ),
            ],
        ),
        "stage3_branched_shape_rhythm": ProcessGraph(
            name="stage3_branched_shape_rhythm",
            outputs={"events"},
            base_meta={"seq_len": seq_len, "num_events": num_events},
            graph=[
                SampleLabelsNode(
                    labels={
                        "shape": shape_names,
                        "rhythm": rhythm_names,
                    }
                ),
                Switch(
                    label_key="rhythm",
                    cases={
                        0: EventTrainNode(
                            seq_len=seq_len,
                            num_events=num_events,
                            schema=schema,
                            mode="regular",
                            type_label_key="shape",
                            type_id=None,
                            amplitude=1.0,
                            amplitude_param="amplitude",
                            missed_gap_factor=2.5,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                        1: EventTrainNode(
                            seq_len=seq_len,
                            num_events=num_events,
                            schema=schema,
                            mode="irregular",
                            type_label_key="shape",
                            type_id=None,
                            amplitude=1.0,
                            amplitude_param="amplitude",
                            missed_gap_factor=2.5,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                        2: EventTrainNode(
                            seq_len=seq_len,
                            num_events=num_events,
                            schema=schema,
                            mode="missed_beat",
                            type_label_key="shape",
                            type_id=None,
                            amplitude=1.0,
                            amplitude_param="amplitude",
                            missed_gap_factor=2.5,
                            out_key="events",
                            centers_out_key="centers",
                        ),
                    },
                ),
            ],
        ),
    }

    print("\n=== Stage-by-stage generation (same views, increasing process complexity) ===")
    for stage_idx, (stage_name, stage_process) in enumerate(stage_processes.items()):
        pipeline = SynthPipeline(process=stage_process, views=views)
        pipeline.to(device)

        rng = torch.Generator(device=device).manual_seed(1000 + stage_idx)
        batch = pipeline(batch_size=4, device=device, rng=rng)

        events_meta = batch["events"].view_meta("EventStreamView")
        events_mask = events_meta["mask"]  # [B, E]
        valid_events = events_mask.sum(dim=1)  # [B]
        clean = batch["clean"].x  # [B, C=1, L]
        view1 = batch["view1"].x  # [B, C=1, L']
        view2 = batch["view2"].x  # [B, C=1, L]

        print(f"\n--- {stage_name} ---")
        print("  valid_events_per_sample:", valid_events.detach().cpu().tolist())
        print("  clean:", tuple(clean.shape), clean.dtype)
        print("  view1:", tuple(view1.shape), view1.dtype)
        print("  view2:", tuple(view2.shape), view2.dtype)
        print(
            "  labels:",
            {key: batch["view1"].y[key].detach().cpu().tolist() for key in batch["view1"].y},
        )

    # ---------------------------------------------------------------------
    # Curriculum: sample a stage per batch using a schedule.
    # ---------------------------------------------------------------------
    schedule = CurriculumSchedule(
        stage_names=list(stage_processes.keys()),
        breakpoints=[
            (0, [1.0, 0.0, 0.0, 0.0]),
            (500, [0.5, 0.5, 0.0, 0.0]),
            (1500, [0.2, 0.5, 0.3, 0.0]),
            (3000, [0.0, 0.2, 0.3, 0.5]),
            (6000, [0.0, 0.0, 0.1, 0.9]),
        ],
    )

    curriculum_process = CurriculumProcess(
        stages=stage_processes,
        schedule=schedule,
        stage_label_key="curriculum_stage",
        initial_step=0,
    )
    curriculum_pipeline = SynthPipeline(process=curriculum_process, views=views)
    curriculum_pipeline.to(device)

    print("\n=== Curriculum schedule: expected vs sampled stage proportions ===")
    histogram_seed = torch.Generator(device=device).manual_seed(1234)
    num_samples = 2048
    for step in (0, 500, 1500, 3000, 6000):
        probs = schedule.probs(step=step, device=device)  # [S]
        cdf = probs.cumsum(dim=0)  # [S]
        u = torch.rand((num_samples,), generator=histogram_seed, device=device)  # [N]
        stage_ids = torch.searchsorted(cdf, u).to(torch.int64).cpu()  # [N]
        counts = curriculum_stage_histogram(stage_ids=stage_ids, num_stages=len(schedule.stage_names))  # [S]
        frac = (counts.to(torch.float32) / float(num_samples)).tolist()  # [S]

        print(f"\nstep={step}")
        print("  expected:", [round(v, 3) for v in probs.detach().cpu().tolist()])
        print("  sampled :", [round(v, 3) for v in frac])

    print("\n=== Curriculum pipeline: sample a few steps and generate batches ===")
    for step in (0, 800, 2000, 4000, 8000):
        curriculum_process.set_step(step)
        rng = torch.Generator(device=device).manual_seed(9000 + step)
        batch = curriculum_pipeline(batch_size=8, device=device, rng=rng)

        process_meta = batch["view1"].meta["process"]
        stage_name = process_meta["stage_name"]
        stage_id = process_meta["stage_id"]
        probs = process_meta["stage_probs"]

        print(f"\nstep={step} -> stage_id={stage_id} stage_name={stage_name}")
        print("  stage_probs:", [round(v, 3) for v in probs])
        print(
            "  curriculum_stage label:",
            batch["view1"].y["curriculum_stage"].detach().cpu().tolist(),
        )

    # ---------------------------------------------------------------------
    # Visualization: show how signal distribution changes by step.
    # ---------------------------------------------------------------------
    import matplotlib.pyplot as plt

    steps = [0, 800, 2000, 4000, 8000]
    per_step_samples = 5

    fig, axes = plt.subplots(
        nrows=len(steps),
        ncols=1,
        figsize=(12, 2.6 * len(steps)),
        sharex=False,
        sharey=False,
    )
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps, strict=True):
        curriculum_process.set_step(step)
        rng = torch.Generator(device=device).manual_seed(9000 + step)
        batch = curriculum_pipeline(batch_size=per_step_samples, device=device, rng=rng)

        signals = batch["clean"].x.detach().cpu()  # [B, C=1, L]
        process_meta = batch["view2"].meta["process"]
        stage_name = process_meta["stage_name"]

        for sample_idx in range(signals.shape[0]):
            ax.plot(
                signals[sample_idx, 0, :].numpy(),
                linewidth=0.9,
                alpha=0.85,
            )

        ax.set_title(f"step={step} stage={stage_name} (clean)")
        ax.set_xlabel("time")
        ax.set_ylabel("x")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "04_curriculum_learning_steps.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved figure", output_path)


if __name__ == "__main__":
    main()
