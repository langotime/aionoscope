from __future__ import annotations

import torch

from toyts.core.curriculum import CurriculumSchedule
from toyts.core.events import EventSchema
from toyts.processes.curriculum import CurriculumProcess
from toyts.processes.graph import ProcessGraph
from toyts.processes.nodes import SetLabelsNode, SingleEventNode


def test_curriculum_schedule_piecewise_linear_probs() -> None:
    device = torch.device("cpu")

    schedule = CurriculumSchedule(
        stage_names=["easy", "hard"],
        breakpoints=[
            (0, [1.0, 0.0]),
            (10, [0.0, 1.0]),
        ],
    )

    probs_0 = schedule.probs(step=0, device=device)  # [S]
    probs_5 = schedule.probs(step=5, device=device)  # [S]
    probs_10 = schedule.probs(step=10, device=device)  # [S]

    assert torch.allclose(probs_0, torch.tensor([1.0, 0.0], device=device))
    assert torch.allclose(probs_5, torch.tensor([0.5, 0.5], device=device))
    assert torch.allclose(probs_10, torch.tensor([0.0, 1.0], device=device))


def test_curriculum_process_deterministic_given_seed() -> None:
    device = torch.device("cpu")
    seq_len = 128

    schema = EventSchema(
        type_names=["gaussian", "sharp_laplace", "biphasic_dog"],
        param_names=["amplitude"],
        time_unit="samples",
    )

    stage0 = ProcessGraph(
        name="stage0",
        outputs={"events"},
        base_meta={"seq_len": seq_len},
        graph=[
            SetLabelsNode(
                labels={
                    "shape": (["gaussian"], 0),
                }
            ),
            SingleEventNode(
                seq_len=seq_len,
                schema=schema,
                type_name="gaussian",
                time_min=8,
                time_max=seq_len - 9,
                amplitude_min=1.0,
                amplitude_max=1.0,
                amplitude_param="amplitude",
                out_key="events",
            ),
        ],
    )

    stage1 = ProcessGraph(
        name="stage1",
        outputs={"events"},
        base_meta={"seq_len": seq_len},
        graph=[
            SetLabelsNode(
                labels={
                    "shape": (["gaussian"], 0),
                }
            ),
            SingleEventNode(
                seq_len=seq_len,
                schema=schema,
                type_name="gaussian",
                time_min=8,
                time_max=seq_len - 9,
                amplitude_min=1.0,
                amplitude_max=1.0,
                amplitude_param="amplitude",
                out_key="events",
            ),
        ],
    )

    schedule = CurriculumSchedule(
        stage_names=["stage0", "stage1"],
        breakpoints=[(0, [1.0, 0.0])],
    )

    process = CurriculumProcess(
        stages={"stage0": stage0, "stage1": stage1},
        schedule=schedule,
        stage_label_key="curriculum_stage",
        initial_step=0,
    )

    rng1 = torch.Generator(device=device).manual_seed(1234)
    out1 = process(batch_size=16, device=device, rng=rng1)

    rng2 = torch.Generator(device=device).manual_seed(1234)
    out2 = process(batch_size=16, device=device, rng=rng2)

    assert out1.events is not None
    assert out2.events is not None
    assert torch.equal(out1.events.times, out2.events.times)
    assert torch.equal(out1.events.type_ids, out2.events.type_ids)
    assert torch.equal(out1.events.params, out2.events.params)
    assert torch.equal(out1.events.mask, out2.events.mask)
    assert torch.equal(out1.y["curriculum_stage"], out2.y["curriculum_stage"])
    assert out1.meta["stage_id"] == 0
    assert out2.meta["stage_id"] == 0

