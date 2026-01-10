from __future__ import annotations

import torch

from toyts import MissingnessView, Observation, ViewChain


def test_missingness_mask_regen() -> None:
    device = torch.device("cpu")

    x = torch.arange(0, 24, device=device, dtype=torch.float32).view(2, 1, 12)  # [B, C, L]
    observation = Observation(
        x=x,
        y={},
        meta={"process": {"process": "Dummy"}},
    )

    view = MissingnessView(
        dropout_prob=0.2,
        gap_prob=0.4,
        gap_length=3,
        hold_prob=0.3,
    )
    chain = ViewChain(view)

    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    output = chain(observation, rng=generator)
    meta = output.view_meta("MissingnessView")

    masks = MissingnessView.sample_masks(
        meta,
        shape=x.shape,
        device=device,
    )

    reconstructed = x * masks["dropout_mask"]  # [B, C, L]
    reconstructed = reconstructed * masks["gap_mask"]  # [B, C, L]
    previous = torch.cat(
        [reconstructed[:, :, :1], reconstructed[:, :, :-1]],
        dim=2,
    )  # [B, C, L]
    reconstructed = torch.where(
        masks["hold_mask"],
        previous,
        reconstructed,
    )  # [B, C, L]

    torch.testing.assert_close(reconstructed, output.x)
