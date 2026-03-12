from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core.samplers import Sampler
from .phenotypes import (
    MI_LOC,
    PAC_CODES,
    PR_LONG_CODES,
    PR_SHORT_CODES,
    PRC_CODES,
    PVC_CODES,
    QRS_AXIS_LEFT,
    QRS_AXIS_RIGHT,
    QRS_LVH,
    QRS_RVH,
    QRS_SEPTAL,
    QRS_VOLT_HIGH_CODES,
    QRS_VOLT_LOW_CODES,
    QRS_WIDE_MILD,
    QRS_WIDE_STRONG,
)
from .scp import ptbxl_all_codes, ptbxl_codes_by_group


@dataclass(frozen=True)
class PTBXLLabelSetSampler(Sampler):
    """Sample multi-label PTB-XL SCP code sets with rhythm + diagnostic/form codes."""

    scp_codes: list[str]
    normal_prob: float
    family_probs: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.scp_codes:
            raise ValueError("scp_codes must be non-empty.")
        if len(set(self.scp_codes)) != len(self.scp_codes):
            raise ValueError("scp_codes must be unique.")
        if self.scp_codes != ptbxl_all_codes():
            raise ValueError(
                "PTBXLLabelSetSampler requires scp_codes in PTB-XL CSV order. "
                "Use ptbxl_all_codes()."
            )
        if not (0.0 <= self.normal_prob <= 1.0):
            raise ValueError(
                "normal_prob must be in [0, 1]. "
                f"Got {self.normal_prob}."
            )

        if self.family_probs is None:
            object.__setattr__(
                self,
                "family_probs",
                {
                    "pr": 0.18,
                    "conduction": 0.18,
                    "qt": 0.12,
                    "qwave": 0.1,
                    "mi": 0.12,
                    "ischemia": 0.14,
                    "injury": 0.12,
                    "stt_nonspecific": 0.2,
                    "hypertrophy": 0.18,
                    "voltage": 0.12,
                    "ectopy": 0.16,
                },
            )
        for key, value in self.family_probs.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"family_probs['{key}'] must be in [0, 1]. "
                    f"Got {value}."
                )

    def _family_codes(self) -> dict[str, list[str]]:
        return {
            "pr": list(PR_LONG_CODES + PR_SHORT_CODES),
            "conduction": list(QRS_WIDE_MILD + QRS_WIDE_STRONG + QRS_AXIS_LEFT + QRS_AXIS_RIGHT),
            "qt": ["LNGQT"],
            "qwave": ["QWAVE"],
            "mi": list(MI_LOC.keys()),
            "ischemia": ["ISC_"] + ["ISCAN", "ISCAS", "ISCAL", "ISCIN", "ISCIL", "ISCLA"],
            "injury": ["INJAS", "INJAL", "INJIN", "INJIL", "INJLA"],
            "stt_nonspecific": [
                "NST_",
                "STD_",
                "STE_",
                "NDT",
                "NT_",
                "TAB_",
                "INVT",
                "LOWT",
                "DIG",
                "EL",
                "ANEUR",
            ],
            "hypertrophy": list(QRS_LVH + QRS_RVH + QRS_SEPTAL + ("LAO/LAE", "RAO/RAE")),
            "voltage": list(QRS_VOLT_LOW_CODES + QRS_VOLT_HIGH_CODES),
            "ectopy": list(PAC_CODES + PVC_CODES + PRC_CODES),
        }

    def sample(
        self,
        *,
        shape: tuple[int, ...],
        rng: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is not torch.bool:
            raise ValueError("PTBXLLabelSetSampler requires dtype=torch.bool.")
        if len(shape) != 2:
            raise ValueError(
                "PTBXLLabelSetSampler requires shape [B, S]. "
                f"Got shape={shape}."
            )
        batch_size, num_codes = shape
        if num_codes != len(self.scp_codes):
            raise ValueError(
                "PTBXLLabelSetSampler expected num_codes matching scp_codes. "
                f"Got num_codes={num_codes}, len(scp_codes)={len(self.scp_codes)}."
            )

        labels = torch.zeros((batch_size, num_codes), device=device, dtype=torch.bool)  # [B, S]
        code_to_index = {code: idx for idx, code in enumerate(self.scp_codes)}

        rhythm_codes = ptbxl_codes_by_group("rhythm")
        rhythm_indices = torch.tensor(
            [code_to_index[code] for code in rhythm_codes],
            device=device,
            dtype=torch.int64,
        )  # [R]
        rhythm_probs = torch.ones((len(rhythm_codes),), device=device) / len(rhythm_codes)  # [R]
        rhythm_choice = torch.multinomial(
            rhythm_probs,
            num_samples=batch_size,
            replacement=True,
            generator=rng,
        )  # [B]
        labels[torch.arange(batch_size, device=device), rhythm_indices[rhythm_choice]] = True

        normal_mask = torch.rand((batch_size,), generator=rng, device=device) < self.normal_prob  # [B]
        if torch.any(normal_mask):
            norm_idx = code_to_index["NORM"]
            labels[normal_mask, norm_idx] = True

        active_mask = ~normal_mask  # [B]
        families = self._family_codes()
        for family, codes in families.items():
            prob = self.family_probs.get(family, 0.0)
            if prob <= 0:
                continue
            family_active = (
                torch.rand((batch_size,), generator=rng, device=device) < prob
            ) & active_mask  # [B]
            if not torch.any(family_active):
                continue
            indices = torch.tensor(
                [code_to_index[code] for code in codes],
                device=device,
                dtype=torch.int64,
            )  # [F]
            pick = torch.randint(
                0,
                len(codes),
                (batch_size,),
                generator=rng,
                device=device,
            )  # [B]
            chosen = indices[pick]  # [B]
            labels[family_active, chosen[family_active]] = True

        return labels

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "ptbxl_label_set",
            "normal_prob": self.normal_prob,
            "family_probs": dict(self.family_probs or {}),
        }
