from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PTBXL_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

PTBXL_LOCATIONS: dict[str, list[int]] = {
    "global": list(range(12)),
    "inferior": [1, 2, 5],
    "lateral": [0, 4, 10, 11],
    "anterior": [7, 8, 9],
    "anteroseptal": [6, 7, 8, 9],
    "anterolateral": [0, 4, 9, 10, 11],
    "inferolateral": [1, 2, 5, 10, 11],
    "posterior": [6, 7],
}

PTBXL_LOCATION_KEYS = list(PTBXL_LOCATIONS.keys())

PTBXL_EVENT_TYPE_NAMES: list[str] = [
    "p",
    "qrs",
    "t",
    "qrs_wide",
    "qrs_delta",
    "qrs_axis_left",
    "qrs_axis_right",
    "qrs_lvh",
    "qrs_rvh",
    "qrs_septal",
    "p_lae",
    "p_rae",
]

for loc in PTBXL_LOCATION_KEYS:
    PTBXL_EVENT_TYPE_NAMES.append(f"qrs_qwave_{loc}")
    PTBXL_EVENT_TYPE_NAMES.append(f"st_shift_{loc}")
    PTBXL_EVENT_TYPE_NAMES.append(f"t_invert_{loc}")

PTBXL_EVENT_TYPE_NAMES += [
    "pace_spike",
    "flutter_wave",
]


@dataclass(frozen=True)
class PTBXLEffectGroup:
    event_type: str
    anchor: Literal["p", "qrs", "t"]
    scale_from: Literal["p", "qrs", "t"]
    scale_range: tuple[float, float]
    sign: float
    codes: tuple[str, ...]


PR_LONG_CODES = ("LPR", "1AVB", "2AVB", "3AVB")
PR_SHORT_CODES = ("WPW",)
QT_LONG_CODES = ("LNGQT",)

QRS_VOLT_LOW_CODES = ("LVOLT",)
QRS_VOLT_HIGH_CODES = ("HVOLT",)

T_LOW_CODES = ("LOWT", "DIG")
T_MILD_CODES = ("NDT", "NT_", "TAB_", "EL")

PAC_CODES = ("PAC",)
PVC_CODES = ("PVC",)
PRC_CODES = ("PRC(S)",)

NORM_CODES = ("NORM",)

MI_LOC = {
    "IMI": "inferior",
    "ASMI": "anteroseptal",
    "AMI": "anterior",
    "ALMI": "anterolateral",
    "LMI": "lateral",
    "ILMI": "inferolateral",
    "IPMI": "posterior",
    "IPLMI": "inferolateral",
    "PMI": "posterior",
}

ISC_LOC = {
    "ISCIN": "inferior",
    "ISCAS": "anteroseptal",
    "ISCAL": "anterolateral",
    "ISCIL": "inferolateral",
    "ISCLA": "lateral",
    "ISCAN": "anterior",
}

INJ_LOC = {
    "INJIN": "inferior",
    "INJAS": "anteroseptal",
    "INJAL": "anterolateral",
    "INJIL": "inferolateral",
    "INJLA": "lateral",
}

QRS_WIDE_MILD = ("IRBBB", "ILBBB", "IVCD", "ABQRS")
QRS_WIDE_STRONG = ("CRBBB", "CLBBB")

QRS_AXIS_LEFT = ("LAFB",)
QRS_AXIS_RIGHT = ("LPFB",)

QRS_LVH = ("LVH", "VCLVH")
QRS_RVH = ("RVH",)
QRS_SEPTAL = ("SEHYP",)

P_LAE = ("LAO/LAE",)
P_RAE = ("RAO/RAE",)

ST_DEPRESS_GLOBAL = ("NST_", "STD_", "ISC_", "DIG")
ST_ELEV_GLOBAL = ("STE_", "ANEUR")

T_INVERT_GLOBAL_MILD = ("NDT", "NT_", "TAB_", "ISC_", "DIG", "EL")
T_INVERT_GLOBAL_STRONG = ("INVT",)


def ptbxl_effect_groups() -> list[PTBXLEffectGroup]:
    groups: list[PTBXLEffectGroup] = []

    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_wide",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.15, 0.35),
            sign=1.0,
            codes=QRS_WIDE_MILD,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_wide",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.35, 0.6),
            sign=1.0,
            codes=QRS_WIDE_STRONG,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_delta",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.1, 0.25),
            sign=1.0,
            codes=("WPW",),
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_axis_left",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.15, 0.35),
            sign=1.0,
            codes=QRS_AXIS_LEFT,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_axis_right",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.15, 0.35),
            sign=1.0,
            codes=QRS_AXIS_RIGHT,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_lvh",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.2, 0.6),
            sign=1.0,
            codes=QRS_LVH,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_rvh",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.2, 0.6),
            sign=1.0,
            codes=QRS_RVH,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_septal",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.2, 0.5),
            sign=1.0,
            codes=QRS_SEPTAL,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="p_lae",
            anchor="p",
            scale_from="p",
            scale_range=(0.3, 0.7),
            sign=1.0,
            codes=P_LAE,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="p_rae",
            anchor="p",
            scale_from="p",
            scale_range=(0.3, 0.7),
            sign=1.0,
            codes=P_RAE,
        )
    )

    for code, loc in ISC_LOC.items():
        groups.append(
            PTBXLEffectGroup(
                event_type=f"st_shift_{loc}",
                anchor="qrs",
                scale_from="qrs",
                scale_range=(0.08, 0.2),
                sign=-1.0,
                codes=(code,),
            )
        )
        groups.append(
            PTBXLEffectGroup(
                event_type=f"t_invert_{loc}",
                anchor="t",
                scale_from="t",
                scale_range=(0.5, 1.0),
                sign=-1.0,
                codes=(code,),
            )
        )

    for code, loc in INJ_LOC.items():
        groups.append(
            PTBXLEffectGroup(
                event_type=f"st_shift_{loc}",
                anchor="qrs",
                scale_from="qrs",
                scale_range=(0.12, 0.3),
                sign=1.0,
                codes=(code,),
            )
        )
        groups.append(
            PTBXLEffectGroup(
                event_type=f"t_invert_{loc}",
                anchor="t",
                scale_from="t",
                scale_range=(0.6, 1.1),
                sign=-1.0,
                codes=(code,),
            )
        )

    for code, loc in MI_LOC.items():
        groups.append(
            PTBXLEffectGroup(
                event_type=f"qrs_qwave_{loc}",
                anchor="qrs",
                scale_from="qrs",
                scale_range=(0.15, 0.45),
                sign=-1.0,
                codes=(code,),
            )
        )
        groups.append(
            PTBXLEffectGroup(
                event_type=f"st_shift_{loc}",
                anchor="qrs",
                scale_from="qrs",
                scale_range=(0.12, 0.3),
                sign=1.0,
                codes=(code,),
            )
        )
        groups.append(
            PTBXLEffectGroup(
                event_type=f"t_invert_{loc}",
                anchor="t",
                scale_from="t",
                scale_range=(0.6, 1.1),
                sign=-1.0,
                codes=(code,),
            )
        )

    groups.append(
        PTBXLEffectGroup(
            event_type="qrs_qwave_global",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.1, 0.3),
            sign=-1.0,
            codes=("QWAVE",),
        )
    )

    groups.append(
        PTBXLEffectGroup(
            event_type="st_shift_global",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.06, 0.15),
            sign=-1.0,
            codes=ST_DEPRESS_GLOBAL,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="st_shift_global",
            anchor="qrs",
            scale_from="qrs",
            scale_range=(0.08, 0.2),
            sign=1.0,
            codes=ST_ELEV_GLOBAL,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="t_invert_global",
            anchor="t",
            scale_from="t",
            scale_range=(0.4, 0.8),
            sign=-1.0,
            codes=T_INVERT_GLOBAL_MILD,
        )
    )
    groups.append(
        PTBXLEffectGroup(
            event_type="t_invert_global",
            anchor="t",
            scale_from="t",
            scale_range=(0.8, 1.3),
            sign=-1.0,
            codes=T_INVERT_GLOBAL_STRONG,
        )
    )

    return groups
