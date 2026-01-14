from __future__ import annotations

import importlib
import sys

from .toyts import *  # noqa: F403
from .toyts import __all__ as __all__  # noqa: F401

# Expose ToyTS subpackages from the nested layout so imports like `toyts.views.*`
# work when running from the monorepo root (where `toyts/` is a wrapper package).
_ALIAS_SUBPACKAGES = (
    "core",
    "datasets",
    "kernels",
    "processes",
    "views",
)
for _name in _ALIAS_SUBPACKAGES:
    _module = importlib.import_module(f"{__name__}.toyts.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
