from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True, order=True)
class version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> "version":
        s = s.strip()
        if s.startswith(("v", "V")):
            s = s[1:]
        parts = (s.split(".") + ["0", "0"])[:3]
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


baseline_defaults: Dict[str, Any] = {
    "use_latex": False,
    "theme": "blue",
    "format_displays": False,
    "version_specific_defaults": None,
    "ask_before_overwriting_objects_in_vmf": True,
    "forgo_warnings": False,
    "default_symbolic_engine": "sympy",
    "verbose_label_printing": True,
    "VLP": None,
    "extra_support_for_math_in_tables": False,
}


changes: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("0.2.6", {"theme": "appalachian"}),
    ("0.3.0", {"theme": "graph_paper", "verbose_label_printing": False}),
    (
        "0.3.13",
        {
            "use_latex": True,
            "format_displays": True,
            "extra_support_for_math_in_tables": "infer",
        },
    ),
)


_parsed_changes = tuple((version.parse(v), patch) for v, patch in changes)


def needs_sympy_hook(target_version: str) -> bool:
    v = version.parse(target_version)
    return v <= version.parse("0.3.10")


def defaults_for_version(
    target_version: str,
    *,
    current_version: str,
    vlp: Any,
) -> Dict[str, Any]:
    target = version.parse(target_version)
    out = deepcopy(baseline_defaults)

    for v, patch in _parsed_changes:
        if v <= target:
            out.update(patch)

    out["VLP"] = vlp
    out["version_specific_defaults"] = f"v{target.major}.{target.minor}.{target.patch}"
    return out
