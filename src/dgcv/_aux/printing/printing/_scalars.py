from __future__ import annotations

from typing import Any

from ..._backends._display import latex as _backend_latex
from ..._backends._symbolic_router import _scalar_is_zero
from ..._utilities._config import get_dgcv_settings_registry
from ._string_processing import coeff_needs_parens_latex

joinders = {
    "literal": {
        "general": {"latex": r"\otimes ", "plain": "@"},
        "symmetric": {"latex": r"\odot ", "plain": "&"},
        "skew": {"latex": r"\wedge ", "plain": "*"},
        "dual": {"latex": r"*", "plain": r"^''"},
        "scalar_mul": {"plain": "*", "latex": " "},
    },
    "readable": {
        "general": {"latex": r"\otimes ", "plain": " ⊗ "},
        "symmetric": {"latex": r"\odot ", "plain": " ⊙ "},
        "skew": {"latex": r"\wedge ", "plain": " ∧ "},
        "dual": {"latex": r"*", "plain": "^*"},
        "scalar_mul": {"plain": " * ", "latex": " "},
    },
}


def _print_style() -> str:
    style = get_dgcv_settings_registry().get("print_style")
    return style if style in joinders else "literal"


def _shape_joiner(shape: str, fmt: str) -> str:
    style = _print_style()
    return joinders[style].get(shape, joinders[style]["general"])[fmt]


def _scalar_mul(fmt: str = "plain") -> str:
    style = _print_style()
    return joinders[style]["scalar_mul"][fmt]


def _dual_marker(fmt: str = "plain") -> str:
    style = _print_style()
    return joinders[style]["dual"][fmt]


def _scalar_is_one(x: Any) -> bool:
    io = getattr(
        x, "is_one", None
    )  # present only for deprecated poly subpackage syntax
    if isinstance(io, bool):
        return io
    return _scalar_is_zero(x - 1)


def _scalar_is_minus_one(x: Any) -> bool:
    im1 = getattr(
        x, "is_minus_one", None
    )  # present only for deprecated poly subpackage syntax
    if isinstance(im1, bool):
        return im1
    return _scalar_is_zero(x + 1)


def _coeff_latex(scalar, bypass=None) -> str:
    if _scalar_is_one(scalar):
        return "" if bypass != "" else "1"
    if _scalar_is_minus_one(scalar):
        return "-" if bypass != "" else "-1"
    s = _backend_latex(scalar)
    if coeff_needs_parens_latex(s):
        return rf"\left({s}\right)"
    return s
