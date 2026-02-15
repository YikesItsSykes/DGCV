"""
package: dgcv - Differential Geometry with Complex Variables
module: _settings

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import warnings
from typing import Literal, Optional

from dgcv import __version__

from ._config import get_dgcv_settings_registry, vlp
from .backends._cls_coercion import attach_sympy_hook, detach_sympy_hook
from .backends._engine import (
    invalidate_engine_cache,
    is_sage_available,
    is_sympy_available,
)
from .backends._notebooks import invalidate_notebook_cache, is_ipython_available
from .backends._updates import needs_sympy_hook
from .base import dgcv_class

__all__ = ["set_dgcv_settings", "view_dgcv_settings", "reset_dgcv_settings"]
# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
_infer_vscode_jupyter_unspecified = object()


def _infer_vscode_jupyter():
    cache = getattr(_infer_vscode_jupyter, "_cache", _infer_vscode_jupyter_unspecified)
    if cache is not _infer_vscode_jupyter_unspecified:
        return cache
    try:
        import os

        try:
            from IPython import get_ipython

            ip = get_ipython()
            in_jupyter = ip is not None and ip.__class__.__name__.startswith("ZMQ")
        except Exception:
            in_jupyter = False

        in_vscode = (
            "VSCODE_PID" in os.environ
            or "VSCODE_IPC_HOOK" in os.environ
            or os.environ.get("TERM_PROGRAM") == "vscode"
        )
        val = bool(in_jupyter and in_vscode)
    except Exception:
        val = False
    _infer_vscode_jupyter._cache = val
    return val


def set_dgcv_settings(
    theme: Optional[str] = None,
    format_displays: bool | None = None,
    use_latex: bool | None = None,
    print_style: Optional[Literal["readable", "literal"]] = None,
    version_specific_defaults: Optional[str] = None,
    ask_before_overwriting_objects_in_vmf: bool | None = None,
    forgo_warnings: bool | None = None,
    default_engine: Optional[Literal["sage", "sympy", "dgcv_custom"]] = None,
    verbose_label_printing: bool | None = None,
    pass_solve_requests_to_symbolic_engine: bool | None = None,
    DEBUG: bool | None = None,
    extra_support_for_math_in_tables: bool | None = None,
    preferred_variable_format: Optional[Literal["complex", "real"]] = None,
    use_numeric_methods: bool | None = None,
    force_rich_display: bool | None = None,
    **kwargs,
):
    dgcvSR = get_dgcv_settings_registry()

    # depricated_keywords
    _depr_kw = "apply_awkward_workarounds_to_fix_VSCode_display_issues"
    if _depr_kw in kwargs:
        warnings.warn(
            "The settings keyword `apply_awkward_workarounds_to_fix_VSCode_display_issues` is deprecated. "
            "Use `extra_support_for_math_in_tables` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if extra_support_for_math_in_tables is None:
            extra_support_for_math_in_tables = kwargs.get(_depr_kw)

    def _norm_v(v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if s[0] in ("v", "V"):
            return "v" + s[1:]
        return "v" + s

    def _resolve_vscode_patch_value(v):
        if v == "infer":
            return _infer_vscode_jupyter()
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        warnings.warn(
            "dgcv: extra_support_for_math_in_tables should be True/False or 'infer'; "
            f"coercing {v!r} to bool.",
            stacklevel=2,
        )
        return bool(v)

    def _set_engine_symbolic(new_engine):
        if dgcvSR.get("default_symbolic_engine") != new_engine:
            dgcvSR["default_symbolic_engine"] = new_engine
            invalidate_engine_cache()

    def _apply_keyval(k, v):
        if k == "default_symbolic_engine":
            if v in ("sage", "sagemath"):
                if is_sage_available():
                    _set_engine_symbolic("sage")
                else:
                    warnings.warn(
                        "dgcv: requested default_engine='sage' but Sage is not available; "
                        "default_symbolic_engine was not changed.",
                        stacklevel=2,
                    )
            elif v in ("sympy",):
                if is_sympy_available():
                    _set_engine_symbolic("sympy")
                else:
                    warnings.warn(
                        "dgcv: requested default_engine='sympy' but SymPy is not available; "
                        "default_symbolic_engine was not changed.",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"dgcv: unrecognized default_engine value {v!r}. "
                    "Supported options are 'sympy' and 'sage'. Default_symbolic_engine was not changed.",
                    stacklevel=2,
                )
            return

        if k == "format_displays":
            new_val = bool(v)
            old_val = bool(dgcvSR.get("format_displays", False))
            dgcvSR["format_displays"] = new_val
            if new_val != old_val:
                invalidate_notebook_cache()
            if new_val is True:
                from ._dgcv_display import dgcv_init_printing

                dgcv_init_printing()
            return

        if k == "extra_support_for_math_in_tables":
            dgcvSR[k] = _resolve_vscode_patch_value(v)
            return

        if k == "verbose_label_printing":
            dgcvSR["verbose_label_printing"] = v
            if dgcvSR["verbose_label_printing"] is False:
                dgcvSR["VLP"] = vlp
            return

        dgcvSR[k] = v

    current_vsd = _norm_v(f"v{__version__}")

    requested_vsd = (
        _norm_v(version_specific_defaults)
        if version_specific_defaults is not None
        else None
    )

    if requested_vsd is not None:
        if requested_vsd != current_vsd:
            from dgcv.backends._updates import defaults_for_version

            version_defaults = defaults_for_version(
                requested_vsd,
                current_version=__version__,
                vlp=vlp,
            )

            for k, v in version_defaults.items():
                if k == "version_specific_defaults":
                    continue
                _apply_keyval(k, v)

        dgcvSR["version_specific_defaults"] = requested_vsd

    if requested_vsd is not None and needs_sympy_hook(requested_vsd):
        attach_sympy_hook(dgcv_class)
    else:
        detach_sympy_hook(dgcv_class)

    if theme is not None:
        _apply_keyval("theme", theme)

    if use_latex is not None:
        _apply_keyval("use_latex", use_latex)

    if ask_before_overwriting_objects_in_vmf is not None:
        _apply_keyval(
            "ask_before_overwriting_objects_in_vmf",
            ask_before_overwriting_objects_in_vmf,
        )

    if forgo_warnings is not None:
        _apply_keyval("forgo_warnings", forgo_warnings)

    if default_engine is not None:
        engine = str(default_engine).lower()
        if engine in ("sage", "sagemath"):
            _apply_keyval("default_symbolic_engine", "sage")
        elif engine in ("sympy",):
            _apply_keyval("default_symbolic_engine", "sympy")
        else:
            _apply_keyval("default_symbolic_engine", engine)

    if format_displays is not None:
        _apply_keyval("format_displays", format_displays)

    if print_style is not None:
        if print_style in {"readable", "literal"}:
            _apply_keyval("print_style", print_style)

    if force_rich_display is not None:
        _apply_keyval("force_rich_display", force_rich_display)

        if force_rich_display:
            try:
                if not is_ipython_available():
                    warnings.warn(
                        "force_rich_display=True was requested, but IPython does not "
                        "appear to be available. Some outputs may render as raw "
                        "HTML or unformatted objects.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception:
                warnings.warn(
                    "force_rich_display=True was requested, but display environment "
                    "could not be verified. Some outputs may render as raw HTML "
                    "or unformatted objects.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if preferred_variable_format is not None:
        if preferred_variable_format in {"real", "complex"}:
            _apply_keyval("preferred_variable_format", preferred_variable_format)

    if use_numeric_methods is not None:
        _apply_keyval("use_numeric_methods", use_numeric_methods)

    if extra_support_for_math_in_tables is not None:
        _apply_keyval(
            "extra_support_for_math_in_tables",
            extra_support_for_math_in_tables,
        )

    if verbose_label_printing is not None:
        _apply_keyval("verbose_label_printing", verbose_label_printing)

    if pass_solve_requests_to_symbolic_engine is not None:
        _apply_keyval(
            "pass_solve_requests_to_symbolic_engine",
            pass_solve_requests_to_symbolic_engine,
        )

    if DEBUG is not None:
        _apply_keyval("DEBUG", DEBUG)


def view_dgcv_settings():
    settings = get_dgcv_settings_registry()
    if not settings:
        print("dgcv settings registry is empty.")
        return

    hidden = {
        "VLP",
        "numeric_error_thresholds",
        "default_numeric_engine",
        "DEBUG",
    }
    items = [(k, settings[k]) for k in settings.keys() if k not in hidden]
    if settings.get("use_numeric_methods", False):
        items.append(("default_numeric_engine", settings.get("default_numeric_engine")))
        items.append(
            (
                "numeric_error_thresholds.abs_tolerance",
                settings.get("numeric_error_thresholds", {}).get("abs_tolerance"),
            )
        )
        items.append(
            (
                "numeric_error_thresholds.rel_tolerance",
                settings.get("numeric_error_thresholds", {}).get("rel_tolerance"),
            )
        )
        items.append(
            (
                "numeric_error_thresholds.policy",
                settings.get("numeric_error_thresholds", {}).get("policy"),
            )
        )
    if not items:
        print("dgcv settings registry is empty.")
        return

    items = sorted(items, key=lambda x: x[0])

    width = max(len(k) for k, _ in items)

    print("\ndgcv settings")
    print("-" * (width + 20))
    for k, v in items:
        print(f"{k:<{width}} : {v!r}")
    print("-" * (width + 20))


def reset_dgcv_settings():
    """
    Reset dgcv settings to their default values.
    """
    from ._config import _vscodepatch

    dgcvSR = get_dgcv_settings_registry()

    dgcvSR.clear()
    dgcvSR.update(
        {
            "use_latex": True,
            "theme": "graph_paper",
            "format_displays": True,
            "version_specific_defaults": f"v{__version__}",
            "ask_before_overwriting_objects_in_vmf": True,
            "forgo_warnings": False,
            "default_symbolic_engine": "sympy",
            "verbose_label_printing": False,
            "VLP": vlp,
            "conjugation_prefix": "BAR",
            "preferred_variable_format": "complex",
            "pass_solve_requests_to_symbolic_engine": True,
            "extra_support_for_math_in_tables": _vscodepatch,
            "use_numeric_methods": False,
            "default_numeric_engine": "numpy",
            "numeric_error_thresholds": {
                "abs_tol": 1e-9,
                "rel_tol": 1e-9,
                "policy": "balanced",  # balanced, reckless,
            },
            "DEBUG": False,
            "print_style": "readable",
            "force_rich_display": False,
        }
    )

    invalidate_engine_cache()
    invalidate_notebook_cache()
