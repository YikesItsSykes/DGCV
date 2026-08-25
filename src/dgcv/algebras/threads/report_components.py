from __future__ import annotations

from ..._aux._backends._symbolic_router import _scalar_is_zero
from .heating import _ideal_iso_label


def _alg_name_plain(alg) -> str:
    try:
        return alg.__str__(VLP=False)
    except Exception:
        return str(getattr(alg, "label", None) or "Unnamed Algebra")


def _alg_name_latex(alg) -> tuple[str, str]:
    try:
        s = alg._repr_latex_(abbrev=True, raw=True)
        s = str(s).replace("$", "").replace(r"\displaystyle", "").strip()
        if not s:
            raise RuntimeError
        return f"${s}$", f"${s}$"
    except Exception:
        nm = _alg_name_plain(alg)
        cap = nm if nm[:1].isupper() else (nm[:1].upper() + nm[1:])
        return nm, cap


def _fmt_bool_cache(v):
    return "true" if v is True else ("false" if v is False else "not yet evaluated")


_singularity_sources = (
    ("radical", "radical"),
    ("LD", "Levi decomposition"),
    ("derived_series", "derived series"),
    ("simple_ideals", "simple subalgebras"),
    ("center", "center"),
    ("subalgebra_ranks", "subalgebra ranks"),
    ("structure", "structure coefficients"),
)


def _is_trivial_level(level) -> bool:
    if not level:
        return True
    if isinstance(level, (list, tuple)) and len(level) == 1:
        return _scalar_is_zero(level[0])
    return False


def _level_dim(elems) -> int:
    if _is_trivial_level(elems):
        return 0
    try:
        return len(elems)
    except Exception:
        return 0


def _basic_items_plain(refAlg, *, subAlg: bool, algebra_name: str) -> list[str]:
    items = []
    params = list(getattr(refAlg, "_parameters", []) or [])
    if subAlg:
        items.append(
            f"Subalgebra family contained in {algebra_name}"
            if params
            else f"Subalgebra contained in {algebra_name}"
        )
    items.append(_dimension_item(refAlg))
    if params:
        items.append(f"Parameters: {', '.join(repr(p) for p in params)}")

    lie = getattr(refAlg, "_lie_algebra_cache", None)
    if lie is True:
        items.append("Lie algebra: true")
        st = getattr(refAlg, "_educed_properties", dict()).get("special_type", None)
        if st is None:
            for attr, name in (
                ("_is_simple_cache", "simple"),
                ("_is_semisimple_cache", "semisimple"),
                ("_is_abelian_cache", "abelian"),
                ("_is_nilpotent_cache", "nilpotent"),
                ("_is_solvable_cache", "solvable"),
            ):
                if getattr(refAlg, attr, None) is True:
                    st = name
                    break
        if st is not None:
            items.append(f"special properties: {st}")
        elif (
            getattr(refAlg, "_is_semisimple_cache", None) is False
            and getattr(refAlg, "_is_solvable_cache", None) is False
        ):
            items.append("special properties: neither solvable nor semisimple")
        else:
            items.append("special properties: not yet evaluated")
    elif lie is False:
        items.append("Lie algebra: false")
        items.append(
            f"Skew symmetric: {_fmt_bool_cache(getattr(refAlg, '_skew_symmetric_cache', None))}"
        )
        items.append(
            f"Jacobi identity satisfied: {_fmt_bool_cache(getattr(refAlg, '_jacobi_identity_cache', None))}"
        )
    else:
        items.append("Lie algebra: not yet evaluated")

    return items


def _radical_kind(rad) -> str:
    if getattr(rad, "_is_abelian_cache", None) is True:
        return "abelian"
    series = getattr(rad, "_derived_series_cache", None)
    if isinstance(series, (list, tuple)) and series and series[0] is not None:
        levels = list(series[0])
        if len(levels) < 2 or _is_trivial_level(getattr(levels[1], "basis", levels[1])):
            return "abelian"
    if getattr(rad, "_is_nilpotent_cache", None) is True:
        return "nilpotent"
    series = getattr(rad, "_lower_central_series_cache", None)
    if isinstance(series, (list, tuple)) and series and series[0] is not None:
        levels = list(series[0])
        if levels and _is_trivial_level(getattr(levels[-1], "basis", levels[-1])):
            return "nilpotent"
    return "solvable"


def _levi_terse(refAlg) -> str:
    """
    One-line plain-text form of a Levi decomposition.

    Parameters
    ----------
    refAlg : algebra_class or subalgebra_class

    Returns
    -------
    str
        For example `'su(2)+sl(2,R)'`, with a trailing semidirect factor such
        as `'[1-d. abelian]'` when the radical is nontrivial. A leading
        `'complexified: '` marks a decomposition that only holds after
        complexifying.

    Notes
    -----
    Reads caches only, reporting `'not yet evaluated'` rather than computing.
    """
    if getattr(refAlg, "dimension", 0) == 0:
        return "0-d. trivial"
    if getattr(refAlg, "_lie_algebra_cache", None) is False:
        return "not a Lie algebra"
    ld = getattr(refAlg, "_Levi_deco_cache", None)
    if getattr(refAlg, "_lie_algebra_cache", None) is not True or not isinstance(
        ld, dict
    ):
        return "not yet evaluated"
    comps = ld.get("LD_components", None)
    if not isinstance(comps, (list, tuple)) or len(comps) < 2:
        return "not yet evaluated"

    Levi_component, rad = comps[0], comps[1]
    simples = ld.get("simple_ideals", None)
    semi_dim = getattr(Levi_component, "dimension", 0) if Levi_component else 0
    rad_dim = getattr(rad, "dimension", 0) if rad else 0

    complexified = False
    head = ""
    if semi_dim:
        if simples:
            parts = []
            for a in simples:
                text, scope = _ideal_iso_label(
                    a, use_latex=False, refAlg=refAlg, return_scope=True
                )
                if scope == "complexification":
                    complexified = True
                parts.append(
                    text
                    if text is not None
                    else f"[{getattr(a, 'dimension', '?')}-d. simple]"
                )
            head = "\u2295".join(parts)
        else:
            head = f"[{semi_dim}-d. semisimple]"

    tail = f"[{rad_dim}-d. {_radical_kind(rad)}]" if rad_dim else ""

    if head and tail:
        out = f"{head}\u22c9{tail}"
    else:
        out = head or tail or "0-d. trivial"
    return f"complexified: {out}" if complexified else out


def _dimension_item(refAlg, use_latex=False):
    field = "R" if getattr(refAlg, "base_field", "complex") == "real" else "C"
    if use_latex:
        return rf"Dimension over $\mathbb{{{field}}}$: {refAlg.dimension}"
    return f"Dimension over {field}: {refAlg.dimension}"


def _ellide(str_list, *, max_items: int):
    str_list = list(str_list or [])
    if len(str_list) <= max_items:
        return str_list
    k = max_items // 2
    return str_list[:k] + ["..."] + str_list[-k:]


def _fmt_angle_list(xs, *, max_items: int = 12) -> str:
    toks = [str(x) for x in _ellide(xs, max_items=max_items)]
    return "<" + ", ".join(toks) + ">"


def _fmt_grading_plain(grading, *, max_items: int = 12) -> str:
    if not isinstance(grading, (list, tuple)) or not grading:
        return "None"
    out = []
    for g in grading:
        if not isinstance(g, (list, tuple)):
            out.append(str(g))
            continue
        toks = [str(x) for x in _ellide(list(g), max_items=max_items)]
        out.append("(" + ", ".join(toks) + ")")
    return "[" + ", ".join(out) + "]"
