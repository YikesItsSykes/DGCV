from __future__ import annotations

from ..._aux.printing.printing._dgcv_display import LaTeX, LaTeX_eqn_system


def render_value(value, latex):
    if value is None or isinstance(value, (bool, int, str)):
        return str(value)
    if latex:
        try:
            return f"${LaTeX(value)}$"
        except Exception:
            return str(value)
    return str(value)


def conditions_str(bucket, plain_text=False, **kwargs):
    punct = kwargs.pop("_punct", None)
    if not isinstance(punct, str):
        punct = ","
    closed = bucket.get("closed", {})
    open_rules = bucket.get("open", {})
    if plain_text:
        parts = [f"{e} = {v}" for e, v in closed.items()]
        parts += [f"{e} \u2260 {v}" for e, vals in open_rules.items() for v in vals]
        return ", ".join(parts)
    open_list = [(e, v) for e, vals in open_rules.items() for v in vals]
    if len(open_list) == 0:
        out = LaTeX_eqn_system(
            closed, **{"one_line": True, "punctuation": punct, **kwargs}
        )
    else:
        out = LaTeX_eqn_system(
            closed,
            **{
                "one_line": True,
                "punctuation": punct,
                "conjuction": " ",
                "force_oxford_comma": True,
                **kwargs,
            },
        )[:-2]
    if len(open_list) == 1:
        if len(closed) > 0:
            if len(closed) == 1 and out[-1:] == ",":
                out = out[:-1]
            out += (
                r"\quad\text{and}\quad "
                + (
                    LaTeX_eqn_system(
                        open_list,
                        **{
                            "one_line": True,
                            "punctuation": punct,
                            "relation": r"\neq",
                            **kwargs,
                        },
                    )[2:]
                )
            )
        else:
            out = LaTeX_eqn_system(
                open_list,
                **{
                    "one_line": True,
                    "punctuation": punct,
                    "relation": r"\neq",
                    **kwargs,
                },
            )
    elif len(open_list) > 1:
        if len(closed) > 0:
            out += LaTeX_eqn_system(
                open_list,
                **{
                    "one_line": True,
                    "punctuation": punct,
                    "relation": r"\neq",
                    "force_oxford_comma": True,
                    **kwargs,
                },
            )[2:]
        else:
            out = LaTeX_eqn_system(
                open_list,
                **{
                    "one_line": True,
                    "punctuation": punct,
                    "relation": r"\neq",
                    **kwargs,
                },
            )

    return out

_RELATION_LATEX = {"<": r" < ", "=": r" = ", ">": r" > ", "!=": r" \neq ", None: r" \,?\, "}
_RELATION_PLAIN = {"<": "<", "=": "=", ">": ">", "!=": "≠", None: "?"}


def represents_strings(node, latex, filter_conditions=None):
    flt = filter_conditions if callable(filter_conditions) else (lambda x: x)
    out = []
    for expr, relation in node.sampled_relations:
        key = flt(expr)
        if latex:
            out.append(
                LaTeX_eqn_system(
                    {key: 0}, relation=_RELATION_LATEX[relation], math_mode="$"
                )
            )
        else:
            out.append(f"{key} {_RELATION_PLAIN[relation]} 0")
    return out


def condition_strings(node, sources, latex, filter_conditions=None, key_filter=None):
    flt = filter_conditions if callable(filter_conditions) else (lambda x: x)
    keep = key_filter if callable(key_filter) else (lambda e: True)
    out = []
    for e, slot in node.case_rules.items():
        if not keep(e):
            continue
        if "closed" in slot:
            v, src = slot["closed"]
            if src in sources:
                ke, ve = flt(e), flt(v)
                out.append(
                    LaTeX_eqn_system({ke: ve}, math_mode="$")
                    if latex
                    else f"{ke} = {ve}"
                )
        for v, src in slot.get("open", ()):
            if src in sources:
                ke, ve = flt(e), flt(v)
                out.append(
                    LaTeX_eqn_system({ke: ve}, relation=r" \neq ", math_mode="$")
                    if latex
                    else f"{ke} \u2260 {ve}"
                )
    return out

