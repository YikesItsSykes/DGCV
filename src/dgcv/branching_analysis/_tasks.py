from __future__ import annotations

import functools
import inspect

from .._aux._backends._symbolic_router import (
    IndeterminateSignError,
    _scalar_is_zero,
    _scalar_sign,
    exact_nonzero,
    subs,
)
from .._aux._utilities._config import dgcv_warning

_UNSET = object()

_COMPARISON_ALIASES = {
    "relational": "relational",
    "relation": "relational",
    "sign": "relational",
    "signs": "relational",
    "zeroness": "zeroness",
    "zero": "zeroness",
    "vanishing": "zeroness",
}


def positional_capacity(f):
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return 1
    count = 0
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            return 2
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            count += 1
            if count >= 2:
                return 2
    return count


def is_builder(obj):
    if isinstance(obj, functools.partial):
        return True
    return inspect.isfunction(obj) or inspect.ismethod(obj)


def inherited(node, name):
    while node is not None:
        value = node.__dict__.get(name, None)
        if value is not None:
            return value
        node = node.__dict__.get("_parent", None)
    return None


def sample_point(node):
    chain = []
    while node is not None:
        chain.append(node)
        node = node.__dict__.get("_parent", None)
    merged = {}
    for anc in reversed(chain):
        own = anc.__dict__.get("_sample_point", None)
        if own:
            merged.update(own)
    return merged


def inherited_constraints(node):
    chain = []
    while node is not None:
        chain.append(node)
        node = node.__dict__.get("_parent", None)
    merged = {}
    for anc in reversed(chain):
        own = anc.__dict__.get("_inherited_constraints", None)
        if own:
            merged.update(own)
    return merged


def constraint_rules(node):
    hidden = inherited_constraints(node)
    own = node.closed_case_rules
    if not hidden:
        return dict(own)
    resolved = {e: subs(v, own) for e, v in hidden.items() if e not in own}
    return {**resolved, **own}


def substitutions(node):
    point = sample_point(node)
    rules = constraint_rules(node)
    if not point:
        return rules
    resolved = {e: subs(v, point) for e, v in rules.items() if e not in point}
    return {**resolved, **point}


def comparison_mode(node):
    raw = inherited(node, "_comparison")
    return resolve_comparison(raw)


def resolve_comparison(raw):
    if raw is None:
        return "relational"
    mode = _COMPARISON_ALIASES.get(str(raw).lower())
    if mode is None:
        raise ValueError(
            "`comparison` should be None, 'relational', or 'zeroness'."
        )
    return mode


def _zeroness(value):
    if _scalar_is_zero(value):
        return "="
    try:
        if exact_nonzero(value):
            return "!="
    except Exception:
        pass
    return None


def classify_sample(value, comparison="relational"):
    mode = resolve_comparison(comparison)
    if mode == "relational":
        try:
            sign = _scalar_sign(value)
        except IndeterminateSignError:
            return _zeroness(value)
        except Exception:
            return _zeroness(value)
        return {-1: "<", 0: "=", 1: ">"}[sign]
    return _zeroness(value)


def sampled_conditions(node):
    own = node.__dict__.get("_sampled_conditions", None)
    return list(own) if own else []


def sampled_relations(node):
    if node._sampled_relations is _UNSET:
        rules = substitutions(node)
        mode = comparison_mode(node)
        out = []
        for expr in sampled_conditions(node):
            relation = classify_sample(subs(expr, rules), mode)
            if relation is None:
                dgcv_warning(
                    f"Cannot certify the sign or vanishing of `{expr}` at the sample "
                    f"point; it may depend on unsampled free symbols.",
                    wc_label="dgcvOperationsNote",
                )
            out.append((expr, relation))
        node._sampled_relations = tuple(out)
    return node._sampled_relations


def symbol_pool(node, subject):
    out = []
    es = node.general_equation_system
    if hasattr(es, "free_symbols") or hasattr(es, "variables"):
        out.append(es)
    else:
        try:
            out.extend(list(es))
        except TypeError:
            out.append(es)
    if subject is not None and subject is not es:
        out.append(subject)
    return out


def has_equation_system(node):
    es = node.general_equation_system
    try:
        return len(es) > 0
    except TypeError:
        return es is not None


def clear_profile_cache(node):
    node._profile_result = _UNSET
    node._profile_error = None
    node._status = _UNSET
    node._specialized = _UNSET
    node._sampled_relations = _UNSET
    for sc in node._subcases.values():
        clear_profile_cache(sc)


def invoke_with(node, f, first):
    capacity = positional_capacity(f)
    if capacity <= 0:
        return f()
    if capacity == 1:
        return f(first)
    return f(first, node)


def invoke(node, f):
    return invoke_with(node, f, node.specialized)


def status_text(value):
    if isinstance(value, bool):
        return "complete" if value else "incomplete"
    return None if value is None else str(value)


def compute_status(node):
    r = node.result
    if node._profile_error is not None:
        return "error"
    rule = node.status_rule
    if rule is not None:
        try:
            return status_text(invoke(node, rule))
        except Exception as e:
            if inherited(node, "_raise_on_profile_error") is True:
                raise
            node._profile_error = e
            return "error"
    if isinstance(r, dict) and "status" in r:
        return status_text(r["status"])
    if node.profile is not None:
        return None
    if has_equation_system(node) or node._explicit_completion:
        return "complete" if node.complete else "incomplete"
    return None


def completion_message(node):
    if hasattr(node, "completion_message"):
        f = node.completion_message
        if callable(f):
            return str(f(node))
        else:
            return str(f)
    else:
        return f"number of free variables = {len(getattr(node, 'free_variables', {}))}"


def result_message(node, sep=", "):
    if hasattr(node, "completion_message"):
        f = node.completion_message
        return str(f(node)) if callable(f) else str(f)
    r = node.result
    if isinstance(r, dict) and "message" in r:
        return str(r["message"])
    fields = node.result_fields
    if fields:
        return sep.join(f"{k} = {v}" for k, v in fields.items())
    return None


def leaf_annotation(node, evaluate=True, sep=", "):
    if not evaluate and node._status is _UNSET:
        return ""
    st = node.status
    if node.profile is None:
        if st == "complete":
            message = completion_message(node)
            addon = " - " + message if message else ""
            return "complete" + addon
        return st if st else ""
    msg = result_message(node, sep)
    if st is None:
        return msg if msg else ""
    joiner = sep if "\n" in sep else " - "
    return st + (joiner + msg if msg else "")
