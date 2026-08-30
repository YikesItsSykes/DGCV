from __future__ import annotations

from .._aux._backends._symbolic_router import subs
from .._aux._utilities._config import dgcv_warning


def normalize_conditions(raw):
    if not isinstance(raw, dict):
        return {"closed": {}, "open": {}}
    if "closed" in raw or "open" in raw:
        return {
            "closed": dict(raw.get("closed", {})),
            "open": dict(raw.get("open", {})),
        }
    return {"closed": dict(raw), "open": {}}


def merge_conditions(store, conditions, source, warn_open=True):
    # source codes: "i" inherited, "d" defining, "c" corollary
    for e, value in conditions.get("closed", {}).items():
        if "closed" in store.get(e, {}):
            dgcv_warning(
                f"`{e}` already carries the closed condition `{e} = "
                f"{store[e]['closed'][0]}`, so it is no longer present in the reduced "
                f"system. Retaining that condition and ignoring `{e} = {value}`.",
                wc_label="dgcvOperationsNote",
            )
            continue
        store.setdefault(e, {})["closed"] = (value, source)
    for e, values in conditions.get("open", {}).items():
        if warn_open and "closed" in store.get(e, {}):
            dgcv_warning(
                f"`{e}` already carries the closed condition `{e} = "
                f"{store[e]['closed'][0]}`, so it is no longer present in the reduced "
                f"system. Keeping its open condition as book-keeping only.",
                wc_label="dgcvOperationsNote",
            )
        if not isinstance(values, (set, frozenset, list, tuple)):
            values = {values}
        slot = store.setdefault(e, {})
        merged = dict(slot.get("open", ()))
        for value in values:
            merged[value] = source
        slot["open"] = tuple(merged.items())
    return store


def backsub_store(store, sub_dict):
    out = {}
    for e, slot in store.items():
        ns = {}
        if "closed" in slot:
            ns["closed"] = (subs(slot["closed"][0], sub_dict), "i")
        if "open" in slot:
            ns["open"] = tuple((subs(v, sub_dict), "i") for v, _ in slot["open"])
        out[e] = ns
    return out
