from __future__ import annotations

from ...._aux._backends._polynomials import poly_gens
from ...._aux._utilities._config import get_variable_registry
from ...._aux._vmf.vmf import vmf_lookup


def _vmf_atoms_for_view(v, *, view: str):
    data = vmf_lookup(
        v,
        path=True,
        relatives=True,
        flattened_relatives=False,
        system_index=True,
    )
    rel = data.get("relatives") or {}

    st = rel.get("standard") or ()
    if st:
        return tuple(st)

    keys = ("holo", "anti") if view == "complex" else ("real", "imag")
    out = []
    for k in keys:
        w = rel.get(k)
        if w is None:
            continue
        if isinstance(w, tuple):
            out.extend(w)
        else:
            out.append(w)
    return tuple(out)


def _term_from_monom(gens, m, c):
    term = c
    for g, e in zip(gens, m):
        if e:
            term = term * (g ** int(e))
    return term


def _build_holo_anti_index_sets(P):
    gens = tuple(poly_gens(P))

    vr = get_variable_registry()
    conv = vr.get("conversion_dictionaries", {})

    holo_keys = set(conv.get("holToReal", {}).keys())  # strings like "z1"
    anti_keys = set(conv.get("symToHol", {}).keys())  # strings like "BARz1"

    holo_idx = [i for i, g in enumerate(gens) if str(g) in holo_keys]
    anti_idx = [i for i, g in enumerate(gens) if str(g) in anti_keys]

    return gens, holo_idx, anti_idx
