from ..._aux._backends._symbolic_router import subs
from ..._aux._backends._types_and_constants import is_atomic


def _sage_solve_to_dicts(sols, vars_, input_symbols=frozenset()):
    if sols is None:
        return []

    if isinstance(sols, dict):
        sols = [sols]
    elif not isinstance(sols, (list, tuple)):
        sols = [sols]

    vars_set = set(vars_)
    out = []

    for s in sols:
        if isinstance(s, dict):
            out.append({k: v for k, v in s.items() if k in vars_set})
            continue

        rels = list(s) if isinstance(s, (list, tuple)) else [s]
        d = {}
        replacements = {}
        ok = True

        for rel in rels:
            lhs = getattr(rel, "lhs", None)
            rhs = getattr(rel, "rhs", None)

            if callable(lhs) and callable(rhs):
                try:
                    L = rel.lhs()
                    R = rel.rhs()
                except Exception:
                    ok = False
                    break
            elif (
                lhs is not None
                and rhs is not None
                and not callable(lhs)
                and not callable(rhs)
            ):
                L = lhs
                R = rhs
            else:
                ok = False
                break

            if L in vars_set:
                d[L] = R
                if (
                    is_atomic(R)
                    and R not in vars_set
                    and R not in input_symbols
                ):
                    replacements[R] = L
        if replacements:
            d = {k: subs(v, replacements) for k, v in d.items()}

        if ok:
            out.append(d)

    return out


def _linsolve_to_dicts(solset, vars_):
    if not solset:
        return []
    out = []
    for tup in solset:
        if isinstance(tup, dict):
            out.append(tup)
            continue
        try:
            tup = tuple(tup) if hasattr(tup, "__iter__") else ()
        except Exception:
            tup = ()
        if len(tup) == len(vars_):
            out.append(dict(zip(vars_, tup)))
    return out


def _rel_lhs_rhs(rel):
    f = getattr(rel, "lhs", None)
    g = getattr(rel, "rhs", None)
    if callable(f) and callable(g):
        try:
            return rel.lhs(), rel.rhs()
        except Exception:
            return None, None
    if f is not None and g is not None and not callable(f) and not callable(g):
        return f, g
    return None, None


def _engine_solve_to_dicts(sols, vars_):
    if sols is None:
        return []
    if isinstance(sols, dict):
        return [sols]
    if not isinstance(sols, (list, tuple)):
        sols = [sols]

    out = []
    vars_set = set(vars_)

    for s in sols:
        if isinstance(s, dict):
            out.append(s)
            continue

        rels = list(s) if isinstance(s, (list, tuple)) else [s]

        d = {}
        ok = True
        for rel in rels:
            lhs, rhs = _rel_lhs_rhs(rel)
            if lhs is None:
                ok = False
                break
            if lhs in vars_set:
                d[lhs] = rhs

        if ok:
            out.append(d)

    return out
