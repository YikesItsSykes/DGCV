from __future__ import annotations

from ..._aux._backends._numeric_router import zeroish
from ..._aux._backends._symbolic_router import _scalar_is_zero, simplify
from ..._aux._utilities._config import get_dgcv_settings_registry
from ..._aux._vmf._safeguards import get_dgcv_category
from ..dgcv_core import differential_form_class, wedge


def _key_universe(tfs):
    K = set()
    for tf in tfs:
        if get_dgcv_category(tf) == "tensor_field":
            for k, v in tf.coeff_dict.items():
                if not _scalar_is_zero(v):
                    K.add(k)
    return tuple(sorted(K))


def _as_coeff_vector_form(tf, K, syslbl="__dgcv_par__"):
    cd = {}
    varspacPlaceholder = None
    for j, k in enumerate(K):
        v = tf.coeff_dict.get(k, 0)
        if not _scalar_is_zero(v):
            cd[(j, 0, syslbl)] = v
    if not cd:
        cd = {tuple(): 0}
    return differential_form_class(
        coeff_dict=cd, data_shape="skew", variable_spaces={syslbl: varspacPlaceholder}
    )


def _extract_basis_by_wedge_vectorized(
    objs,
    *,
    use_numeric_methods: bool = False,
    dimension_hint=None,
    array_shape_checked: bool = False,
):
    if not objs:
        return []
    if dimension_hint is None:
        dimension_hint = len(objs)

    use_numeric = use_numeric_methods or bool(
        get_dgcv_settings_registry().get("use_numeric_methods", False)
    )

    def _is_zero_scalar(x):
        if use_numeric:
            return zeroish(x)
        return _scalar_is_zero(x)

    if all(get_dgcv_category(o) == "array" for o in objs):
        if array_shape_checked is False:
            shaped = dict()
            for obj in objs:
                shp = obj.shape
                shaped[shp] = shaped.get(shp, []) + [obj]
            if len(shaped) > 1:
                return sum(
                    [
                        _extract_basis_by_wedge_vectorized(
                            val,
                            use_numeric_methods=use_numeric_methods,
                            array_shape_checked=True,
                        )
                        for val in shaped.values()
                    ],
                    [],
                )

        K = set()
        for o in objs:
            K.update(getattr(o, "_data", {}).keys())
        if not K:
            return []

        K = tuple(sorted(K))

        vecs = []
        for o in objs:
            d = getattr(o, "_data", {})
            v = {}
            for k in K:
                val = d.get(k, 0)
                if val is None or _is_zero_scalar(val):
                    continue
                v[k] = val
            vecs.append(v)

        def _kform_is_zero(F):
            return not F or all(_is_zero_scalar(c) for c in F.values())

        def _wedge_kform_vector(F, v):
            out = {}
            if not F or not v:
                return out
            for idx, c in F.items():
                I_set = set(idx)
                for j, vj in v.items():
                    if j in I_set:
                        continue
                    pos = 0
                    for t in idx:
                        if t < j:
                            pos += 1
                        else:
                            break
                    sign = -1 if ((len(idx) - pos) % 2) else 1
                    J = idx[:pos] + (j,) + idx[pos:]
                    coeff = sign * c * vj
                    if _is_zero_scalar(coeff):
                        continue
                    out[J] = out.get(J, 0) + coeff
            if not use_numeric:
                out = {k: simplify(v) for k, v in out.items() if not _is_zero_scalar(v)}
            else:
                out = {k: v for k, v in out.items() if not _is_zero_scalar(v)}
            return out

        obstruction = None
        out = []
        for o, v in zip(objs, vecs):
            if not v:
                continue

            if obstruction is None:
                obstruction = {(k,): val for k, val in v.items()}
                if not use_numeric:
                    obstruction = {
                        idx: simplify(c)
                        for idx, c in obstruction.items()
                        if not _is_zero_scalar(c)
                    }
                out.append(o)
                continue

            w = _wedge_kform_vector(obstruction, v)
            if _kform_is_zero(w):
                continue

            obstruction = w
            out.append(o)

        return out

    K = _key_universe(objs)
    if not K:
        return []

    try:
        vec_forms = [_as_coeff_vector_form(o, K) for o in objs]
    except Exception as exc:
        raise TypeError(
            "Could not compute a linear independence test for these objects because "
            "they do not support the required linear combination behavior."
        ) from exc

    obstruction = None
    out = []
    for o, v in zip(objs, vec_forms):
        if len(out) == dimension_hint:
            break
        if use_numeric:
            if zeroish(v):
                continue
        else:
            if _scalar_is_zero(v):
                continue

        if obstruction is None:
            obstruction = v
            out.append(o)
            continue

        w = wedge(obstruction, v)
        if not use_numeric:
            w = simplify(w)
            if _scalar_is_zero(w):
                continue
        else:
            if zeroish(w):
                continue

        obstruction = w
        out.append(o)

    return out
