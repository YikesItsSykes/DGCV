from __future__ import annotations

from ....._aux._backends._symbolic_router import _scalar_is_zero, simplify, subs
from ....._aux._backends._types_and_constants import verify_conjugates_free
from ....._aux._utilities._config import dgcv_warning
from ....._aux._vmf.vmf import vmf_lookup
from ....conversions.conversions import allToHol, allToReal, allToSym
from .workers import _expand_special_to_general, _is_scalar_coeff_dict


class _tensor_field_reduction:
    def _eval_simplify(self, **kwargs):
        rule = self._simplifyKW.get("simplify_rule", None)
        ign = self._simplifyKW.get("simplify_ignore_list", None)

        if rule is None:
            simplified = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}
        elif rule == "holomorphic":
            simplified = {
                k: simplify(allToHol(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "real":
            simplified = {
                k: simplify(allToReal(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "symbolic_conjugate":
            simplified = {
                k: simplify(allToSym(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        else:
            dgcv_warning(f"Unsupported simplify_rule: {rule}.")
            simplified = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}

        return self.__class__(
            coeff_dict=simplified,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def __dgcv_simplify__(self, method=None, **kwargs):
        return self._eval_simplify(**kwargs)

    @property
    def __dgcv_zero_obstr__(self):
        cp = self.coef_profile
        cfs = self.coeff_free_symbols
        if "mixed" in cp:
            if "real" in cp:

                def check(x):
                    st = vmf_lookup(x)["sub_type"]
                    return st == "standard" or st == "real" or st == "imag"

                exprs = (allToReal(expr) for expr in self.coeff_dict.values())
                cfs = {x for x in cfs if check(x)}
            else:

                def check(x):
                    st = vmf_lookup(x)["sub_type"]
                    return st == "standard" or st == "holo" or st == "anti"

                def guarded_conv(x):
                    if verify_conjugates_free(x):
                        return x
                    return allToSym(x)

                exprs = (guarded_conv(expr) for expr in self.coeff_dict.values())
                cfs = {x for x in cfs if check(x)}
        else:
            exprs = self.coeff_dict.values()

        return exprs, cfs

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if (
            self.coordinates != other.coordinates
            or self.valence != other.valence
            or self.dgcvType != other.dgcvType
        ):
            return False

        if self.data_shape != other.data_shape:
            a = _expand_special_to_general(self.coeff_dict, self.data_shape)
            b = _expand_special_to_general(other.coeff_dict, other.data_shape)
            keys = set(a.keys()).union(b.keys())
            for k in keys:
                va, vb = a.get(k, 0), b.get(k, 0)
                if va is vb or va == vb:
                    continue
                if simplify(allToReal(va)) != simplify(allToReal(vb)):
                    return False
            return True

        cd, ocd = self.coeff_dict, other.coeff_dict
        keys = set(cd.keys()).union(ocd.keys())
        for k in keys:
            va, vb = cd.get(k, 0), ocd.get(k, 0)
            if va is vb or va == vb:
                continue
            if simplify(allToReal(va)) != simplify(allToReal(vb)):
                return False
        return True

    def __hash__(self):
        if self._hash is not None:
            return self._hash
        g = _expand_special_to_general(self.coeff_dict, self.data_shape)
        items = tuple(sorted((k, simplify(allToReal(v))) for k, v in g.items()))
        self._hash = hash(
            (self.coordinates, self.valence, self.data_shape, self.dgcvType, items)
        )
        return self._hash

    @property
    def is_zero(self) -> bool:
        cd = getattr(self, "coeff_dict", None)
        if not isinstance(cd, dict) or not cd:
            return True

        if _is_scalar_coeff_dict(cd):
            return cd.get(tuple(), 0) == 0

        for v in cd.values():
            if not _scalar_is_zero(v):
                return False
        return True

    def subs(self, substitutions):
        substituted = {k: subs(v, substitutions) for k, v in self.coeff_dict.items()}
        return self.__class__(
            coeff_dict=substituted,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )
