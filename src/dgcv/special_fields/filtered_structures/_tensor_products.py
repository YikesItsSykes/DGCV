from __future__ import annotations

from ..._aux._backends._symbolic_router import subs
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey


class _fast_tensor_products:
    def __init__(self, coeff_dict, alg=None, _validated=None, _atomic_index=-1):
        if isinstance(coeff_dict, _fast_tensor_products):
            coeff_dict, alg, _validated = (
                coeff_dict.coeff_dict,
                coeff_dict.algebra,
                coeff_dict.degree,
            )
        self._atomic_index = _atomic_index
        self.algebra = alg
        if _validated is None:
            if get_dgcv_category(coeff_dict) == "tensorProduct":
                if alg is None:
                    self.algebra = coeff_dict.vector_space
                self.coeff_dict = dict()
                self.degree = 0
                for k, v in coeff_dict.coeff_dict.items():
                    newkey = tuple(factor[0] for factor in k)
                    self.coeff_dict[newkey] = v
                    self.degree = max(self.degree, len(newkey))
            elif isinstance(coeff_dict, dict):
                self.coeff_dict = dict()
                self.degree = 0
                for k, v in coeff_dict.items():
                    if v != 0 or k == tuple():
                        self.coeff_dict[k] = v
                        self.degree = max(self.degree, len(k))
            elif get_dgcv_category(coeff_dict) in {
                "algebra_element",
                "subalgebra_element",
            }:
                if alg is None:
                    self.algebra = coeff_dict.algebra
                self.degree = 1
                self.coeff_dict = {
                    (k,): v for k, v in enumerate(coeff_dict.coeffs) if v != 0
                }
            else:
                self.coeff_dict = dict()
        else:
            self.degree = _validated
            self.coeff_dict = coeff_dict
        if len(self.coeff_dict) == 0:
            self.coeff_dict = {tuple(): 0}
            self.degree = 0
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "fastTensorProduct"
        self._is_zero = None
        self._coeffs = None
        if self.degree < max(len(k) for k in self.coeff_dict):
            raise TypeError("ftp init fail")

    @property
    def is_zero(self):
        if self._is_zero is None:
            self._is_zero = (
                False if any(v != 0 for v in self.coeff_dict.values()) else True
            )
        return self._is_zero

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = list(self.coeff_dict.values())
        return self._coeffs

    def _to_algebra(self, alg=None):
        if alg is None:
            alg = self.algebra
        ae = 0
        basis = getattr(alg, "basis", [])
        dim = len(basis)
        for k, v in self.coeff_dict.items():
            if len(k) != 1:
                return False
            idx = k[0]
            if idx < 0 or idx >= dim:
                return False
            ae += v * basis[idx]

        return ae

    def _convert_to_tp(
        self,
        _hom_id_map=None,
        _hom_id_label=None,
        _hom_id=None,
        _decomp_complete=True,
    ):
        """
        _hom_id_map should be a pair of (str pref, map from neg weights to components)
        """
        from ...core.tensors import tensorProduct

        new_dict = dict()
        card = self.algebra.card
        for k, v in self.coeff_dict.items():
            newkey = tuple(
                (idx, 1 if pos == 0 else 0, card) for pos, idx in enumerate(k)
            )
            new_dict[newkey] = v
        homid = None
        homdecomp = None
        hom_source = _hom_id
        if hom_source:
            decomp, label = hom_source
            if _decomp_complete:
                homdecomp = dict(decomp)
            if _hom_id_map:
                if _hom_id_label:
                    label = _hom_id_label
                pref, ngla = _hom_id_map
                hidd = dict()
                valid = True
                for key, v in decomp.items():
                    jidx, kidx, jdeg, kdeg = key
                    try:
                        jfac = ngla[jdeg][jidx]
                        if kdeg < 0:
                            kfac = ngla[kdeg][kidx]
                        else:
                            kfac = f"{pref}_{kidx + 1}__{{[{kdeg}]}}"
                        hidd[(jfac, kfac)] = v
                    except Exception:
                        valid = False
                        break
                if valid:
                    homid = [hidd, label]
        return tensorProduct(new_dict, _hom_id=homid, _hom_decomp=homdecomp)

    def __add__(self, other):
        if other == 0 or getattr(other, "is_zero", False):
            return self
        if isinstance(other, _fast_tensor_products):
            new_dict = dict(self.coeff_dict)
            deg = self.degree
            for k, v in other.coeff_dict.items():
                deg = max(len(k), deg)
                new_dict[k] = self.coeff_dict.get(k, 0) + v

            return _fast_tensor_products(new_dict, self.algebra, _validated=deg)
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            return self + _fast_tensor_products(other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return (self).__add__(-other)

    def __rsub__(self, other):
        return (-self) + other

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        if not isinstance(terms, (list, tuple)):
            terms = list(terms)
        if not terms:
            return start
        acc = {}
        alg = None
        alg_set = False
        deg = 0
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            alg_set = True
            deg = start.degree
        elif not (isinstance(start, int) and start == 0):
            residual.append(start)
        for t in terms:
            if not isinstance(t, cls):
                residual.append(t)
                continue
            if not alg_set:
                alg = t.algebra
                alg_set = True
            if t.is_zero:
                continue
            for k, v in t.coeff_dict.items():
                deg = max(deg, len(k))
                if v != 0:
                    acc[k] = acc.get(k, 0) + v
        out = cls({k: v for k, v in acc.items() if v != 0}, alg, _validated=deg)
        if residual:
            return sum(residual, out)
        return out

    @classmethod
    def _dgcv_multiadd_scaled(cls, pairs, start=0):
        if not isinstance(pairs, (list, tuple)):
            pairs = list(pairs)
        if not pairs:
            return start
        acc = {}
        alg = None
        alg_set = False
        deg = 0
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            alg_set = True
            deg = start.degree
        elif not (isinstance(start, int) and start == 0):
            residual.append(start)
        for c, t in pairs:
            if not isinstance(t, cls):
                residual.append(c * t)
                continue
            if not alg_set:
                alg = t.algebra
                alg_set = True
            if c == 0 or t.is_zero:
                continue
            for k, v in t.coeff_dict.items():
                deg = max(deg, len(k))
                nv = c * v
                if nv != 0:
                    acc[k] = acc.get(k, 0) + nv
        out = cls({k: v for k, v in acc.items() if v != 0}, alg, _validated=deg)
        if residual:
            return sum(residual, out)
        return out

    def __mul__(self, other):
        if isinstance(other, expr_numeric_types()):
            if other == 0:
                return _fast_tensor_products({tuple(): 0}, self.algebra, _validated=0)
            return _fast_tensor_products(
                {k: other * v for k, v in self.coeff_dict.items()},
                self.algebra,
                _validated=self.degree,
            )
        if isinstance(other, _fast_tensor_products):
            if other.degree == 0:
                return sum(v * self for v in other.coeff_dict.values())
            if self.degree == 1:
                algebraized = self._to_algebra()
                if algebraized is not False:
                    return other * (-algebraized)
            if self.degree == 0:
                return sum(v * other for v in self.coeff_dict.values())
            if other.degree == 1:
                algebraized = self._to_algebra()
                if algebraized is not False:
                    return self * algebraized
            new_dict = dict()
            deg = 0
            ae1 = 0
            ae2 = 0
            alg_basis1, alg_basis2 = (
                getattr(self.algebra, "basis", []),
                getattr(other.algebra, "basis", []),
            )
            alg_dim1, alg_dim2 = len(alg_basis1), len(alg_basis2)
            for k1, v1 in self.coeff_dict.items():
                if len(k1) == 1:
                    idx1 = k1[0]
                    if 0 <= idx1 < alg_dim1:
                        ae1 += v1 * alg_basis1[idx1]
                        continue
                    dgcv_warning(
                        "fast_tensor_products non-aliased mul is being tried for aliased elements",
                        wc_label="debug_log",
                    )
                    continue  ###!!! silent igonore: No multiplication can be defined if idx1>=alg_dim1
                k1L, k1A, k1B, k1T = k1[0], k1[:-1], k1[1:], k1[-1]
                for k2, v2 in other.coeff_dict.items():
                    if len(k2) == 1:
                        idx2 = k2[0]
                        if 0 <= idx2 < alg_dim2:
                            ae2 += v2 * alg_basis2[idx2]
                            continue
                        dgcv_warning(
                            "fast_tensor_products non-aliased mul is being tried for aliased elements",
                            wc_label="debug_log",
                        )
                        continue  ###!!! silent igonore: No multiplication can be defined if idx2>=alg_dim2
                    k2L, k2A, k2B, k2T = k2[0], k2[:-1], k2[1:], k2[-1]
                    if k1T == k2L:
                        newkey = k1B + k2A
                        newval = new_dict.get(newkey, 0) + v1 * v2
                        if newval != 0:
                            deg = max(len(newkey), deg)
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
                    if k1L == k2T:
                        newkey = k2B + k1A
                        newval = new_dict.get(newkey, 0) - v1 * v2
                        if newval != 0:
                            deg = max(len(newkey), deg)
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
            return (
                _fast_tensor_products(new_dict, self.algebra, _validated=deg)
                + ae1 * other
                + self * ae2
            )
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            if self.degree == 0:
                return sum(v * other for v in self.coeff_dict.values())
            if self.degree == 1:
                algebraized = self._to_algebra()
                if algebraized is not False:
                    return algebraized * other
                else:
                    dgcv_warning(
                        "fast_tensor_products non-aliased mul is being tried for aliased elements",
                        wc_label="debug_log",
                    )
                    return self * _fast_tensor_products(other)
            new_dict = dict()
            ac = other.coeffs
            ae1 = 0
            alg_basis1 = getattr(self.algebra, "basis", [])
            alg_dim1 = len(alg_basis1)
            for k1, v1 in self.coeff_dict.items():
                if len(k1) == 1:
                    idx1 = k1[0]
                    if 0 <= idx1 < alg_dim1:
                        ae1 += v1 * alg_basis1[idx1]
                        continue
                    dgcv_warning(
                        "fast_tensor_products non-aliased mul is being tried for aliased elements",
                        wc_label="debug_log",
                    )
                    continue
                k1A, k1T = k1[:-1], k1[-1]
                newval = new_dict.get(k1A, 0) + ac[k1T] * v1
                if newval != 0:
                    new_dict[k1A] = newval
                else:
                    new_dict.pop(k1A, None)
            new_tensor = _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree - 1
            )
            if self.degree == 2:
                algebraized = new_tensor._to_algebra()
                if algebraized is not False:
                    return algebraized + ae1 * other
            return (
                _fast_tensor_products(
                    new_dict, self.algebra, _validated=self.degree - 1
                )
                + ae1 * other
            )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            if other == 0:
                return _fast_tensor_products(dict(), self.algebra, _validated=0)
            return _fast_tensor_products(
                {k: other * v for k, v in self.coeff_dict.items()},
                self.algebra,
                _validated=self.degree,
            )
        if self.degree == 0:
            return sum(v * other for v in self.coeff_dict.values())
        return self * (-other)

    def __neg__(self):
        return _fast_tensor_products(
            {k: -v for k, v in self.coeff_dict.items()},
            self.algebra,
            _validated=self.degree,
        )

    def __matmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return self * other
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            ac = other.coeffs
            new_dict = dict()
            for k, v in self.coeff_dict.items():
                for idx, c in enumerate(ac):
                    if c != 0:
                        newkey = k + (idx,)
                        newval = new_dict.get(newkey, 0) + c * v
                        if newval != 0:
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
            return _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree + 1
            )
        if isinstance(other, _fast_tensor_products):
            ac = other.coeff_dict
            new_dict = dict()
            for k, v in self.coeff_dict.items():
                for idx, c in other.coeff_dict.items():
                    newkey = k + idx
                    newval = new_dict.get(newkey, 0) + c * v
                    if newval != 0:
                        new_dict[newkey] = newval
                    else:
                        new_dict.pop(newkey, None)
            return _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree + other.degree
            )
        return NotImplemented

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def subs(self, subs_data):
        new_dict = {k: subs(v, subs_data) for k, v in self.coeff_dict.items()}
        return _fast_tensor_products(new_dict, self.algebra, _validated=self.degree)
