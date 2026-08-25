from __future__ import annotations

from ....._aux._backends._exact_arith import exact_reciprocal
from ....._aux._backends._symbolic_router import _scalar_is_zero
from ....._aux._backends._types_and_constants import check_dgcv_scalar
from ....._aux._vmf._safeguards import (
    check_dgcv_category,
    get_dgcv_category,
    query_dgcv_categories,
)
from .workers import _process_coeffs_dict_new


class _tensor_field_algebra:
    def swap_tensor_valence(self):
        def key_change(key):
            deg = len(key) // 3
            new_k = tuple(
                j if c < deg or 2 * deg <= c else 1 - j for c, j in enumerate(key)
            )
            return new_k

        cd = {key_change(key): value for key, value in self.coeff_dict.items()}
        if query_dgcv_categories(self, "differential_form"):
            from ..vector_fields import vector_field_class

            return vector_field_class(
                coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
            )
        if query_dgcv_categories(self, "vector_field"):
            from ..differential_forms import differential_form_class

            return differential_form_class(
                coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
            )
        from . import tensor_field_class

        return tensor_field_class(
            coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
        )

    def _with_same_meta(self, *, coeff_dict, data_shape=None, variable_spaces=None):
        return self.__class__(
            coeff_dict=coeff_dict,
            data_shape=self.data_shape if data_shape is None else data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces
            if variable_spaces is None
            else variable_spaces,
        )

    def _coerce_to_general(self):
        if self.data_shape == "general":
            return self
        if self.data_shape == "all":
            return self
        g = self.expanded_coeff_dict
        out = self._with_same_meta(coeff_dict=g, data_shape="general")
        out._shape_checked = True
        return out

    def _maybe_promote_general_to(self, target_shape: str):
        if self._shape_checked:
            return
        self._shape_checked = True
        if self.data_shape != "general":
            return
        if target_shape not in ("symmetric", "skew"):
            return
        if not self.valence or len(set(self.valence)) != 1:
            return
        new_cd, eff_shape = _process_coeffs_dict_new(self.coeff_dict, target_shape)
        if eff_shape == target_shape:
            self.coeff_dict = new_cd
            self.data_shape = eff_shape
            self._expanded_coeff_dict = None
            self._coeffArray = None
            self._cd_formats = None
            self._hash = None
            self._minimal_coordinate_space = None

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self._is_scalar() or other._is_scalar():
            raise TypeError("Cannot add tensors of different degrees.")

        return self._add_tensor(other, coerce_shapes=True)

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def __neg__(self):
        if self._is_scalar():
            return self.__class__(
                coeff_dict={tuple(): -self._scalar_value()},
                data_shape="all",
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
            )
        return self._with_same_meta(
            coeff_dict={k: -v for k, v in self.coeff_dict.items()},
            variable_spaces=self._variable_spaces,
        )

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self + (-other)

    def __matmul__(self, other):
        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()
        if not isinstance(tf, self.__class__):
            return NotImplemented
        return self._shape_product(tf, kind="general")

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            if self._is_scalar():
                return self.__class__(
                    coeff_dict={tuple(): other * self._scalar_value()},
                    data_shape="all",
                    dgcvType=self.dgcvType,
                    _simplifyKW=self._simplifyKW,
                    variable_spaces=self._variable_spaces,
                )
            return self._with_same_meta(
                coeff_dict={k: other * v for k, v in self.coeff_dict.items()},
                variable_spaces=self._variable_spaces,
            )

        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()
        if not isinstance(tf, self.__class__):
            return NotImplemented

        return self._shape_product(tf, kind="general")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if check_dgcv_scalar(scalar):
            return self * exact_reciprocal(scalar)
        return NotImplemented

    def _add_tensor(self, other, *, coerce_shapes: bool, _return_raw: bool = False):
        a = self
        b = other

        a_shape = a.data_shape
        b_shape = b.data_shape

        if coerce_shapes:
            if a_shape == "general" and b_shape in ("symmetric", "skew"):
                a._maybe_promote_general_to(b_shape)
                a_shape = a.data_shape
            if b_shape == "general" and a_shape in ("symmetric", "skew"):
                b._maybe_promote_general_to(a_shape)
                b_shape = b.data_shape

        if a_shape == "all":
            out_shape = b_shape
        elif b_shape == "all":
            out_shape = a_shape
        elif a_shape == b_shape:
            out_shape = a_shape
        else:
            out_shape = "general"

        aa = a if a_shape == out_shape or a_shape == "all" else a._coerce_to_general()
        bb = b if b_shape == out_shape or b_shape == "all" else b._coerce_to_general()

        new_cd = {}
        for k, v in aa.coeff_dict.items():
            if not _scalar_is_zero(v):
                new_cd[k] = v
        for k, v in bb.coeff_dict.items():
            if not _scalar_is_zero(v):
                new_cd[k] = new_cd.get(k, 0) + v

        new_cd, eff_shape = _process_coeffs_dict_new(new_cd, out_shape)
        merged_vs = a._merged_variable_spaces(b)

        if _return_raw:
            return new_cd, eff_shape, merged_vs

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=eff_shape,
            dgcvType=a.dgcvType,
            _simplifyKW=a._simplifyKW,
            variable_spaces=merged_vs,
        )

    def _tp_concat_cd_fast(self, other, shape=None):
        a = self
        b = other
        out = {}

        def _parity_sign(order):
            n = len(order)
            sign = 1
            seen = [False] * n
            for i in range(n):
                if seen[i]:
                    continue
                j = i
                cycle_len = 0
                while not seen[j]:
                    seen[j] = True
                    j = order[j]
                    cycle_len += 1
                if cycle_len and (cycle_len % 2 == 0):
                    sign = -sign
            return sign

        for ka, va in a.coeff_dict.items():
            if _scalar_is_zero(va):
                continue
            if ka:
                da = len(ka) // 3
                ia = ka[:da]
                va_bits = ka[da : 2 * da]
                sa = ka[2 * da :]
            else:
                ia = va_bits = sa = tuple()

            for kb, vb in b.coeff_dict.items():
                if _scalar_is_zero(vb):
                    continue
                if kb:
                    db = len(kb) // 3
                    ib = kb[:db]
                    vb_bits = kb[db : 2 * db]
                    sb = kb[2 * db :]
                else:
                    ib = vb_bits = sb = tuple()

                inds = ia + ib
                bits = va_bits + vb_bits
                sys = sa + sb
                n = len(inds)

                if n == 0:
                    nk = tuple()
                    out[nk] = out.get(nk, 0) + va * vb
                    continue

                if shape in ("skew", "symmetric"):
                    if shape == "skew":
                        order = sorted(
                            range(n), key=lambda k: (str(inds[k]), bits[k], sys[k])
                        )
                        sign = _parity_sign(order)
                    else:
                        order = sorted(
                            range(n), key=lambda k: (inds[k], bits[k], sys[k])
                        )
                        sign = 1

                    inds2 = tuple(inds[k] for k in order)
                    bits2 = tuple(bits[k] for k in order)
                    sys2 = tuple(sys[k] for k in order)

                    if shape == "skew":
                        seen = set()
                        for t in zip(inds2, bits2, sys2):
                            if t in seen:
                                sign = 0
                                break
                            seen.add(t)
                        if sign == 0:
                            continue

                    nk = inds2 + bits2 + sys2
                    out[nk] = out.get(nk, 0) + sign * va * vb
                    continue

                nk = inds + bits + sys
                out[nk] = out.get(nk, 0) + va * vb

        return out

    def _shape_product(self, other, *, kind: str):
        a = self
        b = other

        if kind == "general":
            aa = a if a.data_shape in ("general", "all") else a._coerce_to_general()
            bb = b if b.data_shape in ("general", "all") else b._coerce_to_general()
            cd = aa._tp_concat_cd_fast(bb)
            cd, eff_shape = _process_coeffs_dict_new(cd, "general")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        if kind == "skew":
            if a.data_shape in ("skew", "all") and b.data_shape in ("skew", "all"):
                cd = a._tp_concat_cd_fast(b, shape="skew")
                cd, eff_shape = _process_coeffs_dict_new(cd, "skew")
                return self.__class__(
                    coeff_dict=cd,
                    data_shape=eff_shape,
                    dgcvType=a.dgcvType,
                    _simplifyKW=a._simplifyKW,
                    variable_spaces=a._merged_variable_spaces(b),
                )
            ab = a._shape_product(b, kind="general")
            ba = b._shape_product(a, kind="general")
            cd = dict(ab.coeff_dict)
            for k, v in ba.coeff_dict.items():
                if not _scalar_is_zero(v):
                    cd[k] = cd.get(k, 0) - v
            cd, eff_shape = _process_coeffs_dict_new(cd, "skew")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        if kind == "symmetric":
            if a.data_shape in ("symmetric", "all") and b.data_shape in (
                "symmetric",
                "all",
            ):
                cd = a._tp_concat_cd_fast(b)
                cd, eff_shape = _process_coeffs_dict_new(cd, "symmetric")
                return self.__class__(
                    coeff_dict=cd,
                    data_shape=eff_shape,
                    dgcvType=a.dgcvType,
                    _simplifyKW=a._simplifyKW,
                    variable_spaces=a._merged_variable_spaces(b),
                )

            ab = a._shape_product(b, kind="general")
            ba = b._shape_product(a, kind="general")
            cd = dict(ab.coeff_dict)
            for k, v in ba.coeff_dict.items():
                if not _scalar_is_zero(v):
                    cd[k] = cd.get(k, 0) + v
            cd, eff_shape = _process_coeffs_dict_new(cd, "symmetric")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        raise ValueError(f"Unknown product kind '{kind}'.")

    def tp(self, *others):
        return self.tensor_product(*others)

    def tensor_product(self, *others):
        out = self
        for o in others:
            if check_dgcv_category(o):
                coerce = getattr(o, "as_tensor_field", None)
                if callable(coerce):
                    o = coerce()
            if not isinstance(o, self.__class__):
                return NotImplemented
            out = out._shape_product(o, kind="general")
        return out

    def skew_product(self, *others):
        out = self
        df = query_dgcv_categories(out, {"differential_form"})  # bool
        for o in others:
            if check_dgcv_scalar(o):
                out = o * out
                continue
            if not get_dgcv_category(o) == "tensor_field":
                return NotImplemented

            if not (df and query_dgcv_categories(o, {"differential_form"})):
                if check_dgcv_category(out):
                    coerce = getattr(out, "as_tensor_field", None)
                    if callable(coerce):
                        out = coerce()
                if check_dgcv_category(o):
                    coerce = getattr(o, "as_tensor_field", None)
                    if callable(coerce):
                        o = coerce()
            out = out._shape_product(o, kind="skew")
        return out

    def wedge(self, *others):
        # alias
        return self.skew_product(*others)

    def symmetric_product(self, *others):
        out = self
        for o in others:
            if not get_dgcv_category(o) == "tensor_field":
                return NotImplemented
            coerce = getattr(o, "as_tensor_field", None)
            if callable(coerce):
                o = coerce()
            coerce = getattr(self, "as_tensor_field", None)
            s = coerce() if callable(coerce) else self
            out = s._shape_product(o, kind="symmetric")
        return out
