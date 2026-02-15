"""
package: dgcv - Differential Geometry with Complex Variables
module: matrix_ring

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
import numbers
from collections import deque

from .._config import get_dgcv_settings_registry
from .._dgcv_display import LaTeX
from .._safeguards import check_dgcv_category, get_dgcv_category, retrieve_passkey
from ..backends._types_and_constants import expr_numeric_types, is_atomic
from ..base import dgcv_class
from ..printing._string_processing import _format_label_with_hi_low, latex_superscript

__all__ = ["matrix_atom", "symbolic_matrix"]


# -----------------------------------------------------------------------------
# matrix ring classes
# -----------------------------------------------------------------------------
class matrix_atom(dgcv_class):
    def __init__(self, label: str, properties: dict | None = None):
        if properties is None:
            properties = dict()
        self.label = label
        self.transposed = properties.get("transposed", False) is True
        self.conjugated = properties.get("conjugated", False) is True

        shp = properties.get("shape", None)
        self.shape = shp

        if shp == (1, 1):
            properties = dict(properties)
            properties.setdefault("central", True)
            properties.setdefault("symmetric", True)

        self._properties = properties

        self._registered = properties.get("_registered", False)
        self._tex_label = properties.get("_tex_label", label)
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "symbolic_matrix"
        self._dgcv_categories = {"atom"}

    def __eq__(self, other):
        if not isinstance(other, matrix_atom):
            return NotImplemented
        return (
            self.label == other.label
            and self.transposed == other.transposed
            and self.conjugated == other.conjugated
        )

    def __hash__(self):
        return hash((self.label, self.transposed, self.conjugated))

    @property
    def is_zero(self) -> bool:
        return self == symb_zero_matrix

    @property
    def is_one(self) -> bool:
        return self == symb_id_matrix

    def is_central(self):
        if self._properties.get("central", False):
            return True
        shp = self.shape
        return shp == (1, 1)

    def copy(self, new_properties: dict | None = None):
        props = dict(self._properties)
        if new_properties:
            props.update(new_properties)
        return matrix_atom(self.label, props)

    def __str__(self):
        label = self.label
        if self.conjugated:
            label = f"{label}.conjugate()"
        if self.transposed:
            label = f"{label}.transpose()"
        return label

    def __repr__(self):
        return f"matrix_atom({str(self)!r}, shape={self.shape!r})"

    def _repr_latex_(self, verbose: bool = False, raw: bool = False, **kwargs):
        label = self.label
        stripped, conj_label = conj_prefex_check(label)
        base = conj_label if stripped else label
        base = _format_label_with_hi_low(base)
        if stripped != self.conjugated:
            base = (
                latex_superscript(base, "*")
                if self.transposed
                else f"\\overline{{{base}}}"
            )
        elif self.transposed is True:
            base = latex_superscript(base, "T")
        return base if raw else rf"$\displaystyle {base}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def __dgcv_simplify__(self, *args, **kwargs):
        return self

    def subs(self, subs_dict: dict):
        d = subs_dict if isinstance(subs_dict, dict) else dict(subs_dict)

        v = d.get(self, None)
        if v is not None:
            return v

        for attr in ("transpose", "conjugate", "conjugate_transpose"):
            f = getattr(self, attr, None)
            if not callable(f):
                continue
            k = f()
            sub_val = d.get(k, None)
            if sub_val is None:
                continue

            if check_dgcv_category(sub_val) is not None:
                g = getattr(sub_val, attr, None)
                return g() if callable(g) else sub_val
            return sub_val

        return self

    def conjugate(self):
        if self._properties.get("real", False):
            return self
        properties = dict(self._properties)
        properties["conjugated"] = not self.conjugated
        return matrix_atom(self.label, properties)

    def transpose(self):
        properties = dict(self._properties)
        shape = properties.get("shape", None)
        if shape:
            properties["shape"] = (shape[1], shape[0])
        if self._properties.get("symmetric", False):
            return matrix_atom(self.label, properties)
        if self._properties.get("skew", False):
            return -matrix_atom(self.label, properties)
        properties["transposed"] = not self.transposed
        return matrix_atom(self.label, properties)

    def conjugate_transpose(self):
        return self.conjugate().transpose()

    def __neg__(self):
        return symbolic_matrix.neg(self)

    def __pow__(self, power):
        nil_degree = self._properties.get("nilpotent_degree", None)
        modulus = self._properties.get("unipotent_degree", None)
        idem_degree = self._properties.get("eventual_idempotency", None)

        if power == 0:
            return symb_id_matrix
        if power == 1:
            return self

        is_int = isinstance(power, numbers.Integral)
        is_numeric = isinstance(power, expr_numeric_types()) and not isinstance(
            power, bool
        )

        if is_int:
            if isinstance(idem_degree, numbers.Integral) and power > idem_degree:
                power = idem_degree

            if isinstance(nil_degree, numbers.Integral) and nil_degree <= power:
                return symb_zero_matrix

            if isinstance(modulus, expr_numeric_types()):
                try:
                    power = power % modulus
                    if power == 0:
                        return symb_id_matrix
                except Exception:
                    pass

            return symbolic_matrix(("pow", self, power)).__dgcv_simplify__(deep=False)

        if is_numeric:
            return symbolic_matrix(("pow", self, power)).__dgcv_simplify__(deep=False)

        return NotImplemented

    def __add__(self, other):
        return symbolic_matrix.add(self, other)

    def __radd__(self, other):
        return symbolic_matrix.add(other, self)

    def __sub__(self, other):
        return symbolic_matrix.add(self, symbolic_matrix.neg(other))

    def __rsub__(self, other):
        return symbolic_matrix.add(other, symbolic_matrix.neg(self))

    def __mul__(self, other):
        if self.shape == (1, 1):
            if get_dgcv_category(other) != "symbolic_matrix" and not isinstance(
                other, expr_numeric_types()
            ):
                rm = getattr(other, "__rmul__", None)
                if callable(rm):
                    res = rm(self.copy(new_properties={"shape": None, "central": True}))
                    if res is not NotImplemented:
                        return res
        return symbolic_matrix.mul(self, other)

    def __rmul__(self, other):
        if get_dgcv_category(other) != "symbolic_matrix" and not isinstance(
            other, expr_numeric_types()
        ):
            return NotImplemented
        return symbolic_matrix.mul(other, self)


def conj_prefex_check(word: str, pattern: str | None = None) -> tuple:
    if pattern is None:
        pattern = get_dgcv_settings_registry().get("conjugation_prefix", None)
    pref = len(pattern) if pattern else 0
    clip, tail = word[:pref], word[pref:]
    if pattern and clip == pattern:
        return (True, tail)
    return (False, (pattern + word) if pattern else word)


symb_id_matrix = matrix_atom(
    "Id",
    properties={
        "symmetric": True,
        "real": True,
        "unipotent_degree": 1,
        "central": True,
        "square": True,
    },
)
symb_zero_matrix = matrix_atom(
    "0",
    properties={
        "symmetric": True,
        "real": True,
        "nilpotent_degree": 1,
        "central": True,
    },
)


def symb_zero_mat_n_by_m(n: int, m: int):
    properties = {"real": True, "central": True}
    if n == m:
        properties["symmetric"] = True
        properties["nilpotent_degree"] = 1
    return matrix_atom("0", properties=properties)


class symbolic_matrix(dgcv_class):
    """
    Noncommutative formal expressions in matrix atoms.

    Canonical form used by __dgcv_simplify__:
      sum_{word} coeff[word] * word
    where word is a tuple of (matrix_atom, positive_int_power) pairs.
    The empty word () represents the identity.
    """

    def __init__(self, AST_tuple: tuple, _inherited_singularities: set = set()):
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "symbolic_matrix"
        self._dgcv_categories = {"symbolic_matrix"}
        self._ast = AST_tuple
        self._shape = None
        self._inherited_singularities = _inherited_singularities
        self._singularities = None
        self._is_zero = None
        self._is_one = None

    @property
    def ast(self):
        return self._ast

    @property
    def op(self):
        return self._ast[0]

    @property
    def args(self):
        return self._ast[1:]

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._infer_shape()
        return self._shape

    @property
    def singularities(self):
        if self._singularities is None:
            computed_singularities = set()  # add logic to populate this.
            self._singularities = computed_singularities | self._inherited_singularities
        return self._singularities

    @staticmethod
    def _is_numeric(x):
        return isinstance(x, expr_numeric_types()) and not isinstance(x, bool)

    @staticmethod
    def _as_expr(x):
        if isinstance(x, symbolic_matrix):
            return x
        if isinstance(x, matrix_atom):
            return x
        if x == symb_id_matrix or x == symb_zero_matrix:
            return x
        if symbolic_matrix._is_numeric(x):
            return symbolic_matrix(("scalar", x))
        return x

    @property
    def is_zero(self) -> bool:
        v = getattr(self, "_is_zero", None)
        if v is not None:
            return v

        z = self._compute_is_zero_one()[0]
        self._is_zero = z
        return z

    @property
    def is_one(self) -> bool:
        v = getattr(self, "_is_one", None)
        if v is not None:
            return v

        o = self._compute_is_zero_one()[1]
        self._is_one = o
        return o

    def _compute_is_zero_one(self) -> tuple[bool, bool]:
        x = self

        if x == symb_zero_matrix:
            return (True, False)
        if x == symb_id_matrix:
            return (False, True)

        op = x.op

        if op == "scalar":
            c = x.args[0]
            if c == 0:
                return (True, False)
            if c == 1:
                return (False, True)
            return (False, False)

        if op == "mul":
            try:
                coeff, word = x._canonical_mul(deep=False)
            except Exception:
                return (False, False)

            if coeff == 0:
                return (True, False)
            if not word:
                return (False, coeff == 1)
            return (False, False)

        if op == "add":
            try:
                coeffs = x._canonical_add_from_terms(list(x.args))
            except Exception:
                return (False, False)

            if not coeffs:
                return (True, False)

            if len(coeffs) == 1:
                ((word, c),) = coeffs.items()
                if c == 0:
                    return (True, False)
                if not word:
                    return (False, c == 1)
            return (False, False)

        if op == "pow":
            base, exp = x.args

            if exp == 0:
                return (False, True)

            if base == symb_zero_matrix:
                return (True, False)

            if base == symb_id_matrix:
                return (False, True)

            return (False, False)

        if op == "conj" or op == "tran":
            inner = x.args[0]
            if inner == symb_zero_matrix:
                return (True, False)
            if inner == symb_id_matrix:
                return (False, True)
            if isinstance(inner, matrix_atom):
                return (inner.is_zero, inner.is_one)
            if isinstance(inner, symbolic_matrix):
                return (inner.is_zero, inner.is_one)
            return (False, False)

        return (False, False)

    @staticmethod
    def _is_zero(x):
        return (
            x == 0
            or x == symb_zero_matrix
            or (isinstance(x, symbolic_matrix) and x.op == "scalar" and x.args[0] == 0)
        )

    @staticmethod
    def _is_one(x):
        return (
            x == 1
            or x == symb_id_matrix
            or (isinstance(x, symbolic_matrix) and x.op == "scalar" and x.args[0] == 1)
        )

    @staticmethod
    def _scalar_value(x):
        if isinstance(x, symbolic_matrix) and x.op == "scalar":
            return x.args[0]
        if symbolic_matrix._is_numeric(x):
            return x
        return None

    @staticmethod
    def scalar(c):
        if c == 0:
            return symb_zero_matrix
        if c == 1:
            return symb_id_matrix
        return symbolic_matrix(("scalar", c))

    @staticmethod
    def neg(x):
        x = symbolic_matrix._as_expr(x)
        if symbolic_matrix._is_zero(x):
            return symb_zero_matrix
        return symbolic_matrix.mul(-1, x)

    @staticmethod
    def add(*terms):
        tlist = [symbolic_matrix._as_expr(t) for t in terms]

        target_shape = None
        for t in tlist:
            shp = getattr(t, "shape", None)
            if shp is not None and shp != (1, 1):
                target_shape = shp
                break

        if target_shape is not None:
            new_list = []
            for t in tlist:
                shp = getattr(t, "shape", None)
                if shp == (1, 1):
                    if isinstance(t, matrix_atom):
                        t0 = t.copy({"shape": None})
                    elif isinstance(t, symbolic_matrix):
                        t0 = t.copy({"shape": None})
                    else:
                        t0 = t
                    new_list.append(
                        symbolic_matrix.mul(
                            t0, symb_id_matrix.copy({"shape": target_shape})
                        )
                    )
                else:
                    new_list.append(t)
            tlist = new_list

        out = symbolic_matrix(("add", *tlist))
        return out.__dgcv_simplify__()

    @staticmethod
    def mul(*factors):
        out = symbolic_matrix(
            ("mul", 1, *[symbolic_matrix._as_expr(f) for f in factors])
        )
        return out.__dgcv_simplify__()

    @staticmethod
    def pow(base, exponent):
        base = symbolic_matrix._as_expr(base)

        if exponent == 0:
            return symb_id_matrix
        if exponent == 1:
            return base
        if base == symb_zero_matrix:
            return symb_zero_matrix
        if base == symb_id_matrix:
            return symb_id_matrix

        out = symbolic_matrix(("pow", base, exponent))
        return out.__dgcv_simplify__()

    def copy(self, new_properties=None):
        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_category = self._dgcv_category
        out._dgcv_categories = set(self._dgcv_categories)
        out._ast = self._ast
        out._shape = self._shape
        if new_properties:
            for k, v in new_properties.items():
                if k == "shape":
                    out._shape = v
                else:
                    setattr(out, k, v)
        return out

    def subs(self, subs_dict: dict):
        d = subs_dict if isinstance(subs_dict, dict) else dict(subs_dict)

        eng_subs = {}
        for k, v in d.items():
            if is_atomic(k):
                eng_subs[k] = v

        def sub_atom(a):
            if check_dgcv_category(a) is not None:
                f = getattr(a, "subs", None)
                return f(d) if callable(f) else a
            f = getattr(a, "subs", None)
            return f(eng_subs) if callable(f) else a

        def rec(x):
            if isinstance(x, matrix_atom):
                return x.subs(d)

            if isinstance(x, symbolic_matrix):
                op = x.op
                if op == "scalar":
                    c = x.args[0]
                    f = getattr(c, "subs", None)
                    c2 = f(eng_subs) if callable(f) else c
                    return symbolic_matrix.scalar(c2)

                if op == "add":
                    return symbolic_matrix.add(*[rec(t) for t in x.args])

                if op == "mul":
                    coeff = x.args[0]
                    f = getattr(coeff, "subs", None)
                    coeff2 = f(eng_subs) if callable(f) else coeff
                    return symbolic_matrix.mul(coeff2, *[rec(t) for t in x.args[1:]])

                if op == "pow":
                    base, exp = x.args
                    exp2 = exp
                    f = getattr(exp, "subs", None)
                    exp2 = f(eng_subs) if callable(f) else exp
                    return symbolic_matrix.pow(rec(base), exp2)

                if op == "conj":
                    y = rec(x.args[0])
                    g = getattr(y, "conjugate", None)
                    return (
                        g()
                        if callable(g)
                        else symbolic_matrix(("conj", y)).__dgcv_simplify__()
                    )

                if op == "tran":
                    y = rec(x.args[0])
                    g = getattr(y, "transpose", None)
                    return (
                        g()
                        if callable(g)
                        else symbolic_matrix(("tran", y)).__dgcv_simplify__()
                    )

                return symbolic_matrix(
                    tuple([op] + [rec(a) for a in x.args])
                ).__dgcv_simplify__()

            if check_dgcv_category(x) is not None:
                f = getattr(x, "subs", None)
                return f(d) if callable(f) else x

            if is_atomic(x):
                f = getattr(x, "subs", None)
                return f(eng_subs) if callable(f) else x

            f = getattr(x, "subs", None)
            return f(eng_subs) if callable(f) else x

        out = rec(self)
        return out.__dgcv_simplify__() if isinstance(out, symbolic_matrix) else out

    def conjugate(self):
        return symbolic_matrix(("conj", self)).__dgcv_simplify__()

    def transpose(self):
        return symbolic_matrix(("tran", self)).__dgcv_simplify__()

    def conjugate_transpose(self):
        return self.conjugate().transpose()

    def __add__(self, other):
        return symbolic_matrix.add(self, other)

    def __radd__(self, other):
        return symbolic_matrix.add(other, self)

    def __sub__(self, other):
        return symbolic_matrix.add(self, symbolic_matrix.neg(other))

    def __rsub__(self, other):
        return symbolic_matrix.add(other, symbolic_matrix.neg(self))

    def __mul__(self, other):
        other = symbolic_matrix._as_expr(other)

        if getattr(self, "shape", None) == (1, 1):
            if get_dgcv_category(other) != "symbolic_matrix" and not isinstance(
                other, expr_numeric_types()
            ):
                rm = getattr(other, "__rmul__", None)
                if callable(rm):
                    res = rm(self.copy(new_properties={"shape": None, "central": True}))
                    if res is not NotImplemented:
                        return res

        if get_dgcv_category(other) == "symbolic_matrix":
            sshape = getattr(self, "shape", None)
            oshape = getattr(other, "shape", None)
            if (
                (sshape is not None)
                and (oshape is not None)
                and (sshape != (1, 1))
                and (oshape != (1, 1))
                and (sshape[1] != oshape[0])
            ):
                raise TypeError(
                    f"* is not supported between dgcv's `symbolic_matrix` objects tagged with incompatible shapes. "
                    f"Shapes given: {sshape} and {oshape}"
                )

        return symbolic_matrix.mul(self, other)

    def __rmul__(self, other):
        if get_dgcv_category(other) != "symbolic_matrix" and not isinstance(
            other, expr_numeric_types()
        ):
            return NotImplemented
        return symbolic_matrix.mul(other, self)

    def __neg__(self):
        return symbolic_matrix.neg(self)

    def __pow__(self, power):
        return symbolic_matrix.pow(self, power)

    def __dgcv_simplify__(self, deep: bool = True, **kwargs):
        op = self.op

        if op == "scalar":
            c = self.args[0]
            return symb_zero_matrix if c == 0 else (symb_id_matrix if c == 1 else self)

        if op == "conj":
            x = self.args[0]
            x = x.__dgcv_simplify__() if deep and isinstance(x, symbolic_matrix) else x

            if x == symb_zero_matrix or x == symb_id_matrix:
                return x
            if isinstance(x, matrix_atom):
                return x.conjugate()

            if isinstance(x, symbolic_matrix):
                if x.op == "scalar":
                    c = x.args[0]
                    cc = getattr(c, "conjugate", None)
                    if callable(cc):
                        try:
                            return symbolic_matrix.scalar(cc())
                        except Exception:
                            return symbolic_matrix(("conj", x))
                    return symbolic_matrix(("conj", x))

                if x.op == "add":
                    return symbolic_matrix(
                        ("add", *[symbolic_matrix(("conj", t)) for t in x.args])
                    ).__dgcv_simplify__(deep=deep)

                if x.op == "mul":
                    coeff = x.args[0]
                    rest = x.args[1:]
                    return symbolic_matrix(
                        ("mul", coeff, *[symbolic_matrix(("conj", f)) for f in rest])
                    ).__dgcv_simplify__(deep=deep)

                if x.op == "pow":
                    base, exp = x.args
                    return symbolic_matrix(
                        ("pow", symbolic_matrix(("conj", base)), exp)
                    ).__dgcv_simplify__(deep=deep)

                if x.op == "conj":
                    return x.args[0]

                return symbolic_matrix(("conj", x))

            return symbolic_matrix(("conj", x))

        if op == "tran":
            x = self.args[0]
            x = x.__dgcv_simplify__() if deep and isinstance(x, symbolic_matrix) else x

            if x == symb_zero_matrix or x == symb_id_matrix:
                return x
            if isinstance(x, matrix_atom):
                return x.transpose()

            if isinstance(x, symbolic_matrix):
                if x.op == "scalar":
                    return x

                if x.op == "add":
                    return symbolic_matrix(
                        ("add", *[symbolic_matrix(("tran", t)) for t in x.args])
                    ).__dgcv_simplify__(deep=deep)

                if x.op == "mul":
                    coeff = x.args[0]
                    rest = list(x.args[1:])
                    rest.reverse()
                    return symbolic_matrix(
                        ("mul", coeff, *[symbolic_matrix(("tran", f)) for f in rest])
                    ).__dgcv_simplify__(deep=deep)

                if x.op == "pow":
                    base, exp = x.args
                    return symbolic_matrix(
                        ("pow", symbolic_matrix(("tran", base)), exp)
                    ).__dgcv_simplify__(deep=deep)

                if x.op == "tran":
                    return x.args[0]

                return symbolic_matrix(("tran", x))

            return symbolic_matrix(("tran", x))

        if op == "pow":
            base, exp = self.args
            if not isinstance(exp, numbers.Integral):
                return self
            if exp < 0:
                raise ValueError(
                    "symbolic_matrix only supports nonnegative integer powers"
                )
            if exp == 0:
                return symb_id_matrix
            if exp == 1:
                return (
                    base.__dgcv_simplify__()
                    if deep and isinstance(base, symbolic_matrix)
                    else base
                )

            base_s = (
                base.__dgcv_simplify__()
                if deep and isinstance(base, symbolic_matrix)
                else base
            )
            if base_s == symb_zero_matrix:
                return symb_zero_matrix
            if base_s == symb_id_matrix:
                return symb_id_matrix
            if isinstance(base_s, matrix_atom):
                return symbolic_matrix(("pow", base_s, int(exp)))

            if isinstance(base_s, symbolic_matrix) and base_s.op == "pow":
                inner_base, inner_exp = base_s.args
                if isinstance(inner_exp, numbers.Integral):
                    return symbolic_matrix(
                        ("pow", inner_base, inner_exp * exp)
                    ).__dgcv_simplify__(deep=deep)

            return symbolic_matrix(("pow", base_s, int(exp)))

        if op == "mul":
            coeff = self.args[0]
            factors = list(self.args[1:])

            if symbolic_matrix._scalar_value(coeff) is None:
                if symbolic_matrix._is_numeric(coeff):
                    coeff = coeff
                else:
                    raise TypeError("mul node has non-numeric coefficient")

            if deep:
                new_factors = []
                for f in factors:
                    if isinstance(f, symbolic_matrix):
                        new_factors.append(f.__dgcv_simplify__(deep=True))
                    else:
                        new_factors.append(f)
                factors = new_factors

            for i, f in enumerate(factors):
                if isinstance(f, symbolic_matrix) and f.op == "add":
                    left = factors[:i]
                    right = factors[i + 1 :]

                    terms = []
                    for t in f.args:
                        terms.append(
                            symbolic_matrix(
                                ("mul", coeff, *left, t, *right)
                            ).__dgcv_simplify__(deep=deep)
                        )

                    return symbolic_matrix(("add", *terms)).__dgcv_simplify__(deep=deep)

            coeff2, word = symbolic_matrix(("mul", coeff, *factors))._canonical_mul(
                deep=False
            )

            if coeff2 == 0:
                return symb_zero_matrix
            if not word:
                return symb_id_matrix if coeff2 == 1 else symbolic_matrix.scalar(coeff2)
            if coeff2 == 1 and len(word) == 1 and word[0][1] == 1:
                a = word[0][0]
                if isinstance(a, matrix_atom) and getattr(a, "shape", None) == (1, 1):
                    had_shapeless = any(
                        getattr(f, "shape", None) is None for f in self.args[1:]
                    )
                    had_id = any(f == symb_id_matrix for f in self.args[1:])
                    if had_id and had_shapeless:
                        return a.copy({"shape": None})
                return a

            built = []
            for a, e in word:
                if e == 1:
                    built.append(a)
                else:
                    if symbolic_matrix._can_aggregate_as_power(a):
                        built.append(symbolic_matrix(("pow", a, e)))
                    else:
                        built.extend([a] * int(e))
            out = symbolic_matrix(("mul", coeff2, *built))
            out._shape = out._infer_shape()
            return out

        if op == "add":
            terms = [
                t.__dgcv_simplify__() if deep and isinstance(t, symbolic_matrix) else t
                for t in self.args
            ]
            coeffs = self._canonical_add_from_terms(terms)
            if not coeffs:
                return symb_zero_matrix

            if len(coeffs) == 1:
                ((word, c),) = coeffs.items()
                if c == 0:
                    return symb_zero_matrix
                if not word:
                    return symb_id_matrix if c == 1 else symbolic_matrix.scalar(c)
                factors = []
                for a, e in word:
                    factors.append(a if e == 1 else symbolic_matrix(("pow", a, e)))
                expr = symbolic_matrix(("mul", c, *factors)).__dgcv_simplify__(
                    deep=False
                )
                return expr

            built_terms = []
            for word, c in coeffs.items():
                if c == 0:
                    continue
                if not word:
                    built_terms.append(
                        symb_id_matrix if c == 1 else symbolic_matrix.scalar(c)
                    )
                    continue
                factors = []
                for a, e in word:
                    factors.append(a if e == 1 else symbolic_matrix(("pow", a, e)))
                built_terms.append(
                    symbolic_matrix(("mul", c, *factors)).__dgcv_simplify__(deep=False)
                )

            if not built_terms:
                return symb_zero_matrix
            if len(built_terms) == 1:
                return built_terms[0]

            out = symbolic_matrix(("add", *built_terms))
            out._shape = out._infer_shape()
            return out

        return self

    def _canonical_mul(self, deep: bool = True):
        if self.op != "mul":
            raise TypeError("_canonical_mul only applies to mul nodes")

        coeff = self.args[0]
        factors = list(self.args[1:])

        if symbolic_matrix._scalar_value(coeff) is None:
            if symbolic_matrix._is_numeric(coeff):
                pass
            else:
                raise TypeError("mul node has non-numeric coefficient")

        central_pows = {}
        word = []

        def _is_central_atom(a):
            return isinstance(a, matrix_atom) and a.is_central()

        def _add_central(a, e: int):
            if e == 0:
                return
            central_pows[a] = central_pows.get(a, 0) + e
            if central_pows[a] == 0:
                del central_pows[a]

        def _add_word(a, e: int):
            if e == 0:
                return
            symbolic_matrix._word_append(word, a, e)

        def _push_atom(a, e: int):
            if _is_central_atom(a):
                _add_central(a, e)
            else:
                _add_word(a, e)

        for f in factors:
            if isinstance(f, symbolic_matrix) and deep:
                f = f.__dgcv_simplify__()

            if f == symb_zero_matrix or symbolic_matrix._is_zero(f):
                return (0, ())

            if f == symb_id_matrix or symbolic_matrix._is_one(f):
                continue

            sv = symbolic_matrix._scalar_value(f)
            if sv is not None:
                coeff *= sv
                continue

            if isinstance(f, symbolic_matrix) and f.op in {"conj", "tran"}:
                x = f.args[0]
                if isinstance(x, matrix_atom):
                    f = x.conjugate() if f.op == "conj" else x.transpose()

            if isinstance(f, symbolic_matrix) and f.op == "mul":
                inner_coeff, inner_word = f._canonical_mul(deep=deep)
                coeff *= inner_coeff
                for a, e in inner_word:
                    _push_atom(a, e)
                continue

            if isinstance(f, matrix_atom):
                _push_atom(f, 1)
                continue

            if isinstance(f, symbolic_matrix) and f.op == "pow":
                b, e = f.args
                if (
                    isinstance(e, numbers.Integral)
                    and e >= 0
                    and isinstance(b, matrix_atom)
                    and symbolic_matrix._is_square_matrix_atom(b)
                ):
                    if e != 0:
                        _push_atom(b, int(e))
                    continue

            _add_word(f, 1)

        if central_pows:
            central_word = []
            for a in sorted(
                central_pows,
                key=lambda z: (
                    getattr(z, "label", ""),
                    getattr(z, "conjugated", False),
                    getattr(z, "transposed", False),
                    getattr(z, "shape", None),
                ),
            ):
                e = central_pows[a]
                if e:
                    central_word.append((a, e))
            return (coeff, tuple(central_word + word))

        return (coeff, tuple(word))

    @staticmethod
    def _is_square_matrix_atom(a) -> bool:
        if not isinstance(a, matrix_atom):
            return False
        props = getattr(a, "properties", None)
        if isinstance(props, dict):
            if bool(props.get("square", False)):
                return True
            shp = props.get("shape", None)
            if isinstance(shp, tuple) and len(shp) == 2:
                return shp[0] == shp[1]
        return False

    @staticmethod
    def _can_aggregate_as_power(atom_like) -> bool:
        if isinstance(atom_like, matrix_atom):
            return symbolic_matrix._is_square_matrix_atom(atom_like)
        return True

    @staticmethod
    def _word_append(word_list, atom_like, exp: int):
        if exp == 0:
            return
        if not word_list:
            word_list.append((atom_like, exp))
            return
        last_a, last_e = word_list[-1]
        if last_a == atom_like:
            if symbolic_matrix._can_aggregate_as_power(atom_like):
                word_list[-1] = (last_a, last_e + exp)
            else:
                word_list.append((atom_like, exp))
        else:
            word_list.append((atom_like, exp))

    def _canonical_add_from_terms(self, terms):
        coeffs = {}
        stack = deque(terms)

        while stack:
            t = stack.popleft()
            if symbolic_matrix._is_zero(t) or t == symb_zero_matrix:
                continue

            if isinstance(t, symbolic_matrix) and t.op == "add":
                stack.extend(t.args)
                continue

            c, w = self._term_to_coeff_word(t)
            if c == 0:
                continue

            coeffs[w] = coeffs.get(w, 0) + c
            if coeffs[w] == 0:
                del coeffs[w]

        return coeffs

    def _term_to_coeff_word(self, t):
        if t == symb_id_matrix:
            return (1, ())

        if isinstance(t, matrix_atom):
            return (1, ((t, 1),))

        if isinstance(t, symbolic_matrix) and t.op == "scalar":
            return (t.args[0], ())

        if isinstance(t, symbolic_matrix) and t.op == "mul":
            c, w = t._canonical_mul(deep=False)
            return (c, w)

        if isinstance(t, symbolic_matrix) and t.op == "pow":
            base, exp = t.args
            if isinstance(exp, numbers.Integral) and exp >= 0:
                if exp == 0:
                    return (1, ())
                if isinstance(base, matrix_atom):
                    return (1, ((base, int(exp)),))
            return (1, ((t, 1),))

        sv = symbolic_matrix._scalar_value(t)
        if sv is not None:
            return (sv, ())

        return (1, ((t, 1),))

    def _infer_shape(self):
        op = self.op

        if op == "scalar":
            return None
        if op == "neg":
            x = self.args[0]
            return getattr(x, "shape", None)
        if op == "add":
            shapes = [getattr(t, "shape", None) for t in self.args]
            known = [s for s in shapes if s is not None]
            if not known:
                return None

            non_scalar = [s for s in known if s != (1, 1)]
            if not non_scalar:
                return (1, 1)

            base = non_scalar[0]
            for s in non_scalar[1:]:
                if s != base:
                    raise TypeError(
                        f"+ between tagged shapes is not supported: {base} vs {s}"
                    )
            return base
        if op == "mul":
            factors = self.args[1:]
            cur = None
            for f in factors:
                if isinstance(f, matrix_atom) and f.is_central():
                    continue
                if isinstance(f, symbolic_matrix) and f.op == "pow":
                    b, e = f.args
                    if isinstance(b, matrix_atom) and b.is_central():
                        continue

                shp = getattr(f, "shape", None)
                if shp is None or shp == (1, 1):
                    continue

                if cur is None or cur == (1, 1):
                    cur = shp
                    continue

                if cur[1] != shp[0]:
                    raise TypeError(
                        f"* between tagged shapes is not supported: {cur} vs {shp}"
                    )
                cur = (cur[0], shp[1])

            return cur
        if op == "pow":
            base, exp = self.args
            shp = getattr(base, "shape", None)
            if shp is None:
                return None
            if shp == (1, 1):
                return (1, 1)
            if shp[0] != shp[1]:
                raise TypeError(
                    f"power is not supported for non-square tagged shape {shp}"
                )
            return shp
        return None

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return f"symbolic_matrix({self._ast!r})"

    def _to_string(self):
        op = self.op
        if op == "scalar":
            return str(self.args[0])
        if op == "add":
            return " + ".join(str(a) for a in self.args)
        if op == "mul":
            coeff = self.args[0]
            factors = self.args[1:]
            core = " * ".join(str(f) for f in factors)
            if coeff == 1:
                return core
            return f"{coeff} * {core}"
        if op == "pow":
            base, exp = self.args
            return f"({base})**{exp}"
        return str(self._ast)

    def _repr_latex_(self, verbose: bool = False, raw: bool = False, **kwargs):
        def emit(x, parent_op=None):
            if isinstance(x, symbolic_matrix):
                op = x.op

                if op == "scalar":
                    return LaTeX(x.args[0])

                if op == "add":
                    s = " + ".join(emit(t, "add") for t in x.args).replace("+ -", "- ")
                    return rf"\left({s}\right)" if parent_op in {"mul", "pow"} else s

                if op == "mul":

                    def _strip_outer_parens(s):
                        s = s.strip()
                        if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
                            depth = 0
                            for i, ch in enumerate(s):
                                if ch == "(":
                                    depth += 1
                                elif ch == ")":
                                    depth -= 1
                                    if depth == 0 and i != len(s) - 1:
                                        return s
                            return s[1:-1].strip()
                        return s

                    def _needs_parens_coeff(s):
                        s = s.strip()
                        if not s:
                            return False

                        i = 0
                        n = len(s)
                        ctx = []
                        frac_wait = 0

                        while i < n:
                            if s.startswith(r"\frac", i):
                                i += 5
                                frac_wait = 2
                                continue

                            ch = s[i]

                            if ch == "^" and i + 1 < n and s[i + 1] == "{":
                                ctx.append("sup")
                                i += 2
                                continue
                            if ch == "_" and i + 1 < n and s[i + 1] == "{":
                                ctx.append("sub")
                                i += 2
                                continue

                            if ch == "{":
                                if frac_wait > 0:
                                    ctx.append("frac")
                                    frac_wait -= 1
                                else:
                                    ctx.append("brace")
                                i += 1
                                continue
                            if ch == "}":
                                if ctx:
                                    ctx.pop()
                                i += 1
                                continue

                            if (ch == "+" or ch == "-") and not ctx:
                                return True

                            i += 1

                        return False

                    def _split_leading_minus(s):
                        s = s.strip()
                        if s.startswith("-"):
                            return True, s[1:].strip()
                        return False, s

                    coeff = x.args[0]
                    factors = x.args[1:]
                    core = " ".join(emit(f, "mul") for f in factors)

                    if coeff == 1:
                        s = core
                    elif coeff == -1:
                        s = rf"-{core}"
                    else:
                        c0 = _strip_outer_parens(LaTeX(coeff))
                        has_minus, tail = _split_leading_minus(c0)

                        if has_minus:
                            if _needs_parens_coeff(tail):
                                c = rf"\left(-{tail}\right)"
                                s = rf"{c}\,{core}" if core else c
                            else:
                                c = f"-{tail}"
                                s = rf"{c}\,{core}" if core else c
                        else:
                            c = rf"\left({c0}\right)" if _needs_parens_coeff(c0) else c0
                            s = rf"{c}\,{core}" if core else c

                    return rf"\left({s}\right)" if parent_op == "pow" else s

                if op == "pow":
                    return rf"{emit(x.args[0], 'pow')}^{{{LaTeX(x.args[1])}}}"

                if op == "conj":
                    return rf"\overline{{{emit(x.args[0])}}}"

                if op == "tran":
                    return rf"{emit(x.args[0])}^{{T}}"
            return LaTeX(x)

        out = emit(self)
        return out if raw else rf"$\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)
