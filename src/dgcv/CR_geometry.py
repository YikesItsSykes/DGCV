"""
package: dgcv - Differential Geometry with Complex Variables
module: CR_geometry

Description: This module provides tools specific to CR (Cauchy-Riemann) geometry within the dgcv package.
It includes functions for constructing CR hypersurfaces and  computing symmetries.

Key Functions:
    - tangencyObstruction(): Computes the tangency obstruction for a holomorphic vector field's
      real part to be tangent to a CR hypersurface.
    - weightedHomogeneousVF(): Creates a general weighted homogeneous vector field in a
      specified coordinate space.
    - findWeightedCRSymmetries(): Attempts to find infinitesimal symmetries of a weighted CR
      hypersurface.
    - model2Nondegenerate(): Builds the defining equation for a 2-nondegenerate model
      hypersurface.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

from ._safeguards import create_key, query_dgcv_categories, retrieve_passkey
from .arrays import matrix_dgcv
from .backends import simplify_dgcv
from .backends._engine import engine_module
from .backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    cancel,
    conjugate,
    expand,
    get_free_symbols,
    im,
    subs,
)
from .backends._types_and_constants import imag_unit
from .conversions import allToReal
from .dgcv_core import holVF_coeffs, polynomial_dgcv, realPartOfVF, variableProcedure
from .polynomials import createMultigradedPolynomial
from .solvers import solve_dgcv
from .vector_fields_and_differential_forms import get_VF
from .vmf import clearVar, listVar, vmf_lookup

__all__ = [
    "tangencyObstruction",
    "weightedHomogeneousVF",
    "findWeightedCRSymmetries",
    "model2Nondegenerate",
]


# -----------------------------------------------------------------------------
# CR tools
# -----------------------------------------------------------------------------
def tangencyObstruction(
    vf,
    CR_defining_expr,
    graph_variable,
    simplify: bool = False,
    data_already_in_real_coor: bool = False,
    *args,
):
    CR_defs = _as_list(CR_defining_expr)
    graph_vars = _as_list(graph_variable)

    if len(CR_defs) != len(graph_vars):
        raise ValueError(
            "`graph_variable` and `CR_defining_expr` must have the same length."
        )

    if not data_already_in_real_coor:
        graph_vars = [allToReal(v) for v in graph_vars]
        CR_defs = [allToReal(f) for f in CR_defs]
        vf = allToReal(vf)

    rVF = realPartOfVF(vf)
    real_eval = [rVF(gv - f) for gv, f in zip(graph_vars, CR_defs)]

    mapping = dict(zip(graph_vars, CR_defs))
    out = [subs(expr, mapping) for expr in real_eval]

    if simplify:
        out = [simplify_dgcv(x) for x in out]

    return out


def weightedHomogeneousVF(
    varSpace,
    weight,
    weights,
    varLabel,
    degreeCap: int = 0,
    _tempVar=None,
    assumeReal=None,
):
    if not _is_sequence(weight):
        weight = [weight]
    if not weights or not _is_sequence(weights[0]):
        weights = [weights]

    weight_list = list(weight)
    weight_systems = list(weights)

    if not all(len(ws) == len(varSpace) for ws in weight_systems):
        raise KeyError(
            "`weightedHomogeneousVF` expects each weight system to match len(varSpace)."
        )
    if len(weight_list) != len(weight_systems):
        raise KeyError(
            "`weightedHomogeneousVF` expects len(weight) == number of weight systems."
        )

    polys = []
    for j in range(len(varSpace)):
        degs = [d + ws[j] for d, ws in zip(weight_list, weight_systems)]
        polys.append(
            createMultigradedPolynomial(
                f"{varLabel}_{j}_",
                degs,
                varSpace,
                weight_systems,
                degreeCap=degreeCap,
                _tempVar=_tempVar,
                assumeReal=assumeReal,
                report=False,
            )
        )

    Dz_list = get_VF(*varSpace)
    out = None
    for p, Dz in zip(polys, Dz_list):
        if not p:
            continue
        term = p * Dz
        out = term if out is None else (out + term)

    if out is None:
        out = type(Dz_list[0])(coeff_dict={tuple(): 0}, data_shape="all")

    return out


def findWeightedCRSymmetries(
    graph_function,
    holomorphic_coordinates,
    coordinate_weights,
    symmetry_weight,
    graph_variable,
    coeff_label=None,
    degreeCap: int = 0,
    returnVectorFieldBasis: bool = False,
    returnAllformats: bool = False,
    simplifyingFactor=None,
    assume_polynomial: bool = False,
    simplify: bool = False,
    parameters=None,
):
    if not coordinate_weights or not _is_sequence(coordinate_weights[0]):
        coordinate_weights = [coordinate_weights]
    if not _is_sequence(symmetry_weight):
        symmetry_weight = [symmetry_weight]

    weight_systems = list(coordinate_weights)
    sym_weight_list = list(symmetry_weight)

    if len(sym_weight_list) != len(weight_systems):
        raise KeyError(
            "`findWeightedCRSymmetries` requires one symmetry weight per weight system."
        )

    if returnAllformats:
        returnVectorFieldBasis = True

    def _coord_ri_from_vmf(coords):
        out = set()
        for v in coords:
            info = vmf_lookup(v, relatives=True)
            if info.get("type") != "coordinate":
                continue
            rel = info.get("relatives") or {}
            r = rel.get("real", None)
            i = rel.get("imag", None)
            if r is not None:
                out.add(r)
            if i is not None:
                out.add(i)
            if r is None and i is None:
                out.add(v)
        return out

    vfA = weightedHomogeneousVF(
        holomorphic_coordinates,
        sym_weight_list,
        weight_systems,
        "ALoc",
        _tempVar=retrieve_passkey(),
        degreeCap=degreeCap,
        assumeReal=True,
    )
    vfB = weightedHomogeneousVF(
        holomorphic_coordinates,
        sym_weight_list,
        weight_systems,
        "BLoc",
        _tempVar=retrieve_passkey(),
        degreeCap=degreeCap,
        assumeReal=True,
    )

    vf_candidate_holo = vfA + imag_unit() * vfB

    tan_obst_list = tangencyObstruction(
        vf_candidate_holo,
        graph_function,
        graph_variable,
        simplify=simplify,
    )

    free = get_free_symbols
    unwrap = _unwrap_symbolic

    var_all = set()
    for TO in tan_obst_list:
        var_all.update(free(unwrap(TO)))

    coord_ri = _coord_ri_from_vmf(holomorphic_coordinates)

    unknowns = set(var_all) - set(coord_ri)
    if parameters is not None:
        unknowns -= set(parameters)

    if coeff_label is None:
        coeff_label = create_key(prefix="X", key_length=5)

    unknowns_t = _stable_tuple(unknowns)

    graph_vars = _as_list(graph_variable)
    poly_gens_vars = _stable_tuple(list(coord_ri) + list(graph_vars))

    coef_eqns = []
    for TO in tan_obst_list:
        expr = unwrap(TO)
        expr = allToReal(expr)
        if not assume_polynomial:
            expr = as_numer_denom(cancel(expr))[0]
        expr = expand(expr)
        P = (
            expr
            if query_dgcv_categories(expr, "polynomial")
            else polynomial_dgcv(expr, poly_gens_vars)
        )
        coef_eqns.extend([allToReal(c) for c in P.get_coeffs(format="unformatted")])

    solutions = solve_dgcv(coef_eqns, unknowns_t, method="linsolve")

    if not solutions:
        if len(tan_obst_list) == 0 or all(
            _scalar_is_zero(unwrap(x)) for x in tan_obst_list
        ):
            clearVar(*listVar(temporary_only=True), report=False)
            if not returnVectorFieldBasis:
                return [0 for _ in holomorphic_coordinates]
            if returnAllformats:
                return [0 for _ in holomorphic_coordinates], []
            return []
        clearVar(*listVar(temporary_only=True), report=False)
        raise ValueError(
            "findWeightedCRSymmetries: no solution to the coefficient system."
        )

    sol = solutions[0] if isinstance(solutions, (list, tuple)) else solutions

    vf_solved = subs(vf_candidate_holo, sol)

    VFCLoc = holVF_coeffs(vf_solved, holomorphic_coordinates)

    sub_syms = set()
    for term in VFCLoc:
        sub_syms.update(free(unwrap(term)))

    sub_syms -= set(holomorphic_coordinates)
    sub_syms -= set(coord_ri)
    if parameters is not None:
        sub_syms.difference_update(parameters)

    if not sub_syms:
        clearVar(*listVar(temporary_only=True), report=False)
        if not returnVectorFieldBasis:
            return VFCLoc
        if returnAllformats:
            return VFCLoc, []
        return []

    sub_syms_t = _stable_tuple(sub_syms)

    created = variableProcedure(
        coeff_label,
        len(sub_syms_t),
        assumeReal=True,
        return_created_object=True,
    )
    coeff_vars = tuple(created[0]) if created else ()

    repl = dict(zip(sub_syms_t, coeff_vars))
    vf_param = subs(vf_solved, repl)

    clearVar(*listVar(temporary_only=True), report=False)

    if not returnVectorFieldBasis:
        return holVF_coeffs(vf_param, holomorphic_coordinates)

    zeroing = {w: 0 for w in coeff_vars}
    VF_basis = []
    for v in coeff_vars:
        d = dict(zeroing)
        d[v] = 1
        VF_basis.append(subs(vf_param, d))

    clearVar(coeff_label, report=False)

    if returnAllformats:
        return holVF_coeffs(vf_param, holomorphic_coordinates), VF_basis
    return VF_basis


def model2Nondegenerate(
    hermitian_matrix: Any,
    symmetric_matrix_function: Any,
    base_coordinates: Sequence[Any],
    transverse_coordinate: Any,
    return_matrices: bool = False,
    simplify: bool = True,
):
    mod = _sym_engine_module()

    def _simp(x: Any) -> Any:
        return simplify_dgcv(x) if simplify else x

    def _as_md(mat: Any):
        if query_dgcv_categories(mat, {"matrix"}):
            return mat
        try:
            return matrix_dgcv(mat)
        except Exception:
            pass

        if hasattr(mod, "Matrix"):
            try:
                return matrix_dgcv(mod.Matrix(mat))
            except Exception:
                pass
        if hasattr(mod, "matrix"):
            try:
                return matrix_dgcv(mod.matrix(mat))
            except Exception:
                pass
        raise TypeError("`model2Nondegenerate` expects array-like matrix data.")

    def _eye(n: int):
        try:
            return matrix_dgcv(
                [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            )
        except Exception:
            pass
        if hasattr(mod, "eye"):
            return mod.eye(n)
        if hasattr(mod, "eye_matrix"):
            return mod.eye_matrix(n)
        if hasattr(mod, "identity_matrix"):
            return mod.identity_matrix(n)
        if hasattr(mod, "Identity"):
            return mod.Identity(n)
        raise TypeError("No identity matrix constructor available.")

    def _T(M: Any):
        t = getattr(M, "T", None)
        if t is not None:
            return t() if callable(t) else t
        tr = getattr(M, "transpose", None)
        if callable(tr):
            return tr()
        if hasattr(mod, "transpose"):
            return mod.transpose(M)
        raise TypeError("Matrix transpose not supported for this object.")

    def _inv(M: Any):
        inv = getattr(M, "inv", None)
        if callable(inv):
            return inv()
        try:
            return M ** (-1)
        except Exception:
            pass
        inv2 = getattr(M, "inverse", None)
        if callable(inv2):
            return inv2()
        raise TypeError("Matrix inversion not supported for this object.")

    def _mat_conj(M: Any):
        if query_dgcv_categories(M, {"matrix"}):
            return M.apply(conjugate, in_place=False, skip_none=True)
        try:
            return conjugate(M)
        except Exception:
            pass
        raise TypeError("Matrix conjugation not supported for this object.")

    A = _as_md(hermitian_matrix)
    S = _as_md(symmetric_matrix_function)

    shp = getattr(A, "shape", None)
    if not (isinstance(shp, tuple) and len(shp) == 2 and shp[0] == shp[1]):
        raise TypeError("`hermitian_matrix` must be square.")
    n = int(shp[0])

    Iden = _eye(n)
    BARS = _mat_conj(S)

    zVec = matrix_dgcv([[v] for v in base_coordinates])
    bzVec = matrix_dgcv([[conjugate(v)] for v in base_coordinates])

    half = mod.Rational(1, 2) if hasattr(mod, "Rational") else (1 / 2)

    At = _T(A)
    zt = _T(zVec)
    bzt = _T(bzVec)

    M1 = Iden - (BARS * At * S * A)
    M2 = Iden - (A * BARS * At * S)

    hFun = half * (A * _inv(M1) + _inv(M2) * A)
    sFun = A * _inv(M1) * BARS * At
    bsFun = At * _inv(Iden - (S * A * BARS * At)) * S * A

    expr = (zt * hFun * bzVec)[0, 0] + half * (
        (zt * sFun * zVec)[0, 0] + (bzt * bsFun * bzVec)[0, 0]
    )

    result = _simp(expr) - im(transverse_coordinate)

    if return_matrices:
        return result, hFun, sFun
    return result


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def _sym_engine_module():
    return engine_module()


def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _as_list(x: Any) -> List[Any]:
    return list(x) if _is_sequence(x) else [x]


def _stable_tuple(seq: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen = set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return tuple(out)


def _unwrap_symbolic(obj: Any) -> Any:
    if obj is None:
        return obj

    f = getattr(obj, "as_expr", None)
    if callable(f):
        try:
            return f()
        except TypeError:
            return f

    for attr in ("polyExpr", "_polyExpr", "expr", "_expr"):
        if hasattr(obj, attr):
            return getattr(obj, attr)

    P = getattr(obj, "poly_obj_unformatted", None)
    if P is not None and hasattr(P, "as_expr"):
        try:
            return P.as_expr()
        except Exception:
            pass

    return obj
