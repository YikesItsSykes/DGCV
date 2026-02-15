"""
package: dgcv - Differential Geometry with Complex Variables
module: polynomials

Description: Polynomial construction utilities for dgcv.

This module provides creator functions that build (multi)graded polynomials in a given
variable space, using dgcv's variable management framework to generate coefficient
families. The returned objects are dgcv.polynomial_dgcv instances.

key functions
---------
- createPolynomial
- createBigradPolynomial
- createMultigradedPolynomial
- monomialWeight
- getWeightedTerms

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import functools
import numbers
import operator
import warnings
from typing import Any, List, Optional, Sequence, Union

from .backends._polynomials import make_poly, poly_coeffs, poly_gens, poly_monoms
from .backends._symbolic_router import expand, simplify, subs
from .backends._types_and_constants import zero
from .combinatorics import chooseOp
from .dgcv_core import polynomial_dgcv, variableProcedure
from .vmf import clearVar

ScalarLike = Union[int, float, complex]
WeightVec = Sequence[int]
WeightSystems = Sequence[Sequence[int]]


__all__ = [
    "createPolynomial",
    "createBigradPolynomial",
    "createMultigradedPolynomial",
    "monomialWeight",
    "getWeightedTerms",
]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def _monomial_from_exponents(vars_: Sequence[Any], exps: Sequence[int]) -> Any:
    return functools.reduce(
        operator.mul, (v ** int(e) for v, e in zip(vars_, exps) if int(e) != 0), 1
    )


def createPolynomial(
    coeff_label: str,
    degree: int,
    variables: Sequence[Any],
    homogeneous: bool = False,
    weightedHomogeneity: Optional[Sequence[int]] = None,
    degreeCap: int = 0,
    _tempVar: Optional[bool] = None,
    returnMonomialList: bool = False,
    assumeReal: Optional[bool] = None,
    remove_guardrails: bool = False,
    report: bool = True,
) -> Union[polynomial_dgcv, List[Any]]:
    clearVar(coeff_label, report=report)

    vars_ = tuple(variables)
    n = len(vars_)
    deg = int(degree)

    if homogeneous:
        exponent_tuples = list(chooseOp(range(deg + 1), n, restrictHomogeneity=deg))

    elif isinstance(weightedHomogeneity, (list, tuple)):
        w = list(weightedHomogeneity)
        if len(w) != n:
            raise ValueError("weightedHomogeneity must match len(variables).")

        if 0 in w:
            zero_idx = [j for j, ww in enumerate(w) if ww == 0]
            nonzero_idx = [j for j, ww in enumerate(w) if ww != 0]

            vars_reordered = tuple(
                [vars_[j] for j in nonzero_idx] + [vars_[j] for j in zero_idx]
            )
            w_reordered = [w[j] for j in nonzero_idx] + [w[j] for j in zero_idx]

            nonzero_exps = [
                e
                for e in chooseOp(range(deg + 1), len(nonzero_idx))
                if sum(a * b for a, b in zip(e, w_reordered[: len(nonzero_idx)])) == deg
            ]
            zero_exps = list(chooseOp(range(int(degreeCap) + 1), len(zero_idx)))
            exponent_tuples = [e0 + e1 for e0 in nonzero_exps for e1 in zero_exps]
            vars_ = vars_reordered
        else:
            exponent_tuples = [
                e
                for e in chooseOp(range(deg + 1), n)
                if sum(a * b for a, b in zip(e, w)) == deg
            ]

    else:
        exponent_tuples = [e for e in chooseOp(range(deg + 1), n) if sum(e) <= deg]

    created = variableProcedure(
        coeff_label,
        len(exponent_tuples),
        _tempVar=_tempVar,
        assumeReal=assumeReal,
        remove_guardrails=remove_guardrails,
        return_created_object=True,
    )
    coeffs = created[0] if created else ()

    monomials: List[Any] = []
    for idx, exps in enumerate(exponent_tuples):
        monomials.append(coeffs[idx] * _monomial_from_exponents(vars_, exps))

    if returnMonomialList:
        return monomials

    expr = sum(monomials) if monomials else zero()
    return polynomial_dgcv(expr, varSpace=vars_)


def createBigradPolynomial(
    coeff_label: str,
    degrees: Sequence[int],
    variables: Sequence[Any],
    weights_1: Sequence[int],
    weights_2: Sequence[int],
    _tempVar: Optional[bool] = None,
    returnMonomialList: bool = False,
    remove_guardrails: bool = False,
    report: bool = True,
) -> Union[polynomial_dgcv, List[Any]]:
    clearVar(coeff_label, report=report)

    degs = tuple(degrees)
    if len(degs) != 2:
        raise ValueError("degrees must have length 2.")
    d0, d1 = int(degs[0]), int(degs[1])

    vars_ = tuple(variables)
    w0 = tuple(weights_1)
    w1 = tuple(weights_2)
    if not (len(vars_) == len(w0) == len(w1)):
        raise ValueError("variables, weights_1, weights_2 must have the same length.")

    max_deg = max(abs(d0), abs(d1))
    candidates = chooseOp(range(max_deg + 1), len(vars_))
    exponent_tuples = [
        exps
        for exps in candidates
        if sum(e * ww for e, ww in zip(exps, w0)) == d0
        and sum(e * ww for e, ww in zip(exps, w1)) == d1
    ]

    created = variableProcedure(
        coeff_label,
        len(exponent_tuples),
        _tempVar=_tempVar,
        remove_guardrails=remove_guardrails,
        return_created_object=True,
    )
    coeffs = created[0] if created else ()

    monomials: List[Any] = []
    for idx, exps in enumerate(exponent_tuples):
        monomials.append(coeffs[idx] * _monomial_from_exponents(vars_, exps))

    if returnMonomialList:
        return monomials

    expr = sum(monomials) if monomials else zero()
    return polynomial_dgcv(expr, varSpace=vars_)


def createMultigradedPolynomial(
    coeff_label: str,
    degrees: Union[int, Sequence[int]],
    vars: Sequence[Any],
    weight_systems: Union[Sequence[int], Sequence[Sequence[int]]],
    _tempVar: Optional[bool] = None,
    returnMonomialList: bool = False,
    remove_guardrails: bool = False,
    assumeReal: bool = False,
    report: bool = True,
    degreeCap: int = 10,
) -> Union[polynomial_dgcv, List[Any]]:
    clearVar(coeff_label, report=report)

    degs = tuple(degrees) if isinstance(degrees, (list, tuple)) else (int(degrees),)
    ws = weight_systems
    if not isinstance(ws, (list, tuple)) or (
        ws and not isinstance(ws[0], (list, tuple))
    ):
        ws = [ws]
    ws = [tuple(w) for w in ws]

    vars_ = tuple(vars)

    if len(degs) != len(ws):
        raise KeyError("degrees and weight_systems must have the same length.")
    if any(len(w) != len(vars_) for w in ws):
        raise ValueError("Each weight vector must match len(vars).")

    max_deg = 0
    for d, w in zip(degs, ws):
        if all(j > 0 for j in w) or all(j < 0 for j in w):
            dd = int(abs(d))
            if max_deg == 0 or dd < max_deg:
                max_deg = dd

    if not isinstance(max_deg, numbers.Integral):
        max_deg = int(abs(max_deg))
    if max_deg == 0:
        max_deg = max(int(abs(d)) for d in degs)
        max_deg = max(max_deg, int(degreeCap))

    candidates = chooseOp(range(max_deg + 1), len(vars_))
    exponent_tuples = [
        exps
        for exps in candidates
        if all(
            sum(e * wj for e, wj in zip(exps, ws[k])) == int(degs[k])
            for k in range(len(degs))
        )
    ]

    created = variableProcedure(
        coeff_label,
        len(exponent_tuples),
        _tempVar=_tempVar,
        assumeReal=assumeReal,
        remove_guardrails=remove_guardrails,
        return_created_object=True,
    )
    coeffs = created[0] if created else ()

    monomials: List[Any] = []
    for idx, exps in enumerate(exponent_tuples):
        monomials.append(coeffs[idx] * _monomial_from_exponents(vars_, exps))

    if returnMonomialList:
        return monomials

    expr = sum(monomials) if monomials else zero()
    return polynomial_dgcv(expr, varSpace=vars_)


def monomialWeight(
    monomial: Any, variables: Sequence[Any], weights: Sequence[int]
) -> Any:
    """
    Return the weighted degree of a monomial in `variables` with weight vector `weights`.
    """
    if len(variables) != len(weights):
        raise ValueError("variables and weights must have the same length.")

    expr = simplify(expand(monomial))

    try:
        P = make_poly(expr, variables)
        terms = poly_monoms(P)
        if len(terms) > 1:
            warnings.warn("Input appears to have multiple terms; proceeding anyway.")
    except Exception:
        warnings.warn(
            "Input could not be interpreted as a polynomial; proceeding anyway."
        )

    coeff = simplify(subs(expr, {v: 1 for v in variables}))
    if coeff == 0:
        return zero()

    try:
        P = make_poly(expr, variables)
        monoms = poly_monoms(P)
        if not monoms:
            return zero()
        m = monoms[0]
        return sum(int(e) * int(w) for e, w in zip(m, weights))
    except Exception:
        return sum(int(w) for w in weights) * 0


def getWeightedTerms(
    poly: polynomial_dgcv,
    target_degrees: Sequence[int],
    weight_systems: Sequence[Sequence[int]],
    *,
    format: str = "unformatted",
) -> polynomial_dgcv:
    """
    Filter terms of `poly` whose monomials satisfy weighted degree constraints.
    """
    if len(target_degrees) != len(weight_systems):
        raise ValueError("target_degrees and weight_systems must have the same length.")

    if format == "unformatted":
        P = poly.poly_obj_unformatted
    elif format == "complex":
        P = poly.poly_obj_complex
    elif format == "real":
        P = poly.poly_obj_real
    else:
        raise ValueError("format must be one of: 'unformatted', 'complex', 'real'")

    gens = poly_gens(P)
    monoms = poly_monoms(P)
    coeffs = poly_coeffs(P)

    admissible = list(range(len(monoms)))
    for deg, w in zip(target_degrees, weight_systems):
        if len(w) != len(gens):
            raise ValueError(
                "Each weight vector must have length equal to number of gens."
            )
        admissible = [
            j
            for j in admissible
            if sum(int(monoms[j][k]) * int(w[k]) for k in range(len(gens))) == int(deg)
        ]

    terms: List[Any] = []
    for j in admissible:
        term = coeffs[j]
        for g, e in zip(gens, monoms[j]):
            ee = int(e)
            if ee:
                term = term * (g**ee)
        terms.append(term)

    expr = sum(terms) if terms else zero()
    return polynomial_dgcv(expr, varSpace=poly.varSpace, parameters=poly.parameters)
