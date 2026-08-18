from __future__ import annotations

import random
import uuid
from fractions import Fraction

from ..._aux._backends._engine import engine_kind, engine_module
from ..._aux._backends._polynomials import (
    _ordered_factors,
    _unwrap_pow,
    as_numer_denom,
    make_poly,
    poly_coeffs,
    poly_monoms,
    poly_total_degree,
)
from ..._aux._backends._symbolic_router import (
    IndeterminateSignError,
    _scalar_is_zero,
    _scalar_sign,
    factor,
    get_free_symbols,
    ratio,
    simplify,
    subs,
)
from ..._aux._backends._types_and_constants import symbol
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import zip_sum
from ..._aux._vmf._safeguards import create_key, get_dgcv_category
from ...core.arrays import _as_matrix_dgcv, matrix_dgcv
from ...core.solvers import solve_dgcv
from .algebra_classifications import RealFormReport, complex_type_from_root_lengths

# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------


def fast_rank(mat, surface_singularities=False, simplify_singularities=None) -> int:
    M = _as_matrix_dgcv(mat)
    if M is None:
        M = matrix_dgcv(mat)
    return M.rank(
        allow_formal_inverse=surface_singularities,
        simplify_steps=False
        if not surface_singularities
        else simplify_singularities
        if simplify_singularities is not None
        else True,
        record_divisors=surface_singularities,
    )


def _commutant_eigenspace_vectors_old(
    solMat,
    *,
    tries=30,
    bound=None,
):
    n = solMat.nrows

    free_vars = set()
    for v in solMat._data.values():
        if v is None:
            continue
        free_vars |= get_free_symbols(v)
    free_vars = list(free_vars)

    if bound is None:
        bound = max(100, 10 * n)

    last_err = None

    for _ in range(max(1, int(tries))):
        if free_vars:
            spec = {var: random.randint(1, bound) for var in free_vars}
            M = subs(solMat, spec)
        else:
            M = solMat

        try:
            if engine_kind() == "sympy":
                sp = engine_module()
                lam = sp.Symbol(create_key(prefix="lam"))
                Id = matrix_dgcv.identity(n)

                try:
                    cp = sp.Matrix(M.to_list()).charpoly(lam).as_poly(lam)
                except Exception:
                    cp = sp.Poly(simplify((M - lam * Id).det()), lam)

                try:
                    evals = list(cp.all_roots())
                except Exception:
                    evals = []

                evals = [r for r in evals if r is not None]
                evals_u = []
                for r in evals:
                    if r not in evals_u:
                        evals_u.append(r)
                if len(evals_u) < 2:
                    continue

                eigspaces = []
                for r in evals_u:
                    ns = (M - r * Id).nullspace()
                    if ns:
                        eigspaces.append((r, ns))

                if len(eigspaces) < 2:
                    continue

            else:
                eigdata = M._eigenvects_by_engine()
                eigspaces = [(lam, vecs) for (lam, _mult, vecs) in eigdata if vecs]
                if len(eigspaces) < 2:
                    continue

            cols = []
            for _, vecs in eigspaces:
                for v in vecs:
                    if isinstance(v, matrix_dgcv):
                        cols.append([v[i, 0] for i in range(v.nrows)])
                    else:
                        cols.append(list(v))

            basis_cols = []
            for c in cols:
                if not basis_cols:
                    basis_cols.append(c)
                    if len(basis_cols) == n:
                        break
                    continue

                r0 = matrix_dgcv.from_cols(basis_cols).rank()
                r1 = matrix_dgcv.from_cols(basis_cols + [c]).rank()
                if r1 > r0:
                    basis_cols.append(c)

                if len(basis_cols) == n:
                    break

            if len(basis_cols) != n:
                continue

            return M, [r for r, _ in eigspaces], eigspaces

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Unable to obtain a commutant specialization yielding >= 2 eigenspaces and a full spanning set of eigenvectors."
    ) from last_err


def _commutant_eigenspace_vectors(mat, free_vars, max_attempts=6):
    ordered_vars = sorted(free_vars, key=str)
    expected = len(ordered_vars)
    dim = mat.shape[0]
    for attempt in range(max_attempts):
        rng = random.Random(9000 + attempt)
        weights = rng.sample(range(1, 16 * (attempt + 2)), expected)
        specialized = mat.subs(dict(zip(ordered_vars, weights)))
        try:
            eigen_data = specialized._eigenvects_by_engine()
        except Exception:
            continue
        if len(eigen_data) != expected:
            continue
        vectors = [list(edata[-1]) for edata in eigen_data]
        if sum(len(block) for block in vectors) != dim:
            continue
        return vectors
    return None


def _centroid_ideal_components(solMat, free_vars, attempts=8):
    order = sorted(free_vars, key=str)
    var = symbol(create_key("_cpoly"))
    for attempt in range(attempts):
        bound = 8 * (attempt + 1)
        z = solMat.subs({v: random.randint(-bound, bound) for v in order})
        minimal_poly = _matrix_minimal_polynomial(z, len(order), var)
        if minimal_poly is None:
            continue
        degree, poly_expr = minimal_poly
        if degree != len(order):
            continue
        return _split_along_minimal_polynomial(z, poly_expr, var)
    raise RuntimeError(
        "decompose_semisimple_algebra could not certify a generator of the centroid "
        f"after {attempts} random specializations, so the decomposition into simple "
        "ideals cannot be trusted. This is expected when the given algebra is not "
        "semisimple."
    )


def _matrix_minimal_polynomial(mat, degree_bound, var):
    n = mat.nrows
    columns = []
    power = matrix_dgcv.identity(n)
    for step in range(degree_bound + 1):
        columns.append([power[i] for i in range(n * n)])
        if step < degree_bound:
            power = power @ mat
    relations = matrix_dgcv.from_cols(columns).nullspace()
    if not relations:
        return None
    relation = relations[0]
    degree = None
    for k in range(degree_bound, -1, -1):
        if not _scalar_is_zero(relation[k]):
            degree = k
            break
    if degree is None:
        return None
    lead = relation[degree]
    poly_expr = 0
    for k in range(degree + 1):
        c = relation[k]
        if _scalar_is_zero(c):
            continue
        poly_expr = poly_expr + ratio(c, lead) * var**k
    return degree, poly_expr


def _split_along_minimal_polynomial(mat, poly_expr, var):
    seen = set()
    irreducibles = []
    for f in _ordered_factors(factor(poly_expr)):
        base = _unwrap_pow(f)
        degree = poly_total_degree(base, [var])
        if not degree:
            continue
        stamp = str(base)
        if stamp in seen:
            continue
        seen.add(stamp)
        irreducibles.append((base, degree))

    components = []
    for base, degree in irreducibles:
        coeffs = _univariate_coefficients(base, var, degree)
        centroid_type = None
        if degree == 1:
            centroid_type = "real"
        elif degree == 2:
            try:
                if _scalar_sign(coeffs[1] ** 2 - 4 * coeffs[2] * coeffs[0]) < 0:
                    centroid_type = "complex"
            except IndeterminateSignError:
                pass
        vectors = _evaluate_polynomial_at_matrix(coeffs, mat).nullspace()
        if vectors:
            components.append((vectors, centroid_type))
    components.sort(key=_component_order_key)
    return components


def _univariate_coefficients(expr, var, degree):
    poly = make_poly(expr, [var])
    coeffs = [0] * (degree + 1)
    for monom, c in zip(poly_monoms(poly), poly_coeffs(poly)):
        coeffs[monom[0]] = c
    return coeffs


def _evaluate_polynomial_at_matrix(coeffs, mat):
    n = mat.nrows
    out = matrix_dgcv.zeros(n, n)
    power = matrix_dgcv.identity(n)
    last = len(coeffs) - 1
    for k, c in enumerate(coeffs):
        if not _scalar_is_zero(c):
            out = out + c * power
        if k < last:
            power = power @ mat
    return out


def _component_order_key(component):
    leads = []
    for v in component[0]:
        positions = [i for i in range(len(v)) if not _scalar_is_zero(v[i])]
        leads.append(positions[0] if positions else -1)
    return (len(component[0]), tuple(sorted(leads)))


def adjointRepresentation(alg, list_format=False, assume_Lie_algebra=False):
    if get_dgcv_category(alg) in {"algebra", "subalgebra"}:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            dgcv_warning(
                "The algebra passed to `adjointRepresentation` is not a Lie algebra; there is likely a mistake if applying  `adjointRepresentation`."
            )
        get_slice = alg._structure_data_slice
        shp = (alg.dimension, alg.dimension)
        return [
            matrix_dgcv(get_slice(idx), shape=shp).transpose()
            for idx in range(alg.dimension)
        ]
    else:
        raise Exception(
            "adjointRepresentation expected to receive an algebra instance."
        ) from None


def decompose_semisimple_algebra(
    alg,
    assume_semisimple=False,
    format_as_lists_of_elements=False,
    surface_singularities=False,
    simplify_singularities=None,
    return_centroid_types=False,
):
    """
    Decompose a semisimple Lie algebra into simple ideals.

    Parameters
    ----------
    alg : algebra_class or subalgebra_class
        Algebra to decompose.
    assume_semisimple : bool, default False
        Skip the semisimplicity check.
    format_as_lists_of_elements : bool, default False
        Return each ideal as a list of basis elements rather than a subalgebra.
    surface_singularities : bool, default False
        Also return parameter-space singularities raised by the linear solver.
    simplify_singularities : bool, optional
        Forwarded to the linear solver when surfacing singularities.
    return_centroid_types : bool, default False
        Also return one centroid type per ideal: `"real"` for an absolutely
        simple ideal, `"complex"` for a realification of a complex simple
        algebra, or `None` when undetermined.

    Returns
    -------
    list
        The simple ideals, followed by the centroid types and then the
        singularities when either is requested.
    """
    assert get_dgcv_category(alg) in {"algebra", "subalgebra"}

    sing = []

    def _package(components, centroid_types):
        if return_centroid_types is True:
            if surface_singularities is True:
                return components, tuple(centroid_types), sing
            return components, tuple(centroid_types)
        if surface_singularities is True:
            return components, sing
        return components

    def _whole_algebra(centroid_type):
        components = [list(alg.basis)] if format_as_lists_of_elements else [alg]
        return _package(components, [centroid_type])

    if alg.dimension == 0:
        return _package([alg], [None])
    if assume_semisimple is False and not alg.is_semisimple():
        raise TypeError(
            "decompose_semisimple_algebra was given a non-semisimple algebra to decompose."
        )

    n = alg.dimension
    get_slice = alg._structure_data_slice
    slice_shape = (n, n)
    mbasis = [
        matrix_dgcv(
            {(k, j): v for (j, k), v in get_slice(idx).items()}, shape=slice_shape
        )
        for idx in range(n)
    ]

    pref = create_key("_var")
    variables = [symbol(f"{pref}{j}") for j in range(n * n)]
    vMat = matrix_dgcv(dict(enumerate(variables)), shape=(n, n))

    mats = []
    for mat in mbasis:
        comm = (vMat @ mat) - (mat @ vMat)
        mats += list(comm._data.values())
    if surface_singularities is True:
        sol, sing = solve_dgcv(
            mats,
            variables,
            method="linsolve",
            return_divisors=True,
            pass_to_symbolic_engine=False,
            simplify_pivots=simplify_singularities
            if simplify_singularities is not None
            else True,
            simplify_result=False,
        )
    else:
        sol = solve_dgcv(mats, variables, method="linsolve", simplify_result=False)
    if not sol:
        raise RuntimeError("solve_dgcv failed in decompose_semisimple_algebra.")

    solMat = vMat.subs(sol[0])

    free_vars = set()
    for v in solMat._data.values():
        if v is None:
            continue
        free_vars |= get_free_symbols(v)
    params = getattr(alg, "_parameters", set())
    if params:
        free_vars -= params
    if len(free_vars) < 2:
        return _whole_algebra("real")

    if params or getattr(alg, "base_field", "complex") == "complex":
        raw = [
            (vecs, None) for vecs in _commutant_eigenspace_vectors(solMat, free_vars)
        ]
    else:
        raw = _centroid_ideal_components(solMat, free_vars)
        if len(raw) == 1:
            return _whole_algebra(raw[0][1])

    simples = []
    centroid_types = []
    for vectors, centroid_type in raw:
        new_basis = [zip_sum(v, alg.basis) for v in vectors]
        if not new_basis:
            continue
        if format_as_lists_of_elements is True:
            simples.append(new_basis)
        else:
            ideal = alg.subalgebra(new_basis, simplify_basis=True)
            if centroid_type is not None:
                ideal._verified_ideal = True
                ideal._centroid_type = centroid_type
            simples.append(ideal)
        centroid_types.append(centroid_type)

    if not simples:
        return _whole_algebra(None)
    return _package(simples, centroid_types)


def killingForm(alg, assume_Lie_algebra=False):
    if get_dgcv_category(alg) not in {"algebra", "subalgebra"}:
        raise Exception(
            "killingForm expected to receive an algebra instance."
        ) from None
    if alg._killing_form is None:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            raise Exception(
                "killingForm expects argument to be a Lie algebra instance of the algebra"
            ) from None
        aRepLoc = adjointRepresentation(alg, assume_Lie_algebra=assume_Lie_algebra)
        alg._killing_form = matrix_dgcv(
            [
                [(aRepLoc[j] * aRepLoc[k]).trace() for k in range(alg.dimension)]
                for j in range(alg.dimension)
            ]
        )

    return alg._killing_form


def _combine_matrices(mats, coeffs):
    out = None
    for coeff, mat in zip(coeffs, mats):
        if _scalar_is_zero(coeff):
            continue
        term = coeff * mat
        out = term if out is None else out + term
    if out is None:
        return matrix_dgcv.zeros(mats[0].nrows, mats[0].ncols)
    return out


def _trace_of_product(left, right):
    total = 0
    for (i, j), value in left.iter_nonzero_items():
        entry = right[j, i]
        if _scalar_is_zero(entry):
            continue
        total = total + value * entry
    return total


def _cartan_casimir(ads, coeffs, dimension):
    cartan = _combine_matrices(ads, coeffs).nullspace()
    rank = len(cartan)
    if rank == 0 or rank >= dimension:
        return None
    adh = [_combine_matrices(ads, [vec[i] for i in range(dimension)]) for vec in cartan]
    for a in range(rank):
        for b in range(a + 1, rank):
            if any(True for _ in (adh[a] @ cartan[b]).iter_nonzero_items()):
                return None
    gram = matrix_dgcv(
        [[_trace_of_product(adh[a], adh[b]) for b in range(rank)] for a in range(rank)]
    )
    if _scalar_is_zero(gram.det()):
        return None
    inverse_gram = gram.inverse()
    casimir = None
    for a in range(rank):
        dual = _combine_matrices(adh, [inverse_gram[a, b] for b in range(rank)])
        term = adh[a] @ dual
        casimir = term if casimir is None else casimir + term
    return rank, casimir


def _as_fraction(value):
    try:
        numerator, denominator = as_numer_denom(value)
        return Fraction(int(numerator), int(denominator))
    except (TypeError, ValueError, AttributeError):
        return None


def _casimir_spectrum(casimir, rank, dimension):
    var = symbol(create_key("_rootLength"))
    minimal = _matrix_minimal_polynomial(casimir, 4, var)
    if minimal is None:
        return None
    degree, poly_expr = minimal
    if degree not in (2, 3):
        return None
    values = []
    for f in _ordered_factors(factor(poly_expr)):
        base = _unwrap_pow(f)
        base_degree = poly_total_degree(base, [var])
        if base_degree == 0:
            continue
        if base_degree != 1:
            return None
        coeffs = _univariate_coefficients(base, var, 1)
        value = _as_fraction(ratio(-coeffs[0], coeffs[1]))
        if value is None or value in values:
            return None
        values.append(value)
    if len(values) != degree or values.count(Fraction(0)) != 1:
        return None
    nonzero = sorted(value for value in values if value)
    root_count = dimension - rank
    if len(nonzero) == 1:
        multiplicities = [root_count]
    else:
        low, high = nonzero
        first = (rank - root_count * high) / (low - high)
        multiplicities = [first, root_count - first]
    lengths = []
    for value, count in zip(nonzero, multiplicities):
        if count <= 0 or count.denominator != 1:
            return None
        lengths.append((value, int(count)))
    if sum(value * count for value, count in lengths) != rank:
        return None
    if _as_fraction(_trace_of_product(casimir, casimir)) != sum(
        value * value * count for value, count in lengths
    ):
        return None
    return tuple(lengths)


def compute_root_length_profile(alg, attempts=4, assume_Lie_algebra=True):
    """
    Measure the root lengths of a complexified simple Lie algebra.

    Parameters
    ----------
    alg : algebra_class or subalgebra_class
        Assumed simple; the caller is responsible for that check.
    attempts : int, default 4
        Number of random elements tried before giving up.
    assume_Lie_algebra : bool, default True
        Skip the Lie algebra check when building the adjoint representation.

    Returns
    -------
    RealFormReport or None
        `None` when no attempt produced a Cartan subalgebra with a rational
        length spectrum, which includes the parametric case.

    Notes
    -----
    The centralizer of a generic element is a Cartan subalgebra `h` exactly
    when it is abelian and the Killing form restricted to it is
    nondegenerate, both of which are checked, so the rank reported here is
    exact rather than an approximation. The operator
    `sum_a ad(h_a) ad(k_a)`, with `k_a` the Killing dual basis of `h`, acts
    on each root space by the squared length of that root and vanishes on
    `h`, so its spectrum is the wanted data and stays rational over any
    real form.
    """
    if get_dgcv_category(alg) not in {"algebra", "subalgebra"}:
        raise Exception(
            "compute_root_length_profile expected to receive an algebra instance."
        ) from None
    dimension = alg.dimension
    if dimension < 3:
        return None
    ads = adjointRepresentation(alg, assume_Lie_algebra=assume_Lie_algebra)
    rng = random.Random(4177)
    for attempt in range(attempts):
        bound = 8 * (attempt + 1)
        coeffs = [rng.randint(-bound, bound) for _ in range(dimension)]
        data = _cartan_casimir(ads, coeffs, dimension)
        if data is None:
            continue
        rank, casimir = data
        lengths = _casimir_spectrum(casimir, rank, dimension)
        if lengths is None:
            continue
        candidates, certain = complex_type_from_root_lengths(dimension, rank, lengths)
        longest = max(value for value, _ in lengths)
        coxeter = 1 / longest
        return RealFormReport(
            lengths=lengths,
            dimension=dimension,
            rank=rank,
            dual_coxeter_number=int(coxeter) if coxeter.denominator == 1 else None,
            candidates=candidates,
            complex_type=candidates[0] if certain else None,
            certain=certain,
        )
    return None


def _ordered_union(first, second):
    out = list(first)
    seen = set(out)
    for item in second:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _fresh_solve_variables(count):
    pref = "v" + uuid.uuid4().hex[:8]
    return [symbol(f"{pref}{j}") for j in range(count)]


def _solve_weight_kwargs(
    heavy, surface_singularities, simplify_singularities, method="linsolve"
):
    kwargs = {"method": method, "simplify_result": False}
    if surface_singularities:
        kwargs["return_divisors"] = True
        kwargs["pass_to_symbolic_engine"] = False
        if heavy:
            kwargs["simplify_pivots"] = True
        else:
            kwargs["simplify_pivots"] = (
                simplify_singularities if simplify_singularities is not None else True
            )
    elif heavy:
        kwargs["simplify_pivots"] = True
    return kwargs


def _indep_check(
    elems,
    newE,
    return_decomp_coeffs=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    surface_singularities=False,
    simplify_singularities=None,
    _force_eqn_simiplify=False,
    force_heavy_solve=False,
):
    if not isinstance(elems, (list, tuple)) or len(elems) == 0:
        if return_decomp_coeffs:
            return (True, {}, []) if surface_singularities else (True, {})
        return (True, []) if surface_singularities else True
    if _scalar_is_zero(newE):
        if return_decomp_coeffs:
            return (False, [{}], []) if surface_singularities else (False, [{}])
        return (False, []) if surface_singularities else False
    count = len(elems)
    if _solve_variables is None or len(_solve_variables) < count:
        variables = _fresh_solve_variables(count)
    else:
        variables = _solve_variables[:count]
    eqn = zip_sum(variables, elems) - newE
    if _force_eqn_simiplify or force_heavy_solve:
        eqn = simplify(eqn)

    solve_kwargs = _solve_weight_kwargs(
        force_heavy_solve,
        surface_singularities,
        simplify_singularities,
        method=method,
    )
    if surface_singularities:
        sol, sing = solve_dgcv(
            eqn,
            variables,
            print_solve_stats=print_solve_stats,
            **solve_kwargs,
        )
    else:
        sol = solve_dgcv(
            eqn,
            variables,
            print_solve_stats=print_solve_stats,
            **solve_kwargs,
        )
    if len(sol) == 0:
        if return_decomp_coeffs:
            return (True, [], sing) if surface_singularities else (True, [])
        return (True, sing) if surface_singularities else True
    if surface_singularities:
        sing = [subs(v, sol[0]) for v in sing]
    if return_decomp_coeffs:
        s = sol[0]
        coeffs = {idx: s.get(var, 0) for idx, var in enumerate(variables)}
        var_set = set(variables)
        free_vars = set()
        for c in coeffs.values():
            free_vars |= get_free_symbols(c)
        free_vars &= var_set
        if len(free_vars) == 0:
            coeffs = [coeffs]
        else:
            zeroing = {u: 0 for u in free_vars}
            expanded = []
            for v in sorted(free_vars, key=str):
                rule = {**zeroing, v: 1}
                expanded.append({idx: c.subs(rule) for idx, c in coeffs.items()})
            coeffs = expanded
        return (False, coeffs, sing) if surface_singularities else (False, coeffs)
    return (False, sing) if surface_singularities else False


def _elem_scale(elem, surface_singularities=False):
    coeffs = getattr(elem, "coeffs", None)
    if isinstance(coeffs, (list, tuple)):
        for c in coeffs:
            if not _scalar_is_zero(c):
                try:
                    out = elem / c
                    if surface_singularities:
                        if get_free_symbols(c):
                            return out, [c]
                        else:
                            return out, []
                    return out
                except Exception:
                    return elem
    return elem


def _basis_builder(
    elems,
    newE,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if _scalar_is_zero(newE):
        return (list(elems), []) if surface_singularities else list(elems)
    if ALBS is True:
        newE = _elem_scale(newE, surface_singularities=surface_singularities)
        if surface_singularities:
            newE, sing = newE
    elif surface_singularities:
        sing = []
    if not isinstance(elems, (list, tuple)):
        raise TypeError(
            f"_basis_builder expects `elems` to be a list, recieved {elems} of type {type(elems)}"
        )
    if len(elems) == 0:
        out = ([newE], sing) if surface_singularities else [newE]
        return out
    check = _indep_check(
        elems,
        newE,
        print_solve_stats=print_solve_stats,
        method=method,
        return_decomp_coeffs=False,
        _solve_variables=_solve_variables,
        surface_singularities=surface_singularities,
        simplify_singularities=simplify_singularities,
        force_heavy_solve=force_heavy_solve,
    )
    if surface_singularities:
        check, sing2 = check
    if check is True:
        return (
            (list(elems) + [newE], _ordered_union(sing, sing2))
            if surface_singularities
            else list(elems) + [newE]
        )
    else:
        return (
            (list(elems), _ordered_union(sing, sing2))
            if surface_singularities
            else list(elems)
        )


def _extract_basis(
    element_list,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    return_indices=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if not isinstance(element_list, (list, tuple)):
        element_list = list(element_list)
    basis = []
    idxs = [] if return_indices else None
    sing = []
    if _solve_variables is None and len(element_list) > 0:
        _solve_variables = _fresh_solve_variables(len(element_list))
    for i, newE in enumerate(element_list):
        old_len = len(basis)
        basis = _basis_builder(
            basis,
            newE,
            ALBS=ALBS,
            print_solve_stats=print_solve_stats,
            method=method,
            _solve_variables=_solve_variables,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )

        if surface_singularities:
            basis, new_sing = basis
            sing = _ordered_union(sing, new_sing)

        if return_indices and len(basis) == old_len + 1:
            idxs.append(i)
    if surface_singularities:
        out = (basis, idxs, sing) if return_indices else (basis, sing)
    else:
        out = (basis, idxs) if return_indices else basis
    return out
