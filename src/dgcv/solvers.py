"""
package: dgcv - Differential Geometry with Complex Variables
module: solvers

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import random

from ._config import get_dgcv_settings_registry
from .arrays import matrix_dgcv
from .backends._engine import _get_sage_module, engine_kind, engine_module
from .backends._symbolic_router import get_free_symbols, simplify, subs
from .backends._types_and_constants import (
    expr_numeric_types,
    expr_types,
    is_atomic,
    one,
    rational,
    zero,
)
from .eds.eds import (
    _equation_formatting,
    _sympy_to_abstract_ZF,
    abstract_ZF,
    zeroFormAtom,
)
from .eds.eds_representations import DF_representation

__all__ = ["solve_dgcv"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def normalize_equations_and_vars(eqns, vars_to_solve):
    if isinstance(eqns, DF_representation):
        eqns = eqns.flatten()
    if not isinstance(eqns, (list, tuple)):
        eqns = [eqns]

    if vars_to_solve is None:
        vars_to_solve = set()
        for eqn in eqns:
            try:
                vars_to_solve |= set(get_free_symbols(eqn))
            except Exception:
                pass

    if isinstance(vars_to_solve, set):
        vars_to_solve = list(vars_to_solve)
    if not isinstance(vars_to_solve, (list, tuple)):
        vars_to_solve = [vars_to_solve]
    return eqns, vars_to_solve


def _rows_from_engine_matrix(A):
    nrows = getattr(A, "nrows", None)
    ncols = getattr(A, "ncols", None)
    if callable(nrows) and callable(ncols):
        r = int(nrows())
        c = int(ncols())
        return [[A[i, j] for j in range(c)] for i in range(r)]

    shape = getattr(A, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        r, c = shape
        return [[A[i, j] for j in range(c)] for i in range(r)]

    rows = getattr(A, "tolist", None)
    if callable(rows):
        v = A.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v

    raise TypeError("Unsupported matrix type returned by engine")


def _col_from_engine_vector(b):
    nrows = getattr(b, "nrows", None)
    ncols = getattr(b, "ncols", None)
    if callable(nrows) and callable(ncols):
        r = int(nrows())
        c = int(ncols())
        if c != 1:
            raise ValueError("Expected column vector")
        return [b[i, 0] for i in range(r)]

    shape = getattr(b, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        r, c = shape
        if c != 1:
            raise ValueError("Expected column vector")
        return [b[i, 0] for i in range(r)]

    if isinstance(b, (list, tuple)):
        return list(b)

    raise TypeError("Unsupported vector type returned by engine")


def _rref_solve_unique(A_rows, b_col, *, simplify_steps=False):
    m = len(A_rows)
    if m == 0:
        return []

    n = len(A_rows[0])
    if len(b_col) != m:
        raise ValueError("Row mismatch between A and b")

    aug = [list(A_rows[i]) + [b_col[i]] for i in range(m)]
    row = 0
    pivots = []

    for col in range(n):
        if row >= m:
            break

        pivot_row = None
        for r in range(row, m):
            if not _is_zero(aug[r][col]):
                pivot_row = r
                break

        if pivot_row is None:
            continue

        if pivot_row != row:
            aug[row], aug[pivot_row] = aug[pivot_row], aug[row]

        piv = aug[row][col]
        inv_piv = one / piv
        aug[row] = [inv_piv * v for v in aug[row]]
        if simplify_steps:
            aug[row] = [simplify(v) for v in aug[row]]

        for r in range(m):
            if r == row:
                continue
            factor = aug[r][col]
            if _is_zero(factor):
                continue
            aug[r] = [aug[r][c] - factor * aug[row][c] for c in range(n + 1)]
            if simplify_steps:
                aug[r] = [simplify(v) for v in aug[r]]

        pivots.append(col)
        row += 1

    for r in range(m):
        if all(_is_zero(aug[r][c]) for c in range(n)) and not _is_zero(aug[r][n]):
            raise ValueError("Inconsistent linear system")

    if len(pivots) != n:
        raise ValueError("Singular or underdetermined system")

    x = [zero for _ in range(n)]
    for r, col in enumerate(pivots):
        x[col] = aug[r][n]
        if simplify_steps:
            x[col] = simplify(x[col])
    return x


def _sage_solve_to_dicts(sols, vars_):
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
                if is_atomic(R) and R not in vars_set:
                    replacements[R] = L
        if replacements:
            d = {k: subs(v, replacements) for k, v in d.items()}

        if ok:
            out.append(d)

    return out


def _is_zero(x):
    if isinstance(x, expr_numeric_types()) and not isinstance(x, bool):
        return x == 0
    iz = getattr(x, "is_zero", None)
    if iz is True:
        return True
    if callable(iz):
        try:
            return bool(iz())
        except Exception:
            pass
    try:
        return simplify(x) == 0
    except Exception:
        return x == 0


def _as_zero_expr(eq):
    lhs = getattr(eq, "lhs", None)
    rhs = getattr(eq, "rhs", None)
    if callable(lhs) and callable(rhs):
        try:
            return lhs() - rhs()
        except Exception:
            pass
    if lhs is not None and rhs is not None and not callable(lhs) and not callable(rhs):
        return lhs - rhs
    return eq


def _dgcv_linsolve(
    processed_eqns, system_vars, *, return_divisors=False, validate=False
):
    if not processed_eqns:
        out = [{v: 0 for v in system_vars}]
        return (out, []) if return_divisors else out

    vars_ = tuple(system_vars)
    n = len(vars_)

    if n == 0:
        ok = True
        for e in processed_eqns:
            try:
                ok = _as_zero_expr(e) == 0
            except Exception:
                ok = False
            if not ok:
                break
        out = [dict()] if ok else []
        return (out, []) if return_divisors else out

    if engine_kind() == "sage":
        sage = _get_sage_module()

        one = rational(1, 1)
        zero = rational(0, 1)

        base0 = {v: zero for v in vars_}

        rows = []
        rhs = []

        for eq in processed_eqns:
            expr = _as_zero_expr(eq)

            try:
                c0 = subs(expr, base0)
            except Exception:
                out = []
                return (out, []) if return_divisors else out

            coeffs = []
            for v in vars_:
                di = dict(base0)
                di[v] = one
                try:
                    vi = subs(expr, di)
                except Exception:
                    out = []
                    return (out, []) if return_divisors else out
                coeffs.append(vi - c0)

            # optional quick linearity sanity check (keep your behavior)
            if validate:
                try:
                    for _ in range(2):
                        test = {}
                        vals = []
                        for v in vars_:
                            q = rational(random.randint(2, 9), random.randint(2, 9))
                            test[v] = q
                            vals.append(q)
                        lhs_val = subs(expr, test)
                        rhs_val = c0
                        for a, q in zip(coeffs, vals):
                            rhs_val = rhs_val + a * q
                        if lhs_val != rhs_val:
                            out = []
                            return (out, []) if return_divisors else out
                except Exception:
                    out = []
                    return (out, []) if return_divisors else out

            rows.append(coeffs)
            rhs.append(-c0)

        # attempt fast Sage linear algebra over QQ when possible
        try:
            QQ = sage.QQ
            A_QQ = sage.matrix(QQ, rows)
            b_QQ = sage.vector(QQ, rhs)
            m = A_QQ.nrows()
            n = A_QQ.ncols()

            # Work in augmented RREF to build a parametric solution
            Aug = A_QQ.augment(b_QQ.column())
            R = Aug.rref()

            # Detect inconsistency: [0 ... 0 | nonzero]
            for i in range(m):
                all0 = True
                for j in range(n):
                    if R[i, j] != 0:
                        all0 = False
                        break
                if all0 and R[i, n] != 0:
                    out = []
                    return (out, []) if return_divisors else out

            # Identify pivot columns
            pivcol_to_row = {}
            pivot_cols = []
            for i in range(m):
                piv = None
                for j in range(n):
                    if R[i, j] != 0:
                        piv = j
                        break
                if piv is not None:
                    pivcol_to_row[piv] = i
                    pivot_cols.append(piv)

            free_cols = [j for j in range(n) if j not in pivcol_to_row]

            # Parametric solution: dgcv formatting
            x = [0] * n
            for j in free_cols:
                x[j] = vars_[j]

            # Solve pivots from RREF rows:
            # pivot_var + sum_{free} R[row,free]*free_var = R[row, rhs]
            for pc in sorted(pivcol_to_row):
                i = pivcol_to_row[pc]
                val = R[i, n]
                for fc in free_cols:
                    if R[i, fc] != 0:
                        val = val - R[i, fc] * x[fc]
                x[pc] = val

            out = [dict(zip(system_vars, x))]
            return (out, []) if return_divisors else out

        except Exception:
            # Fallback: dgcv elimination with formal inverses
            A = matrix_dgcv(rows)
            b = matrix_dgcv([[v] for v in rhs])

            sol, divs = A.solve_right(
                b,
                return_divisors=True,
                allow_formal_inverse=True,
                parametric_vars=vars_,
            )

            if sol is None:
                out = []
                return (out, divs) if return_divisors else out

            out = [dict(zip(system_vars, sol))]
            return (out, divs) if return_divisors else out

    one = rational(1, 1)
    zero = rational(0, 1)

    base0 = {v: zero for v in vars_}

    rows = []
    rhs = []

    for eq in processed_eqns:
        expr = _as_zero_expr(eq)

        try:
            c0 = subs(expr, base0)
        except Exception:
            return ([], []) if return_divisors else []

        coeffs = []
        for v in vars_:
            di = dict(base0)
            di[v] = one
            try:
                vi = subs(expr, di)
            except Exception:
                return ([], []) if return_divisors else []
            coeffs.append(vi - c0)

        try:
            for _ in range(2):
                test = {}
                vals = []
                for v in vars_:
                    q = rational(random.randint(2, 9), random.randint(2, 9))
                    test[v] = q
                    vals.append(q)
                lhs_val = subs(expr, test)
                rhs_val = c0
                for a, q in zip(coeffs, vals):
                    rhs_val = rhs_val + a * q
                if lhs_val != rhs_val:
                    return ([], []) if return_divisors else []
        except Exception:
            return ([], []) if return_divisors else []

        rows.append(coeffs)
        rhs.append(-c0)

    A = matrix_dgcv(rows)
    b = matrix_dgcv([[v] for v in rhs])

    sol, divs = A.solve_right(
        b,
        return_divisors=True,
        allow_formal_inverse=True,
        parametric_vars=vars_,
    )

    if sol is None:
        return ([], divs) if return_divisors else []

    out = [dict(zip(system_vars, sol))]
    return (out, divs) if return_divisors else out


def solve_dgcv(
    eqns,
    vars_to_solve=None,
    verbose=False,
    method="auto",
    simplify_result=True,
    print_solve_stats=False,
    return_divisors=False,
    pass_to_symbolic_engine=None,
):
    if pass_to_symbolic_engine is None:
        pass_to_symbolic_engine = (
            get_dgcv_settings_registry().get(
                "pass_solve_requests_to_symbolic_engine", False
            )
            is True
        )

    import time

    t0 = time.perf_counter()

    def _log(*a):
        if print_solve_stats:
            print("[solve_dgcv]", *a)

    if isinstance(eqns, (list, tuple)) and len(eqns) == 0:
        if isinstance(vars_to_solve, (list, tuple)):
            out = [{var: var for var in vars_to_solve}]
        elif isinstance(vars_to_solve, expr_numeric_types()):
            out = [{vars_to_solve: vars_to_solve}]
        else:
            out = [dict()]
        return (out, []) if return_divisors else out

    eqns, vars_to_solve = normalize_equations_and_vars(eqns, vars_to_solve)
    processed_eqns, system_vars, extra_vars, variables_dict = _equations_preprocessing(
        eqns, vars_to_solve
    )

    def _simplify(x):
        if not simplify_result:
            return x
        try:
            return simplify(x)
        except Exception:
            return x

    def _expr_reformatting(expr):
        if isinstance(expr, expr_numeric_types()) or not hasattr(expr, "subs"):
            return expr

        dgcv_var_dict = {v[1][0]: v[0] for _, v in variables_dict.items()}

        if not isinstance(expr, expr_types()) or isinstance(expr, zeroFormAtom):
            try:
                return expr.subs(dgcv_var_dict)
            except Exception:
                return expr

        regular_var_dict = {k: v for k, v in dgcv_var_dict.items() if is_atomic(k)}

        try:
            bad = not all(
                isinstance(v, expr_numeric_types()) or isinstance(v, expr_types())
                for v in regular_var_dict.values()
            )
        except Exception:
            bad = True

        if bad:
            return abstract_ZF(_sympy_to_abstract_ZF(expr, regular_var_dict))

        try:
            return expr.subs(regular_var_dict)
        except Exception:
            return expr

    def _extract_reformatting(var):
        s = str(var)
        return variables_dict[s][0] if s in variables_dict else var

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

    mod = engine_module()

    def _engine_linsolve(eqns_, vars_):
        if mod is None:
            return []
        fn = getattr(mod, "linsolve", None)
        if not callable(fn):
            return []
        try:
            sols = fn(eqns_, tuple(vars_))
        except Exception:
            return []
        return _linsolve_to_dicts(sols, tuple(vars_))

    def _engine_solve(eqns_, vars_):
        if mod is None:
            return []
        fn = getattr(mod, "solve", None)
        if not callable(fn):
            return []
        try:
            sols = fn(eqns_, vars_, dict=True)
        except TypeError:
            try:
                sols = fn(eqns_, vars_)
            except Exception:
                return []
        except Exception:
            return []
        return _engine_solve_to_dicts(sols, vars_)

    _log(
        f"engine={(engine_kind() or 'unknown')} method={method} #eqns={len(processed_eqns)} #vars={len(system_vars)}"
    )

    preformatted_solutions = []
    divisors = []

    def _run_custom_linsolve():
        nonlocal preformatted_solutions, divisors
        try:
            if return_divisors:
                preformatted_solutions, d = _dgcv_linsolve(
                    processed_eqns, system_vars, return_divisors=True
                )
                divisors = d or []
            else:
                preformatted_solutions = _dgcv_linsolve(
                    processed_eqns, system_vars, return_divisors=False
                )
        except Exception:
            preformatted_solutions = []

    def _run_engine_linsolve():
        nonlocal preformatted_solutions
        try:
            preformatted_solutions = _engine_linsolve(processed_eqns, system_vars)
        except Exception:
            preformatted_solutions = []

    def _run_engine_solve():
        nonlocal preformatted_solutions
        try:
            if engine_kind() == "sage":
                preformatted_solutions = _engine_solve(eqns, vars_to_solve)
            else:
                preformatted_solutions = _engine_solve(processed_eqns, system_vars)
        except Exception:
            preformatted_solutions = []

    if method not in ("auto", "linsolve", "solve"):
        raise ValueError(
            f"Unknown method '{method}'. Use 'auto', 'linsolve', or 'solve'."
        )

    if method == "solve":
        _run_engine_solve()
    elif method == "linsolve":
        if not pass_to_symbolic_engine:
            _run_custom_linsolve()
        if not preformatted_solutions:
            _run_engine_linsolve()
        if not preformatted_solutions:
            _run_engine_solve()
    else:
        if not pass_to_symbolic_engine:
            _run_custom_linsolve()
        if not preformatted_solutions:
            _run_engine_linsolve()
        if not preformatted_solutions:
            _run_engine_solve()

    solutions_formatted = [
        {
            _extract_reformatting(var): _expr_reformatting(_simplify(expr))
            for var, expr in (solution or {}).items()
        }
        for solution in (preformatted_solutions or [])
        if isinstance(solution, dict)
    ]

    _log(
        f"solutions={len(solutions_formatted)} elapsed_s={time.perf_counter() - t0:.6f}"
    )

    if return_divisors:
        if verbose:
            return solutions_formatted, system_vars, extra_vars, divisors
        return solutions_formatted, divisors

    return (
        (solutions_formatted, system_vars, extra_vars)
        if verbose
        else solutions_formatted
    )


def _equations_preprocessing(eqns: tuple | list, vars: tuple | list):
    processed_eqns = []
    variables_dict = dict()
    for eqn in eqns:
        eqn_formatted, new_var_dict = _equation_formatting(eqn, variables_dict)
        processed_eqns += eqn_formatted
        variables_dict = variables_dict | new_var_dict

    subbedValues = {variables_dict[k][0]: variables_dict[k][1] for k in variables_dict}
    pre_system_vars = [
        subbedValues[var] if var in subbedValues else var for var in vars
    ]

    system_vars = []
    extra_vars = []
    for var in pre_system_vars:
        if isinstance(var, (list, tuple)) and len(var) == 1:
            var = var[0]
        if is_atomic(var):
            system_vars += [var]
        else:
            extra_vars += [var]
    return processed_eqns, system_vars, extra_vars, variables_dict
