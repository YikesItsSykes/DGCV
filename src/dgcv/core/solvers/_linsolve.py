import random

from ..._aux._backends._calculus import diff
from ..._aux._backends._engine import _get_sage_module, engine_kind
from ..._aux._backends._symbolic_router import get_free_symbols, subs
from ..._aux._backends._types_and_constants import rational
from ..arrays import matrix_dgcv
from ._predicates import _as_zero_expr, _is_zero


def _dgcv_linsolve(
    processed_eqns,
    system_vars,
    *,
    return_divisors=False,
    validate=False,
    simplify_pivots=False,
    sample_if_overdetermined=False,
    allow_underdetermined_solution=True,
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
        return (
            (out, [e for e in processed_eqns if get_free_symbols(e)])
            if return_divisors
            else out
        )

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

            # optional quick linearity sanity check
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

        # attempt fast Sage linear algebra over QQ
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

            # parametric solution in dgcv formatting
            x = [0] * n
            for j in free_cols:
                x[j] = vars_[j]

            # solve pivots from RREF rows:
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
                simplify_steps=simplify_pivots,
            )

            if sol is None:
                out = []
                return (out, divs) if return_divisors else out

            out = [dict(zip(system_vars, sol))]
            return (out, divs) if return_divisors else out

    def _build_and_solve(eqns):
        rows = []
        rhs = []
        for eq in eqns:
            expr = _as_zero_expr(eq)
            coeffs = [diff(expr, v) for v in vars_]
            c0 = subs(expr, base0)
            if all(_is_zero(c) for c in coeffs) and not _is_zero(c0):
                if return_divisors:
                    return ([], [c0]) if get_free_symbols(c0) else ([], [])
                else:
                    return []

            if validate:
                vars_set = set(vars_)
                for c in coeffs:
                    if vars_set & get_free_symbols(c):
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
            simplify_steps=simplify_pivots,
        )

        if sol is None:
            return ([], divs) if return_divisors else []

        return (dict(zip(system_vars, sol)), divs)

    one = rational(1, 1)
    zero = rational(0, 1)
    base0 = {v: zero for v in vars_}

    # determine which equations to use
    m = len(processed_eqns)
    if sample_if_overdetermined is True:
        sample_if_overdetermined = 4
    use_sampling = (
        isinstance(sample_if_overdetermined, (int, float))
        and sample_if_overdetermined > 0
        and m > sample_if_overdetermined * n
    )

    if use_sampling:
        sample_size = max(n, int(sample_if_overdetermined * n))
        sampled_eqns = random.sample(processed_eqns, min(sample_size, m))
        sol_dict, divs = _build_and_solve(sampled_eqns)
        if (
            not allow_underdetermined_solution
            and isinstance(sol_dict, dict)
            and any(set(vars_) & get_free_symbols(v) for v in sol_dict.values())
        ):
            sol_dict, divs = _build_and_solve(processed_eqns)
    else:
        sol_dict, divs = _build_and_solve(processed_eqns)

    if not isinstance(sol_dict, dict):
        return ([], divs) if return_divisors else []

    out = [sol_dict]
    return (out, divs) if return_divisors else out
