from __future__ import annotations

from typing import Any, List, Sequence

from ..._aux._backends._symbolic_router import get_free_symbols, subs
from ..._aux._vmf._safeguards import get_dgcv_category
from ..._aux._vmf.vmf import vmf_lookup
from ..solvers import solve_dgcv
from ._solver_support import _local_unknowns, _make_parameters, _require_subcategory
from ._vmf_frames import _coordinate_basis_from_vmf, _infer_coordinate_space_from_objs


def get_coframe(
    VFList: Sequence[Any],
    *,
    coordinate_space: Sequence[Any] | None = None,
    return_parameters: bool = False,
    register_parameters: bool = False,
    parameters_label: str | None = None,
):
    VFList = list(VFList)
    if not VFList:
        return []

    _require_subcategory(VFList, {"vector_field"}, "get_coframe")

    if coordinate_space is None:
        coordinate_space = _infer_coordinate_space_from_objs(VFList)
    else:
        if not isinstance(coordinate_space, (list, tuple, set)):
            raise TypeError(
                "`get_coframe` coordinate_space must be a list/tuple/set if provided."
            )
        coordinate_space = list(coordinate_space)
        for a in coordinate_space:
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            if ds is None:
                raise TypeError(
                    "`get_coframe` requires coordinates to be registered in the dgcv VMF with differential objects. "
                    "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
                )

    dfBasis = _coordinate_basis_from_vmf(coordinate_space, coorType="df")
    n = len(dfBasis)
    m = len(VFList)

    if n == 0:
        raise TypeError("`get_coframe` could not obtain a DF basis from VMF.")

    def _as_scalar_eqns(x):
        if get_dgcv_category(x) == "tensor_field":
            return list(x.coeff_dict.values())
        return [x]

    eval_entries: List[List[Any]] = []
    for k in range(m):
        row = []
        vf = VFList[k]
        for df in dfBasis:
            val = df(vf)
            scalars = _as_scalar_eqns(val)
            if len(scalars) != 1:
                row.append(sum(scalars))
            else:
                row.append(scalars[0])
        eval_entries.append(row)

    unknowns = _local_unknowns(m * n)

    eqns: List[Any] = []
    for j in range(m):
        row_vars = unknowns[j * n : (j + 1) * n]
        for k in range(m):
            target = 1 if j == k else 0
            s = 0
            Ek = eval_entries[k]
            for u, e in zip(row_vars, Ek):
                s = s + u * e
            eqns.append(s - target)

    sols = solve_dgcv(eqns, unknowns, method="linsolve", simplify_result=False)
    if not sols:
        raise RuntimeError(
            "`get_coframe` could not solve for a coframe (system unsatisfiable)."
        )

    sol = sols[0]

    unknowns_set = set(unknowns)
    free_vars_set = set()
    for u in unknowns:
        free_vars_set |= set(get_free_symbols(sol.get(u, u))) & unknowns_set
    free = [u for u in unknowns if u in free_vars_set]

    if return_parameters and free:
        params = _make_parameters(
            len(free),
            register_parameters=register_parameters,
            parameters_label=parameters_label,
        )
        sub = dict(zip(free, params))
        sol_use = dict(sol)
        for v in free:
            sol_use[v] = v

        def _coeff(u):
            c = sol_use.get(u, u)
            return subs(c, sub)
    else:
        sol_use = dict(sol)
        for v in free:
            sol_use[v] = 0

        def _coeff(u):
            return sol_use.get(u, u)

    out: List[Any] = []
    for j in range(m):
        row_vars = unknowns[j * n : (j + 1) * n]
        omega_j = 0
        for u, df in zip(row_vars, dfBasis):
            omega_j = omega_j + _coeff(u) * df
        out.append(omega_j)

    return out
