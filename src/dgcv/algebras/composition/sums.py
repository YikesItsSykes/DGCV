from __future__ import annotations

from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from ...core.morphisms.morphisms import homomorphism
from ..linear_algebra import linear_representation


###!!! candidate for vs methods mixin promotion
def direct_sum(
    target_alg,
    other,
    grading=None,
    label=None,
    basis_labels=None,
    register_in_vmf=False,
    initial_basis_index=None,
    simplify_products_by_default=None,
    build_all_gradings=False,
):
    if get_dgcv_category(other) in {
        "algebra",
        "vectorspace",
        "subalgebra",
        "algebra_subspace",
        "vector_subspace",
    }:
        _markers = {
            "sum": True,
            "lockKey": retrieve_passkey(),
            "base_field": (
                "real"
                if target_alg.base_field == "real"
                and getattr(other, "base_field", "complex") == "real"
                else "complex"
            ),
        }
        if build_all_gradings is not True:
            grad1 = target_alg.grading[:1] or [[0] * target_alg.dimension]
            grad2 = other.grading[:1] or [[0] * other.dimension]
        else:
            grad1 = target_alg.grading or [[0] * target_alg.dimension]
            grad2 = other.grading or [[0] * other.dimension]
        builtG = []
        for gl1 in grad1:
            for gl2 in grad2:
                builtG.append(list(gl1) + list(gl2))
        if not isinstance(grading, (list, tuple)):
            grading = []
        if isinstance(grading, (list, tuple)):
            if all(isinstance(elem, (list, tuple)) for elem in grading):
                grading = [list(elem) for elem in grading] + builtG
            elif all(isinstance(elem, expr_numeric_types()) for elem in grading):
                grading = [list(grading)] + builtG
            elif grading is not None:
                dgcv_warning(
                    "The supplied grading data format is incompatible, and was ignored."
                )
                grading = builtG
            else:
                grading = builtG

        if label is None:
            label = f"{getattr(target_alg, 'label', 'algebra_instance')}_plus_{getattr(other, 'label', 'algebra_instance')}"
            _markers["_tex_label"] = (
                f"{target_alg._repr_latex_(raw=True, abbrev=True)}\\oplus {other._repr_latex_(raw=True, abbrev=True)}"
            )
        if basis_labels is None:
            basis_labels = [elem.__repr__() for elem in target_alg.basis] + [
                elem.__repr__() for elem in other.basis
            ]
            _markers["_tex_basis_labels"] = [
                elem._repr_latex_(raw=True) for elem in target_alg.basis
            ] + [elem._repr_latex_(raw=True) for elem in other.basis]

        if register_in_vmf is not True:
            _markers["registered"] = False

        return linear_representation(
            homomorphism(target_alg, other.endomorphism_algebra)
        ).semidirect_sum(
            grading=grading,
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
            _markers=_markers,
        )
    else:
        return NotImplemented


def _sa_direct_sum(
    target_alg,
    other,
    grading=None,
    label=None,
    basis_labels=None,
    register_in_vmf=False,
    initial_basis_index=None,
    simplify_products_by_default=None,
    build_all_gradings=False,
):
    if get_dgcv_category(other) in {
        "algebra",
        "vectorspace",
        "subalgebra",
        "algebra_subspace",
        "vector_subspace",
    }:
        numeric_types = expr_numeric_types()
        _markers = {"sum": True, "lockKey": retrieve_passkey()}
        if build_all_gradings is not True:
            grad1 = target_alg.grading[:1] or [[0] * target_alg.dimension]
            grad2 = other.grading[:1] or [[0] * other.dimension]
        else:
            grad1 = target_alg.grading or [[0] * target_alg.dimension]
            grad2 = other.grading or [[0] * other.dimension]
        builtG = []
        for gl1 in grad1:
            for gl2 in grad2:
                builtG.append(list(gl1) + list(gl2))
        if not isinstance(grading, (list, tuple)):
            grading = []
        if isinstance(grading, (list, tuple)):
            if all(isinstance(elem, (list, tuple)) for elem in grading):
                grading = [list(elem) for elem in grading] + builtG
            elif all(isinstance(elem, numeric_types) for elem in grading):
                grading = [list(grading)] + builtG
            elif grading is not None:
                dgcv_warning(
                    "The supplied grading data format is incompatible, and was ignored."
                )
                grading = builtG
            else:
                grading = builtG

        if label is None:
            label = f"{target_alg.label}_plus_{other.label}"
            _markers["_tex_label"] = (
                f"{target_alg._repr_latex_(raw=True, abbrev=True)}\\oplus {other._repr_latex_(raw=True, abbrev=True)}"
            )
        if basis_labels is None:
            basis_labels = [elem.__repr__() for elem in target_alg.basis] + [
                elem.__repr__() for elem in other.basis
            ]
            _markers["_tex_basis_labels"] = [
                elem._repr_latex_(raw=True) for elem in target_alg.basis
            ] + [elem._repr_latex_(raw=True) for elem in other.basis]

        return linear_representation(
            homomorphism(target_alg, other.endomorphism_algebra)
        ).semidirect_sum(
            grading=grading,
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
            _markers=_markers,
        )
    else:
        return NotImplemented
