from __future__ import annotations

from ..._aux._vmf._safeguards import retrieve_passkey
from ..algebras import algebra_class
from ..linear_algebra import _generate_gl_structure_data


class vector_space_endomorphisms(algebra_class):
    def __init__(target_alg, vector_space):
        target_alg.domain = vector_space
        target_alg._dgcv_categories = {"endomorphism_space"}
        structure_data, grading, _, matrix_representation, tensor_representation = (
            _generate_gl_structure_data(vector_space)
        )
        super().__init__(
            structure_data,
            grading=grading,
            format_sparse=False,
            process_matrix_rep=False,
            preferred_representation=None,
            simplify_products_by_default=None,
            assume_skew=False,
            matrix_representation=matrix_representation,
            tensor_representation=tensor_representation,
            _basis_labels_parent=None,
            _label=None,
            _basis_labels=None,
            _calledFromCreator=retrieve_passkey(),
            _callLock=retrieve_passkey(),
            _print_warning=None,
            _child_print_warning=None,
            _exclude_from_VMF=retrieve_passkey(),
            _markers={
                "endo": True,
                "endo_label": getattr(vector_space, "label", "algebra_instance"),
                "endo_tex": vector_space._repr_latex_(raw=True, abbrev=True),
            },
        )
