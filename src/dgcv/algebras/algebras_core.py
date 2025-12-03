############## dependencies
import math
import numbers
import random
import re
import warnings
from collections.abc import Mapping
from functools import lru_cache
from html import escape as _esc

import sympy as sp

from .._config import (
    _cached_caller_globals,
    dgcv_exception_note,
    from_vsr,
    get_dgcv_settings_registry,
    get_vs_registry,
    greek_letters,
    latex_in_html,
)
from .._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
    retrieve_public_key,
    unique_label,
)
from .._tables import build_matrix_table, panel_view
from ..backends._caches import _get_expr_num_types, _get_fast_scalar_types
from ..dgcv_core import variableProcedure
from ..morphisms import homomorphism
from ..solvers import simplify_dgcv, solve_dgcv
from ..styles import get_style
from ..tensors import mergeVS, tensorProduct
from ..vmf import clearVar, listVar
from .algebras_aux import _validate_structure_data


############## Algebras
# finite dimensional algebra class
class algebra_class:
    def __init__(
        self,
        structure_data,
        grading=None,
        format_sparse=False,
        process_matrix_rep=False,
        preferred_representation=None,
        simplify_products_by_default=None,
        assume_skew=False,
        matrix_representation = None,
        tensor_representation = None,
        _basis_labels_parent=None,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
        _callLock=None,
        _print_warning=None,
        _child_print_warning=None,
        _exclude_from_VMF=None,
        _markers={}
    ):
        if isinstance(structure_data,numbers.Integral):
            structure_data=(((0,)*structure_data,)*structure_data,)*structure_data
        if _calledFromCreator == retrieve_passkey():
            validated_structure_data = structure_data
            params = _markers.get('parameters',set())
        else:
            try:
                vsd = _validate_structure_data(
                    structure_data,
                    process_matrix_rep=process_matrix_rep,
                    assume_skew=assume_skew,
                    assume_Lie_alg=False,
                    basis_order_for_supplied_str_eqns=False,
                )
                if process_matrix_rep is True:
                    if matrix_representation is not None:
                        warnings.warn('The `algebra_class` initializer disregarded the optional parameter value given for `matrix_representation` because `process_matrix_rep` was set to `True`, which forces automated computation of the representation.')
                    validated_structure_data, matrix_representation, params = vsd[0][0],vsd[0][1],vsd[0][2]
                else:
                    validated_structure_data,params=vsd

            except dgcv_exception_note as e:
                raise SystemExit(e)
        # validated_structure_data = tuple(map(tuple, validated_structure_data))
        self.structureData = tuple(tuple(tuple(inner) for inner in middle) for middle in validated_structure_data)
        self._parameters=params
        self._tex_label=None
        self._tex_basis_labels=None
        self._educed_properties = dict()
        if _calledFromCreator == retrieve_passkey():
            if isinstance(_markers.get('_educed_properties',None),dict):
                self._educed_properties=_markers.get('_educed_properties',dict())
            if _markers.get('endo',False):
                if _label is None:
                    self.label = f"gl_{_markers.get('endo_label','')}"
                    self._tex_label = f"\\mathfrak{{gl}}\\left({_markers.get('endo_tex','')}\\right)"
                    self.basis_labels = [f"{self.label}{i+1}" for i in range(len(self.structureData))]
                    self._tex_basis_labels = [f"{self._tex_label}_{{{i+1}}}" for i in range(len(self.structureData))]
                else:
                    self.label = _label          
                    self.basis_labels = _basis_labels
            elif _markers.get('sum',False):
                # If not registered, pick collision-free labels here; otherwise trust provided labels
                if _markers.get('registered', None) is False:
                    incoming_tex_label = _markers.get('_tex_label', None)
                    if incoming_tex_label is None:
                        self.label = unique_label(_label)
                        self._tex_label = None
                    else:
                        self.label, self._tex_label = unique_label(_label, tex_label=incoming_tex_label)

                    incoming_basis = list(_basis_labels or [])
                    incoming_tex_basis = list(_markers.get('_tex_basis_labels', []) or [])
                    have_tex_basis = len(incoming_basis) > 0 and len(incoming_tex_basis) == len(incoming_basis)

                    new_basis = []
                    new_tex_basis = [] if have_tex_basis else None
                    batch_protected = set()
                    if isinstance(self.label, str):
                        batch_protected.add(self.label)
                    for idx, base_lbl in enumerate(incoming_basis):
                        candidate = base_lbl
                        if have_tex_basis:
                            base_tex = incoming_tex_basis[idx]
                            final_lbl, final_tex = unique_label(
                                candidate, tex_label=base_tex, protected=batch_protected
                            )
                            new_basis.append(final_lbl)
                            new_tex_basis.append(final_tex)
                            batch_protected.add(final_lbl)
                        else:
                            final_lbl = unique_label(candidate, protected=batch_protected)
                            new_basis.append(final_lbl)
                            batch_protected.add(final_lbl)

                    self.basis_labels = new_basis
                    if have_tex_basis:
                        self._tex_basis_labels = new_tex_basis
                    elif self._tex_label is not None:
                        self._tex_basis_labels = [
                            f"{self._tex_label}_{{{i+1}}}" for i in range(len(self.structureData))
                        ]
                else:
                    self.label = _label
                    self.basis_labels = _basis_labels
                    if not self.basis_labels:  
                        base = self.label if isinstance(self.label, str) and self.label else "_e"
                        self.basis_labels = [f"{base}{i+1}" for i in range(len(self.structureData))]
                    if _markers.get('_tex_label', None) is not None:
                        self._tex_label = _markers['_tex_label']
                    if _markers.get('_tex_basis_labels', None) is not None:
                        self._tex_basis_labels = _markers['_tex_basis_labels']
                    elif self._tex_label is not None and self._tex_basis_labels is None:
                        self._tex_basis_labels = [
                            f"{self._tex_label}_{{{i+1}}}" for i in range(len(self.structureData))
                        ]
            elif _markers.get('prod', False):
                if _markers.get('registered', None) is False:
                    incoming_tex_label = _markers.get('_tex_label', None)
                    if incoming_tex_label is None:
                        self.label = unique_label(_label)
                        self._tex_label = None
                    else:
                        self.label, self._tex_label = unique_label(_label, tex_label=incoming_tex_label)

                    incoming_basis = list(_basis_labels or [])
                    incoming_tex_basis = list(_markers.get('_tex_basis_labels', []) or [])
                    have_tex_basis = len(incoming_basis) > 0 and len(incoming_tex_basis) == len(incoming_basis)

                    new_basis = []
                    new_tex_basis = [] if have_tex_basis else None
                    batch_protected = set()
                    if isinstance(self.label, str):
                        batch_protected.add(self.label)
                    for idx, base_lbl in enumerate(incoming_basis):
                        candidate = base_lbl
                        if have_tex_basis:
                            base_tex = incoming_tex_basis[idx]
                            final_lbl, final_tex = unique_label(
                                candidate, tex_label=base_tex, protected=batch_protected
                            )
                            new_basis.append(final_lbl)
                            new_tex_basis.append(final_tex)
                            batch_protected.add(final_lbl)
                        else:
                            final_lbl = unique_label(candidate, protected=batch_protected)
                            new_basis.append(final_lbl)
                            batch_protected.add(final_lbl)

                    self.basis_labels = new_basis
                    if have_tex_basis:
                        self._tex_basis_labels = new_tex_basis
                    elif self._tex_label is not None:
                        self._tex_basis_labels = [
                            f"{self._tex_label}_{{{i+1}}}" for i in range(len(self.structureData))
                        ]
                else:
                    self.label = _label
                    self.basis_labels = _basis_labels
                    if _markers.get('_tex_label', None) is not None:
                        self._tex_label = _markers['_tex_label']
                    if _markers.get('_tex_basis_labels', None) is not None:
                        self._tex_basis_labels = _markers['_tex_basis_labels']
                    elif self._tex_label is not None:
                        self._tex_basis_labels = [
                            f"{self._tex_label}_{{{i+1}}}" for i in range(len(self.structureData))
                        ]

            else:
                self.label = _label
                self.basis_labels = _basis_labels
            self._registered = True
        else:
            self.label = "Alg_" + create_key()
            if _basis_labels_parent is True:
                self.basis_labels = [f"{self.label}{i+1}" for i in range(len(self.structureData))]
            else:
                self.basis_labels = [f"_e{i+1}" for i in range(len(self.structureData))]
            self._registered = False
        self._basis_labels_parent = _basis_labels_parent
        self._callLock = _callLock
        self._print_warning = _print_warning
        self._child_print_warning = _child_print_warning
        self._exclude_from_VMF = _exclude_from_VMF
        self.is_sparse = format_sparse
        self.dimension = len(self.structureData)
        self.structureDataDict=_lazy_SD(self.structureData)
        self._built_from_matrices = process_matrix_rep
        self.simplify_products_by_default = simplify_products_by_default
        self.semidirect_decomposition = _markers.get('semidirect_decomposition',None)
        self.tensor_decomposition = _markers.get('tensor_decomposition', None)
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra"

        def validate_and_adjust_grading_vector(vector, dimension):
            vector = list(vector)
            if len(vector) < dimension:
                warnings.warn(
                    f"Grading vector is shorter than the dimension ({len(vector)} < {dimension}). "
                    f"Padding with zeros to match the dimension.",
                    UserWarning,
                )
                vector += [0] * (dimension - len(vector))
            elif len(vector) > dimension:
                warnings.warn(
                    f"Grading vector is longer than the dimension ({len(vector)} > {dimension}). "
                    f"Truncating to match the dimension.",
                    UserWarning,
                )
                vector = vector[:dimension]

            for i, component in enumerate(vector):
                if not isinstance(component, _get_expr_num_types()):
                    raise ValueError(
                        f"Invalid component in grading vector at index {i}: {component}. "
                        f"Expected int, float, or sympy.Expr."
                    ) from None

            return tuple(vector)

        if grading is None:
            self.grading = [tuple([0] * self.dimension)]
        else:
            if isinstance(grading, (list, tuple)) and all(
                isinstance(g, (list, tuple)) for g in grading
            ):
                self.grading = [
                    validate_and_adjust_grading_vector(vector, self.dimension)
                    for vector in grading
                ]
            else:
                self.grading = [
                    validate_and_adjust_grading_vector(grading, self.dimension)
                ]

        self._gradingNumber = len(self.grading)

        for val,lab in zip([matrix_representation,preferred_representation,tensor_representation],['matrix_representation','preferred_representation','tensor_representation']):
            if val is not None and (
                not isinstance(val, (list, tuple))
                or len(val) != self.dimension
            ):
                raise TypeError(f"unsupported format for {lab}.") from None
        if preferred_representation is not None and all(
            isinstance(elem, sp.Matrix) for elem in preferred_representation
        ):
            self._preferred_rep_type = "matrix"
            self._preferred_representation = preferred_representation
        elif preferred_representation is not None and all(
            isinstance(elem, tensorProduct) for elem in preferred_representation
        ):
            self._preferred_rep_type = "tensor"
            self._preferred_representation = preferred_representation
        elif preferred_representation is not None and all(
            isinstance(elem, (list, tuple)) for elem in preferred_representation
        ):
            self._preferred_rep_type = "matrix"
            self._preferred_representation = [
                sp.Matrix(elem) for elem in preferred_representation
            ]
        elif preferred_representation is not None:
            raise TypeError("unsupported format for `preferred_representation`.") from None
        else:
            self._preferred_rep_type=None
            self._preferred_representation=None

        if matrix_representation  is not None and all(
            isinstance(elem, sp.Matrix) for elem in matrix_representation
        ):
            self._mat_rep = matrix_representation
        elif matrix_representation is not None and all(
            isinstance(elem, (list, tuple)) for elem in matrix_representation
        ):
            self._mat_rep = [sp.Matrix(elem) for elem in matrix_representation]
        elif matrix_representation is not None:
            raise TypeError("unsupported format for `matrix_representation`.") from None
        else:
            self._mat_rep = None

        if tensor_representation is not None and all(
            isinstance(elem, tensorProduct) for elem in tensor_representation
        ):
            self._tensor_rep = tensor_representation
        elif tensor_representation is not None:
            raise TypeError("unsupported format for `tensor_representation`.") from None
        else:
            self._tensor_rep=None

        vsr=get_vs_registry()
        self.dgcv_vs_id = len(vsr)
        vsr.append(self)

        self.basis = tuple(
            [
                algebra_element_class(
                    self,
                    [1 if i == j else 0 for j in range(self.dimension)],
                    1,
                    format_sparse=format_sparse,
                )
                for i in range(self.dimension)
            ]
        )
        # immutables
        self._basis_labels = tuple(_basis_labels) if _basis_labels else None
        self._grading = tuple(map(tuple, self.grading))
        # Caches
        self._skew_symmetric_cache = None
        self._jacobi_identity_cache = None
        self._lie_algebra_cache = None
        self._is_semisimple_cache = None
        self._is_simple_cache = None
        self._is_nilpotent_cache = None
        self._is_abelian_cache = None
        self._is_solvable_cache = None
        self._rank_approximation = None
        self._center_cache = None
        self._lower_central_series_cache = None
        self._derived_series_cache = None
        self._grading_compatible = None
        self._grading_report = None
        self._killing_form = None
        self._derived_subalg_cache = None
        self._radical_cache = None
        self._Levi_deco_cache = None
        self._graded_components = None
        self._endomorphisms = None
        self._coproduct = {elem:None for elem in self.basis}

    def _class_builder(self, coeffs, valence, format_sparse=False):
        return algebra_element_class(self, coeffs, valence, format_sparse=format_sparse)

    @property
    def preferred_representation(self):
        if self._preferred_representation is None:
            if self._mat_rep is not None:
                warnings.warn(
                    "A preferred representation format for this algebra was never set up, but a cached matrix representation was found and has been set as the default for `preferred_representation`."
                )
                self._preferred_rep_type = "matrix"
                self._preferred_representation = self._mat_rep
            elif  self._tensor_rep is not None:
                warnings.warn(
                    "A preferred representation format for this algebra was never set up, but a cached tensor product representation was found and has been set as the default for `preferred_representation`."
                )
                self._preferred_rep_type = "tensor"
                self._preferred_representation = self._tensor_rep
            else:
                warnings.warn(
                    "A preferred representation format for this algebra was not specified, so it has been set to its adjoint representation."
                )
                self._preferred_rep_type = "matrix"
                self._preferred_representation = adjointRepresentation(self)
        return self._preferred_representation
    @property
    def tensor_representation(self):
        return self._tensor_rep
    @property
    def matrix_representation(self):
        return self._mat_rep
    @property
    def ambient(self):
        return self
    @property
    def endomorphism_algebra(self):
        if self._endomorphisms is None:
            self._endomorphisms=vector_space_endomorphisms(self)
        return self._endomorphisms
    @property
    def zero_element(self):
        return algebra_element_class(self,(0,)*self.dimension,1)

    def update_grading(self,new_weight_vectors_list,replace_instead_of_add=False):
        if isinstance(new_weight_vectors_list,(list,tuple)):
            if all(isinstance(elem,(list,tuple)) for elem in new_weight_vectors_list):
                   if replace_instead_of_add is True:
                       self.grading = [tuple(elem) for elem in new_weight_vectors_list]
                   else:
                       grad=list(self.grading)+[tuple(elem) for elem in new_weight_vectors_list]
                       self.grading=grad
            else:
                raise TypeError(f'update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}')
        else:
            raise TypeError(f'update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}')

    def contains(self, items, return_basis_coeffs=False,strict_types=False):
        if isinstance(items, (list, tuple)):
            return [self.contains(item,return_basis_coeffs=return_basis_coeffs) for item in items]

        if strict_types is False and items==0:
            if return_basis_coeffs is True:
                return [0]*self.dimension
            return True
        if strict_types is False and get_dgcv_category(items)=='tensorProduct':
            if next(iter(items._vs_spring))==self.dgcv_vs_id and len(items.vector_spaces)==1:
                k,v=next(iter(items.coeff_dict.items()))
                if len(k)==3:
                    ne = v*(from_vsr(k[2]).basis[k[0]])
                    if k[1]==0:
                        ne=ne.dual()
                    return self.contains(ne)
        if get_dgcv_category(items)=='algebra_element' and items.dgcv_vs_id==self.dgcv_vs_id:
            if return_basis_coeffs:
                return list(items.coeffs)
            else:
                return True
        return False

    def _set_product_protocol(self):
        if self.simplify_products_by_default is None:
            if any(not isinstance(j, _get_fast_scalar_types()) for j in self.structureDataDict.values()):
                self.simplify_products_by_default = True
            else:
                self.simplify_products_by_default = False
        elif self.simplify_products_by_default is not True:
            self.simplify_products_by_default = False

    def __eq__(self, other):
        if not isinstance(other, algebra_class):
            return NotImplemented
        return (self.dgcv_vs_id == other.dgcv_vs_id)

    def __hash__(self):
        return hash(self.dgcv_vs_id)

    def __contains__(self, item):
        return item in self.basis

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            return self.basis[indices]
        elif isinstance(indices, list):
            if len(indices) == 1:
                return self.basis[indices[0]]
            elif isinstance(indices, list) and len(indices) == 2:
                return self.structureData[indices[0]][indices[1]]
            elif isinstance(indices, list) and len(indices) == 3:
                return self.structureData[indices[0]][indices[1]][indices[2]]
        else:
            raise TypeError(
                f"To access an algebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}"
            ) from None

    def __repr__(self):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method.",
                    UserWarning,
                )
        return (
            f"algebra_class(dim={self.dimension}, grading={self.grading}, "
            f"label={self.label}, basis_labels={self.basis_labels}, "
            f"struct_data={self.structureData})"
        )

    def _structure_data_summary(self):
        if self.dimension <= 3:
            return self.structureData
        return (
            "Structure data is large. Access the `structureData` attribute for details."
        )

    def __str__(self):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method.",
                    UserWarning,
                )

        formatted_label = self.label if self.label else "Unnamed Algebra"
        formatted_basis_labels = (
            ", ".join(list(self.basis_labels))
            if self.basis_labels
            else "No basis labels assigned"
        )
        return (
            f"Algebra: {formatted_label}\n"
            f"Dimension: {self.dimension}\n"
            f"Grading: {self.grading}\n"
            f"Basis: {formatted_basis_labels}"
        )

    def _display_DGCV_hook(self):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method.",
                    UserWarning,
                )

        def format_algebra_label(label):
            r"""Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
            if label and label[-1].isdigit():
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label and label.islower():
                return rf"\mathfrak{{{label}}}"
            return label or "Unnamed Algebra"

        return format_algebra_label(self.label)

    def _repr_latex_(self, verbose=False, abbrev=False, raw=False):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method.",
                    UserWarning,
                )

        def format_algebra_label(label):
            r"""
            Formats an algebra label for LaTeX. Handles:
            1. Labels with an underscore, splitting into two parts:
            - The first part goes into \mathfrak{} if it is lowercase.
            - The second part becomes a LaTeX subscript.
            2. Labels without an underscore:
            - Checks if the label ends in a numeric tail for subscripting.
            - Otherwise wraps the label in \mathfrak{} if it is entirely lowercase.

            Parameters
            ----------
            label : str
                The algebra label to format.

            Returns
            -------
            str
                A LaTeX-formatted algebra label.
            """

            if self._tex_label is not None:
                return self._tex_label

            if not label:
                return "\\text{{Unnamed Algebra}}"

            if "_" in label:
                # Split the label at the first underscore
                main_part, subscript_part = label.split("_", 1)
                if main_part.islower():
                    return rf"\mathfrak{{{main_part}}}_{{{subscript_part}}}"
                return rf"{main_part}_{{{subscript_part}}}"

            if label[-1].isdigit():
                # Split into text and numeric parts for subscripting
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"

            if label.islower():
                # Wrap entirely lowercase labels in \mathfrak{}
                return rf"\mathfrak{{{label}}}"

            # Return the label as-is if no special conditions apply
            return label

        if abbrev is True:
            if raw is True:
                return format_algebra_label(self.label)
            else:
                return f"${format_algebra_label(self.label)}$"

        def format_basis_label(label, idx):
            return rf"{label}" if label else f"e_{idx}"

        if verbose is True:
            formatted_label = format_algebra_label(self.label)
            formatted_basis_labels = (
                ", ".join(
                    [
                        format_basis_label(bl, idx)
                        for idx, bl in enumerate(self.basis_labels)
                    ]
                )
                if self.basis_labels
                else "No basis labels assigned"
            )
            return (
                f"Algebra: ${formatted_label}$, Basis: ${formatted_basis_labels}$, "
                f"Dimension: ${self.dimension}$, Grading: ${sp.latex(self.grading)}$"
            )
        else:
            formatted_str = f"\\langle{', '.join(elem._repr_latex_() for elem in self.basis)}\\rangle".replace(
                "$", ""
            ).replace(
                "\\displaystyle", ""
            )
            if self.label:
                if raw is True:
                    return f"{format_algebra_label(self.label)}={formatted_str}"
                else:
                    return f"$\\displaystyle {format_algebra_label(self.label)}={formatted_str}$"
            else:
                if raw is True:
                    return f"{formatted_str}"
                else:
                    return f"$\\displaystyle {formatted_str}$"

    def _latex(self, printer=None):
        return self._repr_latex_().replace("$", "").replace("\\displaystyle", "")

    def _sympystr(self):
        return f"algebra_class({self.structureData})"

    def is_skew_symmetric(self, verbose=False, _return_proof_path=False, _ignore_caches=False):
        """
        Checks if the algebra is skew-symmetric.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('is_skew',None),str) and _ignore_caches is False:
            t_message=self._educed_properties.get('is_skew',None)
            self._skew_symmetric_cache = (True,None)
        else:
            t_message=''

        if self._skew_symmetric_cache is None or _ignore_caches is True:
            result, failure = self._check_skew_symmetric()
            self._skew_symmetric_cache = (result, failure)
        else:
            result, failure = self._skew_symmetric_cache

        if verbose:
            if result:
                if self.label is None:
                    print("The algebra is skew-symmetric.")
                else:
                    print(f"{self.label} is skew-symmetric.")
            else:
                i, j, k = failure
                print(
                    f"Skew symmetry fails for basis elements {i}, {j}, at coefficient index {k}."
                )
        if _return_proof_path is True:
            return result, t_message
        return result

    def _check_skew_symmetric(self):
        for i in range(self.dimension):
            for j in range(i, self.dimension):
                for k in range(self.dimension):
                    vector_sum_element = (
                        self.structureData[i][j][k] + self.structureData[j][i][k]
                    )
                    if vector_sum_element != 0:
                        return False, (i, j, k)
        return True, None

    def satisfies_jacobi_identity(self, verbose=False, _return_proof_path=False,_ignore_caches=False):
        """
        Checks if the algebra satisfies the Jacobi identity.
        Includes a warning for unregistered instances only if verbose=True.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('satisfies_Jacobi_ID',None),str) and _ignore_caches is False:
            t_message=self._educed_properties.get('satisfies_Jacobi_ID',None)
            self._jacobi_identity_cache = (True,None)
        else:
            t_message=''


        if self._jacobi_identity_cache is None or _ignore_caches is True:
            result, fail_list = self._check_jacobi_identity()
            self._jacobi_identity_cache = (result, fail_list)
        else:
            result, fail_list = self._jacobi_identity_cache

        if verbose:
            if result:
                if self.label is None:
                    print("The algebra satisfies the Jacobi identity.")
                else:
                    print(f"{self.label} satisfies the Jacobi identity.")
            else:
                print(f"Jacobi identity fails for the following triples: {fail_list}")

        if _return_proof_path is True:
            return result, t_message
        return result

    def _check_jacobi_identity(self):
        skew = self.is_skew_symmetric()
        fail_list = []
        for i in range(self.dimension):
            lower_j = i + 1 if skew else 0
            for j in range(lower_j, self.dimension):
                lower_k = j + 1 if skew else 0
                for k in range(lower_k, self.dimension):
                    if not (
                        self.basis[i] * self.basis[j] * self.basis[k]
                        + self.basis[j] * self.basis[k] * self.basis[i]
                        + self.basis[k] * self.basis[i] * self.basis[j]
                    ).is_zero:
                        fail_list.append((i, j, k))
        if fail_list:
            return False, fail_list
        return True, None

    def _warn_associativity_assumption(self, method_name):
        """
        Issues a warning that the method assumes the algebra is associative.

        Parameters
        ----------
        method_name : str
            The name of the method assuming associativity.

        Notes
        -----
        - This helper method is intended for internal use.
        - Use it in methods where associativity is assumed but not explicitly verified.
        """
        warnings.warn(
            f"{method_name} assumes the algebra is associative. "
            "If it is not then unexpected results may occur.",
            UserWarning,
        )

    def is_lie_algebra(self, verbose=False, return_bool=True):
        warnings.warn(
            "`algebra_class.is_lie_algebra` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library. "
            "It will be removed in 2026. Please use `algebra_class.is_Lie_algebra` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_Lie_algebra(verbose=False, return_bool=True)

    def is_Lie_algebra(self, verbose=False, return_bool=True,_return_proof_path=False, _ignore_caches=False):
        """
        Checks if the algebra is a Lie algebra.
        Includes a warning for unregistered instances only if verbose=True.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed information about the check.
        return_bool : bool, optional
            Affects whether or not a boolian value is returned. If False, nothing is returned, which may be used in combination with verbose=True to have the function simply print a report.

        Returns
        -------
        bool or nothing
            True if the algebra is a Lie algebra, False otherwise. Nothing is returned if return_bool=False is set.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('is_Lie_algebra',None),str):
            t_message=self._educed_properties.get('is_Lie_algebra',None)
            self._lie_algebra_cache = True
            self._jacobi_identity_cache = True
            self._skew_symmetric_cache = True
        else:
            t_message=''

        if self._lie_algebra_cache is not None and _ignore_caches is False:
            if verbose:
                print(
                    f"Cached result: {f'Previously verified {self.label} is a Lie algebra' if self._lie_algebra_cache else f'Previously verified {self.label} is not a Lie algebra'}."
                )
            if _return_proof_path is True:
                return self._lie_algebra_cache, t_message
            return self._lie_algebra_cache

        if not self.is_skew_symmetric(verbose=verbose,_ignore_caches=_ignore_caches):
            self._lie_algebra_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, t_message
                return False
        if not self.satisfies_jacobi_identity(verbose=verbose,_ignore_caches=_ignore_caches):
            self._lie_algebra_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, t_message
                return False
        if self._lie_algebra_cache is None or _ignore_caches is True:
            self._lie_algebra_cache = True

        if verbose:
            if self.label is None:
                print("The algebra is a Lie algebra.")
            else:
                print(f"{self.label} is a Lie algebra.")

        if return_bool is True:
            if _return_proof_path is True:
                return self._lie_algebra_cache, t_message
            return self._lie_algebra_cache

    def _require_lie_algebra(self, method_name):
        """
        Checks that the algebra is a Lie algebra before proceeding.

        Parameters
        ----------
        method_name : str
            The name of the method requiring a Lie algebra.

        Raises
        ------
        ValueError
            If the algebra is not a Lie algebra.
        """
        if not self.is_Lie_algebra():
            raise ValueError(
                f"{method_name} can only be applied to Lie algebras."
            ) from None

    def is_semisimple(self, verbose=False, return_bool=True, _return_proof_path=False):
        """
        Checks if the algebra is semisimple.
        Nothing is returned if return_bool=False is set.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(self._print_warning, str):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('is_simple',None),str):
            t_message=self._educed_properties.get('is_simple',None)
            self._is_simple_cache = True
            self._is_semisimple_cache = True
            self._educed_properties['special_type'] = 'simple'
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        elif isinstance(self._educed_properties.get('is_semisimple',None),str):
            t_message=self._educed_properties.get('is_semisimple',None)
            self._is_semisimple_cache = True
            self._educed_properties['special_type'] = self._educed_properties.get('special_type',None) or 'semisimple'
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        else:
            t_message=''

        if self._is_semisimple_cache is None:
            if self._is_simple_cache is True:   # shouldn't happen since cache corollaries should always be updated 
                self._is_semisimple_cache=True
                self._is_solvable_cache=False
                self._is_abelian_cache=False
                self._is_nilpotent_cache=False
            elif self._Levi_deco_cache is not None:
                LC, MSI = self._Levi_deco_cache['LD_components']
                if getattr(MSI,'dimension',None) == 0 and self.dimension>0:
                    self._is_semisimple_cache=True
                    self._is_solvable_cache=False
                    self._is_abelian_cache=False
                    self._is_nilpotent_cache=False
                elif getattr(MSI,'dimension',None) != 0:
                    self._is_semisimple_cache=False
                    self._is_simple_cache=True
                    if getattr(LC,'dimension',None)==0:
                        self._is_solvable_cache=True
                        self._educed_properties['special_type'] = 'solvable'


        if self._is_semisimple_cache is not None:
            if verbose:
                print(
                    f"Cached result: {f'Previously verified {self.label} is a semisimple Lie algebra' if self._is_semisimple_cache else f'Previously verified {self.label} is not a semisimple Lie algebra.'}."
                )
            if return_bool is True:
                if _return_proof_path is True:
                    return self._is_semisimple_cache, t_message
                else:
                    return self._is_semisimple_cache
            elif _return_proof_path is True:
                return t_message
            else:
                return

        if not self.is_Lie_algebra(verbose=verbose):
            self._is_semisimple_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, 'not a Lie algebra'
                else:
                    return False
            elif _return_proof_path is True:
                return 'not a Lie algebra'
            else:
                return

        if verbose is True:
            print("Progress update: computing determinant of the Killing form...")
        det = sp.simplify(killingForm(self).det())

        if verbose:
            if det != 0:
                self._is_semisimple_cache = True
                self._educed_properties['special_type'] = 'semisimple'
                self._is_nilpotent_cache = False
                self._is_solvable_cache = False
                if self.label is None:
                    print("The algebra is semisimple.")
                else:
                    print(f"{self.label} is semisimple.")
            else:
                self._is_semisimple_cache = False
                self._is_simple_cache = False
                if self.label is None:
                    print("The algebra is not semisimple.")
                else:
                    print(f"{self.label} is not semisimple.")
        if return_bool is True:
            if _return_proof_path is True:
                return det != 0, t_message
            return det != 0

    def is_simple(self, verbose=False, bypass_semisimple_check=False,_return_proof_path=False):
        if isinstance(self._educed_properties.get('is_simple',None),str):
            t_message=self._educed_properties.get('is_simple',None)
            self._is_simple_cache = True
            self._is_semisimple_cache = True
            self._educed_properties['special_type'] = 'simple'
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        else:
            t_message=''

        if bypass_semisimple_check is False and self._is_semisimple_cache is None:
            self.is_semisimple(verbose=verbose)
        if self._is_simple_cache is None:
            self.compute_simple_subalgebras(verbose=verbose)
            if self._Levi_deco_cache['LD_components'][1].dimension == 0:
                self._is_semisimple_cache = True
                self._is_nilpotent_cache = False
                self._is_solvable_cache = False
                if len(self._Levi_deco_cache['simple_ideals'])==1:
                    self._is_simple_cache = True
                    self._educed_properties['special_type'] = 'simple'
                else:
                    self._is_simple_cache = False
                    self._educed_properties['special_type'] = 'semisimple'
            else:
                self._is_semisimple_cache = False
                self._is_simple_cache = False
                if self._Levi_deco_cache['LD_components'][0].dimension==0:
                    self._is_solvable_cache = True
                    if self._educed_properties['special_type'] is None:
                        self._educed_properties['special_type'] = 'solvable'
        if _return_proof_path is True:
            return self._is_simple_cache, t_message
        return self._is_simple_cache

    def is_subspace_subalgebra(
        self, elements, return_structure_data=False, check_linear_independence=False
    ):
        """
        Checks if a set of elements is a subspace is a subalgebra. `check_linear_independence` will additional verify if provided spanning elements are a basis.

        Parameters
        ----------
        elements : list
            A list of algebra_element_class instances.
        return_structure_data : bool, optional
            If True, returns the structure constants for the subalgebra. Returned
            data becomes a dictionary
        check_linear_independence : bool, optional
            If True, a check of linear independence of basis elements is also performed

        Returns
        -------
        dict or bool
            - If return_structure_data=True, returns a dictionary with keys:
            - 'linearly_independent': True/False
            - 'closed_under_product': True/False
            - 'structure_data': 3D list of structure constants
            - Otherwise, returns True if the elements form a subspace subalgebra, False otherwise.
        """

        # Perform linear independence check
        filtered_elem = self.filter_independent_elements(elements)
        new_dim = len(filtered_elem)
        linearly_independent = len(elements) == len(filtered_elem)
        closed_under_product = True
        if return_structure_data is True:
            structure_data = [
                [[0 for _ in range(new_dim)] for _ in range(new_dim)]
                for _ in range(new_dim)
            ]
        for count, elem in enumerate(filtered_elem):
            if closed_under_product is False:
                break
            lIdx = count + 1 if self.is_skew_symmetric() else 0
            for j in range(lIdx, new_dim):
                if closed_under_product is False:
                    break
                if not isinstance(return_structure_data, bool):
                    return_structure_data = False
                ic = _indep_check(
                    filtered_elem,
                    elem * filtered_elem[j],
                    return_decomp_coeffs=return_structure_data,
                )
                passCheck = ic if return_structure_data is False else ic[0]
                if passCheck is True:
                    closed_under_product = False
                    structure_data = None
                elif return_structure_data:
                    structure_data[count][j] = ic[1][0]
                    if self.is_skew_symmetric():
                        structure_data[j][count] = [-val for val in ic[1][0]]

        ##### former algorithm
        # span_matrix = sp.Matrix([list(el.coeffs) for el in filtered_elem]).transpose()

        # linearly_independent = len(elements)==len(filtered_elem)

        # # Check closure under product and build structure data
        # dim = len(filtered_elem)
        # structure_data = [
        #     [[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)
        # ]
        # closed_under_product = True

        # for i, el1 in enumerate(filtered_elem):
        #     if closed_under_product is False:
        #         break
        #     for j, el2 in enumerate(filtered_elem):
        #         product = el1 * el2
        #         prodVec = sp.Matrix(product.coeffs)
        #         solution = span_matrix.solve_least_squares(sp.Matrix(product.coeffs))
        #         if any(entry!=0 for entry in span_matrix*solution-prodVec):
        #             closed_under_product = False
        #             structure_data = None
        #             break
        #         for k, coeff in enumerate(solution):
        #             coeff_simplified = sp.nsimplify(coeff)
        #             structure_data[i][j][k] = coeff_simplified

        if return_structure_data:
            return {
                "linearly_independent": linearly_independent,
                "closed_under_product": closed_under_product,
                "structure_data": structure_data,
            }
        if check_linear_independence:
            return linearly_independent and closed_under_product
        else:
            return closed_under_product

    def check_element_weight(self, element, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of an algebra_element_class with respect to the grading vectors. Weight can be instead computed against another grading vector passed a list of weights as the keyword `test_weights`.

        Parameters
        ----------
        element : algebra_element_class
            The algebra_element_class to analyze.
        test_weights : list of int or sympy.Expr, optional (default: None)
        flatten_weights : (default: False) If True, returns contents of a list if otherwise would have returned a length 1 list

        Returns
        -------
        list or weight value
            A list of weights corresponding to the grading vectors of this algebra (or test_weights if provided).
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' (i.e., All Weights) if the element is the zero element,
            or 'NoW' (i.e., No Weights) if the element is not homogeneous.
            If the list is length 1 and flatten_weights=True then only the contents of the list is returned.

        Notes
        -----
        - 'AllW' (meaning, All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (meaning, No Weights) is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if (
            get_dgcv_category(element) == "subalgebra_element"
            and element.algebra.ambient == self
        ):
            element = element.ambient_rep
        if (
            not get_dgcv_category(element) == "algebra_element"
            or element.algebra != self
        ):
            raise TypeError(
                "Input in `algebra_class.check_element_weight` must be an `algebra_element` class instance belonging to the `algebra` instance whose `check_element_weight` is being called."
            ) from None
        if not test_weights and element._natural_weight is not None:
            if flatten_weights is True:
                return element._natural_weight[0]
            else:
                return element._natural_weight

        if all(coeff == 0 for coeff in element.coeffs):
            return tuple(["AllW"] * self._gradingNumber)
        if test_weights:
            if not isinstance(test_weights, (list, tuple)):
                raise TypeError(
                    f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, etc.). Revieved {test_weights}"
                ) from None
            for weight in test_weights:
                if not isinstance(weight, (list, tuple)):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, etc.).  Revieved {test_weights}"
                    ) from None
                if self.dimension != len(weight) or not all(
                    [isinstance(j, _get_expr_num_types()) for j in weight]
                ):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, etc.) of length {self.dimension}. Revieved {test_weights}"
                    ) from None
            GVs = test_weights
        else:
            GVs = self.grading
        weights = []
        for grading_vector in GVs:
            non_zero_indices = [
                i for i, coeff in enumerate(element.coeffs) if coeff != 0
            ]
            basis_weights = [grading_vector[i] for i in non_zero_indices]
            if len(set(basis_weights)) == 1:
                weights.append(basis_weights[0])
            else:
                weights.append("NoW")
        if not test_weights:
            element._natural_weight=weights
        if flatten_weights and len(weights) == 1:
            return weights[0]
        return tuple(weights)

    def check_grading_compatibility(
        self, verbose=False, test_weights=None, trust_test_weight_format=False
    ):
        """
        Checks if the algebra's structure constants are compatible with the assigned grading. Compatibility with alternate grading assignements can be tested by supplying alternate basis weights in the `test_weights` parameter

        Parameters
        ----------
        verbose : bool, optional (default=False)
            If True, prints detailed information about incompatibilities.
        test_weights : list of (lists of length self.dimension), optional
            elements in the inner lists represent weight values for basis elements to test compatibility against
        trust_test_weight_format : bool, optional (default=False)
            Set to True to forgo sefeguard checks that test_weights is correctly formatted

        Returns
        -------
        bool
            True if the algebra is compatible with all assigned grading vectors or given test weights, False otherwise.

        Notes
        -----
        - The algebra's zero element (weights labeled as 'AllW') are treated as compatible with all grading vectors.
        - Non-homogeneous elements (weights labeled as 'NoW') are treated as incompatible.
        """
        defualt_check = False
        if test_weights is None:
            defualt_check = True
            test_weights = self.grading
        elif trust_test_weight_format is False:
            if not isinstance(test_weights, (list, tuple)):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )
            elif not all(
                isinstance(j, (list, tuple)) and len(j) == self.dimension
                for j in test_weights
            ):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )

        if defualt_check is True and not self._gradingNumber:
            raise ValueError(
                "No grading vectors are assigned to this algebra instance."
            ) from None
        if (
            defualt_check is True
            and isinstance(self._grading_compatible, bool)
            and self._grading_report
        ):
            compatible = self._grading_compatible
            failure_details = self._grading_report
        else:
            compatible = True
            failure_details = []

            for i, el1 in enumerate(self.basis):
                for j, el2 in enumerate(self.basis):
                    product = el1 * el2
                    product_weights = self.check_element_weight(product)

                    for g, grading_vector in enumerate(test_weights):
                        expected_weight = grading_vector[i] + grading_vector[j]

                        if product_weights[g] == "AllW":
                            continue  # Zero product is compatible with all weights

                        if (
                            product_weights[g] == "NoW"
                            or product_weights[g] != expected_weight
                        ):
                            compatible = False
                            failure_details.append(
                                {
                                    "grading_vector_index": g + 1,
                                    "basis_elements": (i + 1, j + 1),
                                    "weights": (grading_vector[i], grading_vector[j]),
                                    "expected_weight": expected_weight,
                                    "actual_weight": product_weights[g],
                                }
                            )
            self._grading_compatible = compatible
            self._grading_report = failure_details

        if verbose:
            if not compatible:
                print("Grading Compatibility Check Failed:")
                for failure in failure_details:
                    print(
                        f"- Grading Vector {failure['grading_vector_index']}: "
                        f"Basis elements {failure['basis_elements'][0]} and {failure['basis_elements'][1]} "
                        f"(weights: {failure['weights'][0]}, {failure['weights'][1]}) "
                        f"produced weight {failure['actual_weight']}, expected {failure['expected_weight']}."
                    )
            else:
                if defualt_check:
                    ps = "all of its assigned Z-gradings."
                else:
                    ps = "the given weight system."
                print(f"The algebra structure of {self.label} is compatible with " + ps)
        return compatible

    def compute_center(self, for_associative_alg=False, assume_Lie_algebra=False):
        """
        Computes the center of the algebra as a subspace.

        Parameters
        ----------
        for_associative_alg : bool, optional
            If True, computes the center for an associative algebra. Defaults to False (assumes Lie algebra).

        Returns
        -------
        list
            A list of algebra_element_class instances that span the center of the algebra.

        Raises
        ------
        ValueError
            If `for_associative_alg` is False and the algebra is not a Lie algebra.

        Notes
        -----
        - For Lie algebras, the center is the set of elements `z` such that `z * x = 0` for all `x` in the algebra.
        - For associative algebras, the center is the set of elements `z` such that `z * x = x * z` for all `x` in the algebra.
        """
        self._set_product_protocol()

        if for_associative_alg is True:
            assume_Lie_algebra is False
        elif assume_Lie_algebra is False and not self.is_Lie_algebra():
            raise ValueError(
                "This algebra is not a Lie algebra. To compute the center for an associative algebra, set for_associative_alg=True."
            ) from None

        temp_label = create_key(prefix="center_var")
        variableProcedure(temp_label, self.dimension, _tempVar=retrieve_passkey())
        temp_vars = _cached_caller_globals[temp_label]

        el = sum(
            (temp_vars[i] * self.basis[i] for i in range(self.dimension)),
            self.basis[0] * 0,
        )

        if for_associative_alg:
            eqns = sum(
                [list((el * other - other * el).coeffs) for other in self.basis], []
            )
        else:
            eqns = sum([list((el * other).coeffs) for other in self.basis], [])

        solutions = solve_dgcv(eqns, temp_vars)
        if not solutions:
            warnings.warn(
                "The internal solver (which defaults to sympy.solve or sympy.linsolve unless running another symbolic engine) returned no solutions, indicating that this computation of the center failed, as solutions do exist. An empty list is being returned."
            )
            return []

        el_sol = el.subs(solutions[0])

        free_variables = tuple(set.union(*[set(j.free_symbols) for j in el_sol.coeffs]))

        return_list = []
        for var in free_variables:
            basis_element = el_sol.subs({var: 1}).subs(
                [(other_var, 0) for other_var in free_variables if other_var != var]
            )
            return_list.append(basis_element)

        clearVar(*listVar(temporary_only=True), report=False)

        return return_list  ###!!! return subalgebra instead

    def compute_derived_algebra(self, from_subalg=None):
        """
        Computes the derived algebra (commutator subalgebra) for Lie algebras.

        Returns
        -------
        algebra
            A new algebra instance representing the derived algebra.

        Raises
        ------
        ValueError
            If the algebra is not a Lie algebra or if the derived algebra cannot be computed.

        Notes
        -----
        - This method only applies to Lie algebras.
        - The derived algebra is generated by all products [x, y] = x * y, where * is the Lie bracket.
        """
        if get_dgcv_category(from_subalg) == "subalgebra":
            refAlg = from_subalg
        else:
            refAlg = self
        refAlg._set_product_protocol()

        ###!!!
        # refAlg._require_lie_algebra("compute_derived_algebra")

        if refAlg._derived_subalg_cache is None:
            commutators = []
            for j, el1 in enumerate(refAlg.basis):
                for k in range(j + 1, len(refAlg.basis)):
                    commutators.append(el1 * refAlg.basis[k])
            refAlg._derived_subalg_cache = refAlg.subalgebra(
                commutators, span_warning=False, simplify_basis=True
            )
        return refAlg._derived_subalg_cache

    def filter_independent_elements(
        self, elements, apply_light_basis_simplification=False
    ):
        """
        Filters a set of elements to retain only a linearly independent subset.

        Parameters
        ----------
        elements : list of algebra_element_class
            The set of elements to filter.

        Returns
        -------
        list of algebra_element_class
            A subset of the input elements that are linearly independent and unique.
        """
        warning_message = ""
        remain_subalg = True
        subalg = None
        if not isinstance(elements, (list, tuple)):
            warning_message += (
                "\n The given value for `elements` is not a list or tuple"
            )
        else:
            nonAE = []
            wrongAlgebra = []
            typeCheck = {"algebra_element", "subalgebra_element"}
            for elem in elements:
                if remain_subalg is True:
                    if get_dgcv_category(elem) == "algebra_element":
                        remain_subalg = False
                    elif get_dgcv_category(elem) == "subalgebra_element":
                        if subalg is None:
                            subalg = elem.algebra
                        elif subalg != elem.algebra:
                            remain_subalg = False
                if get_dgcv_category(elem) not in typeCheck:
                    nonAE.append(elem)
                elif (
                    get_dgcv_category(elem) == "algebra_element"
                    and elem.algebra != self
                ) or (
                    get_dgcv_category(elem) == "subalgebra_element"
                    and elem.algebra.ambient != self
                ):
                    wrongAlgebra.append(elem)
            if len(nonAE) > 0 or len(wrongAlgebra) > 0:
                if len(nonAE) > 0:
                    warning_message += f"\n • These list elements are not `algebra_element` or `subalgebra_element` type: {nonAE}"
                if len(wrongAlgebra) > 0:
                    warning_message += f"\n • These list elements are `algebra_element` or `subalgebra_element` type, but belong to a different, unrelated algebra: {wrongAlgebra}"
        if warning_message:
            raise ValueError(
                "The `algebra` method `filter_independent_elements` can only be applied to lists of elements belong to the parent algebra the method is called from or any its subalgebras. Given data has the following problems:"
                + warning_message
            ) from None
        if remain_subalg is False:
            elements = [(elem.ambient_rep if get_dgcv_category(elem) == "subalgebra_element" else elem) for elem in elements]
        else:
            elements = list(elements)
        return _extract_basis(elements, ALBS=apply_light_basis_simplification)
        #### former algorithm using sp.Matrix methods:
        # unique_elements = []
        # seen_coeffs = set()
        # for el in elements:
        #     if get_dgcv_category(el)=='subalgebra_element':
        #         el = el.ambient_rep
        #     coeff_tuple = tuple(el.coeffs)  # Convert coeffs to a tuple for hashability
        #     if coeff_tuple not in seen_coeffs:
        #         seen_coeffs.add(coeff_tuple)
        #         unique_elements.append(el)
        # coeff_matrix = sp.Matrix([list(el.coeffs) for el in unique_elements]).transpose()
        # independent_vectors = coeff_matrix.columnspace()
        # independent_indices = []
        # for vec in independent_vectors:
        #     for i in range(coeff_matrix.cols):
        #         if list(coeff_matrix[:, i]) == list(vec):
        #             independent_indices.append(i)
        #             break
        # independent_elements = [unique_elements[i] for i in independent_indices]
        # return independent_elements

    def lower_central_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        from_subalg=None,
        align_nested_bases=False,
    ):
        """
        Computes the lower central series of the algebra (or given subalgebra).

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to compute the series. Defaults to the dimension of the algebra.
        from_subalg : subalgebra_class, optional
            performs computation the subalgebra instead of self

        Returns
        -------
        list of lists
            A list where each entry contains the basis for that level of the lower central series.

        Notes
        -----
        - The lower central series is defined as:
            g_1 = g,
            g_{k+1} = [g_k, g]
        """
        if get_dgcv_category(from_subalg) == "subalgebra":
            refAlg = from_subalg
        else:
            refAlg = self
        refAlg._set_product_protocol()
        scoped_basis = list(refAlg.basis)
        if refAlg._lower_central_series_cache is None:
            if max_depth is None:
                max_depth = refAlg.dimension
            series = []
            current_basis = scoped_basis
            previous_length = len(current_basis)

            for _ in range(max_depth):
                series.append(current_basis)

                lower_central = []
                for el1 in current_basis:
                    for el2 in scoped_basis:
                        commutator = el1 * el2
                        lower_central.append(commutator)
                independent_generators = self.filter_independent_elements(
                    lower_central, apply_light_basis_simplification=True
                )
                if len(independent_generators) == 0:
                    if len(scoped_basis) > 0:
                        series.append([0 * scoped_basis[0]])
                    break
                if len(independent_generators) == previous_length:
                    break
                current_basis = independent_generators
                previous_length = len(independent_generators)
            if len(series) > 1 and refAlg._derived_subalg_cache is None:
                refAlg._derived_subalg_cache = self.subalgebra(
                    series[1], span_warning=False, simplify_basis=True
                )
            refAlg._lower_central_series_cache = (
                series,
                False,
            )  # series, alignment bool
        if (
            align_nested_bases is True
            and refAlg._lower_central_series_cache[1] is False
        ):
            if len(refAlg._lower_central_series_cache[0]) > 0 and get_dgcv_category(
                refAlg._lower_central_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                ser = [list(alg.basis) for alg in refAlg._lower_central_series_cache[0]]
            else:
                ser = refAlg._lower_central_series_cache[0]
            new_series = [ser[-1]]
            depth = len(ser)
            for idx in range(1, depth):
                old_level = ser[depth - 1 - idx]
                discrep = len(old_level) - len(ser[depth - idx])
                new_level = list(new_series[0])
                for idx2 in range(len(old_level)):
                    if discrep == 0:
                        break
                    elem = old_level[-1 - idx2]
                    if _indep_check(ser[depth - idx], elem):
                        new_level.insert(0, elem)
                        discrep += -1
                new_series.insert(0, new_level)
            refAlg._lower_central_series_cache = (
                new_series,
                True,
            )  # series, alignment bool
        if format_as_subalgebras:
            if len(refAlg._lower_central_series_cache[0]) > 0 and isinstance(
                refAlg._lower_central_series_cache[0][0], list
            ):
                refAlg._lower_central_series_cache = (
                    [
                        refAlg.subalgebra(sa, span_warning=False)
                        for sa in refAlg._lower_central_series_cache[0]
                    ],
                    refAlg._lower_central_series_cache[1],
                )
            returnSer = refAlg._lower_central_series_cache[0]
        else:
            if len(refAlg._lower_central_series_cache[0]) > 0 and get_dgcv_category(
                refAlg._lower_central_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                returnSer = [
                    list(alg.basis) for alg in refAlg._lower_central_series_cache[0]
                ]
            else:
                returnSer = refAlg._lower_central_series_cache[0]
        return returnSer

    def derived_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        from_subalg=None,
        align_nested_bases=False,
    ):
        """
        Computes the derived series of the algebra.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to compute the series. Defaults to the dimension of the algebra.

        Returns
        -------
        list of lists
            A list where each entry contains the basis for that level of the derived series.

        Notes
        -----
        - The derived series is defined as:
            g^{(1)} = g,
            g^{(k+1)} = [g^{(k)}, g^{(k)}]
        """
        if get_dgcv_category(from_subalg) == "subalgebra":
            refAlg = from_subalg
        else:
            refAlg = self
        refAlg._set_product_protocol()
        scoped_basis = list(refAlg.basis)
        if refAlg._derived_series_cache is None:
            if max_depth is None:
                max_depth = refAlg.dimension

            series = []
            current_basis = scoped_basis
            previous_length = len(current_basis)

            for _ in range(max_depth):
                series.append(list(current_basis))

                derived = []
                for count, el1 in enumerate(current_basis):
                    lIdx = count + 1 if refAlg.is_skew_symmetric() else 0
                    for el2 in current_basis[lIdx:]:
                        derived.append(el1 * el2)
                independent_generators = self.filter_independent_elements(
                    derived, apply_light_basis_simplification=True
                )
                if len(independent_generators) == 0:
                    if len(scoped_basis) > 0:
                        series.append([0 * self.basis[0]])
                    break
                if len(independent_generators) == previous_length:
                    break

                current_basis = list(independent_generators)
                previous_length = len(independent_generators)
            if len(series) > 1 and refAlg._derived_subalg_cache is None:
                refAlg._derived_subalg_cache = self.subalgebra(
                    series[1], span_warning=False, simplify_basis=True
                )
            refAlg._derived_series_cache = (series, False)  # series, alignment bool
        if align_nested_bases is True and refAlg._derived_series_cache[1] is False:
            if len(refAlg._derived_series_cache[0]) > 0 and get_dgcv_category(
                refAlg._derived_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                ser = [list(alg.basis) for alg in refAlg._derived_series_cache[0]]
            else:
                ser = refAlg._derived_series_cache[0]
            depth = len(ser)
            new_series = [] if depth == 0 else [ser[-1]]
            build_step = 1
            if len(new_series)==1 and len(new_series[0])==1 and getattr(new_series[0][0],'is_zero',False):
                new_series.insert(0, ser[-2])
                build_step=2
            for idx in range(build_step, depth):
                old_level = ser[depth - 1 - idx]
                discrep = len(old_level) - len(ser[depth - idx])
                new_level = list(new_series[0])
                for idx2 in range(len(old_level)):
                    if discrep == 0:
                        break
                    elem = old_level[-1 - idx2]
                    if _indep_check(ser[depth - idx], elem):
                        new_level.insert(0, elem)
                        discrep += -1
                new_series.insert(0, new_level)
            refAlg._derived_series_cache = (new_series, True)  # series, alignment bool
        if format_as_subalgebras:
            if len(refAlg._derived_series_cache[0]) > 0 and isinstance(
                refAlg._derived_series_cache[0][0], list
            ):
                refAlg._derived_series_cache = (
                    [
                        refAlg.subalgebra(sa, span_warning=False)
                        for sa in refAlg._derived_series_cache[0]
                    ],
                    refAlg._derived_series_cache[1],
                )
            returnSer = refAlg._derived_series_cache[0]
        else:
            if len(refAlg._derived_series_cache[0]) > 0 and get_dgcv_category(
                refAlg._derived_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                returnSer = [list(alg.basis) for alg in refAlg._derived_series_cache[0]]
            else:
                returnSer = refAlg._derived_series_cache[0]
        return returnSer

    def is_nilpotent(self,**kwargs):
        """
        Checks if the algebra is nilpotent.

        Returns
        -------
        bool
            True if the algebra is nilpotent, False otherwise.
        """
        if self._is_nilpotent_cache is None:
            series = self.lower_central_series()
            if len(series[-1])<2: # to allow different conventions for formatting a trivial level basis
                self._is_nilpotent_cache=True
                self._educed_properties['special_type'] = 'nilpotent'
                self._is_semisimple_cache=False
                self._is_simple_cache=False
            else:
                self._is_nilpotent_cache=False
                self._is_abelian_cache=True
        return self._is_nilpotent_cache

    def is_solvable(self,**kwargs):
        """
        Checks if the algebra is solvable.

        Returns
        -------
        bool
            True if the algebra is solvable, False otherwise.
        """
        if self._is_solvable_cache is None:
            if self._is_nilpotent_cache is None or self._is_nilpotent_cache is False:
                series = self.derived_series()
                if len(series[-1])<2: # to allow different conventions for formatting a trivial level basis
                    self._is_solvable_cache=True
                    self._is_semisimple_cache=False
                    self._is_simple_cache=False
                    self._educed_properties['special_type'] = 'solvable'
                else:
                    self._is_solvable_cache=False
                    self._is_abelian_cache=False
                    self._is_nilpotent_cache=False
            else:
                self._is_solvable_cache=self._is_nilpotent_cache
        return self._is_solvable_cache

    def is_abelian(self,**kwargs):            
        if self._is_abelian_cache is None:
            if self._educed_properties.get('special_type',None) == 'abelian':
                self._is_abelian_cache is True
                self._is_nilpotent_cache=True
                self._is_solvable_cache=True
                self._is_semisimple_cache=False
                self._is_simple_cache=False
            else:
                self._is_abelian_cache = all(elem == 0 for elem in self.structureDataDict.values())
                if self._is_abelian_cache is True:
                    self._educed_properties['special_type'] = 'abelian'
                    self._is_nilpotent_cache=True
                    self._is_solvable_cache=True
                    self._is_semisimple_cache=False
                    self._is_simple_cache=False
        return self._is_abelian_cache

    def get_structure_matrix(self, table_format=True, style=None):
        """
        Computes the structure matrix for the algebra.

        Parameters
        ----------
        table_format : bool, optional
            If True (default), returns a nicely formatted table.
            If False, returns a raw list of lists.
        style : str, optional
            table themes.

        Returns
        -------
        list of lists or tableView object
            The structure matrix as a list of lists or a tableView object
            depending on the value of `table_format`.

        Notes
        -----
        - The (j, k)-entry of the structure matrix is the result of `basis[j] * basis[k]`.
        - If `basis_labels` is None, defaults to "_e1", "_e2", ..., "_e{d}".
        """

        dimension = self.dimension
        structure_matrix = [
            [(self.basis[j] * self.basis[k]) for k in range(dimension)]
            for j in range(dimension)
        ]
        return structure_matrix

    def is_ideal(self, subspace_elements):
        """
        Checks if the given list of elgebra elements spans an ideal.

        Parameters
        ----------
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace
            they span.

        Returns
        -------
        bool
            True if the subspace is an ideal, False otherwise.

        Raises
        ------
        ValueError
            If the provided elements do not belong to this algebra.
        """
        # Checks that all subspace elements belong to this algebra
        for el in subspace_elements:
            if not isinstance(el, algebra_element_class) or el.algebra != self:
                raise ValueError(
                    "All elements in subspace_elements must belong to this algebra."
                ) from None

        # Check the ideal condition
        for el in subspace_elements:
            for other in self.basis:
                # Compute the product and check if it is in the span of subspace_elements
                product = el * other
                if not self.is_in_span(product, subspace_elements):
                    return False
        return True

    def is_in_span(self, element, subspace_elements):
        """
        Checks if a given algebra_element_class is in the span of subspace_elements.

        Parameters
        ----------
        element : algebra_element_class
            The element to check.
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace they span.

        Returns
        -------
        bool
            True if the element is in the span of subspace_elements, False otherwise.
        """
        # Build a matrix where columns are the coefficients of subspace_elements
        span_matrix = sp.Matrix(
            [list(el.coeffs) for el in subspace_elements]
        ).transpose()

        # Solve for the coefficients that express `element` as a linear combination
        product_vector = sp.Matrix(element.coeffs)
        solution = span_matrix.solve_least_squares(product_vector)

        # Check if the solution satisfies the equation
        return span_matrix * solution == product_vector

    def weighted_component(self, weights, test_weights=None, trust_test_weight_format=False, from_subalg=None):
        if isinstance(weights,(set,dict)):
            weights=list(weights)
        if get_dgcv_category(from_subalg) == "subalgebra":
            refAlg = from_subalg
        else:
            refAlg = self
        if isinstance(weights, (list, tuple)):
            if all(isinstance(weight, _get_expr_num_types()) for weight in weights):
                weights = [[weight] for weight in weights]
            elif not all(isinstance(weight, (list, tuple)) for weight in weights):
                raise ValueError(
                    "The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights."
                ) from None
            else:
                weights = [list(weight) for weight in weights]
        else:
            raise ValueError(
                f"The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights. Instead recieved{weights}"
            ) from None
        if test_weights is None:
            test_weights = refAlg.grading
        elif trust_test_weight_format is False:
            if not isinstance(test_weights, (list, tuple)):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )
            elif not all(
                isinstance(j, (list, tuple)) and len(j) == refAlg.dimension
                for j in test_weights
            ):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )
        component = []
        for elem in refAlg.basis:
            if list(elem.check_element_weight(test_weights=test_weights)) in weights:
                component.append(elem)
        return algebra_subspace_class(component, parent_algebra=refAlg)

    def multiplication_table(
        self,
        elements=None,
        restrict_to_subspace=False,
        style=None,
        use_latex=None,
        _called_from_subalgebra=None,
    ):
        """
        Multiplication table for the algebra.
        """

        if elements is None:
            elements = self.basis
        elif not all(isinstance(elem, algebra_element_class) and elem.algebra == self for elem in elements):
            raise ValueError("All elements must be instances of algebraElement.") from None

        if restrict_to_subspace is True:
            basis_elements = elements
        elif isinstance(restrict_to_subspace, (list, tuple)) and all(isinstance(elem, algebra_element_class) and elem.algebra == self for elem in restrict_to_subspace):
            basis_elements = restrict_to_subspace
        elif isinstance(_called_from_subalgebra, dict) and _called_from_subalgebra.get("internalLock", None) == retrieve_passkey():
            basis_elements = _called_from_subalgebra["basis"]
        else:
            basis_elements = self.basis

        dgcvSR = get_dgcv_settings_registry()
        if use_latex is None:
            use_latex = dgcvSR.get("use_latex", False)
        style_key = style or dgcvSR.get("theme", "dark")

        def _to_string(element, ul=False):
            if ul:
                s = element._repr_latex_(verbose=False)
                if s.startswith("$") and s.endswith("$"):
                    s = s[1:-1]
                s = s.replace(r"\\displaystyle", "").replace(r"\displaystyle", "").strip()
                return f"${s}$"
            return str(element)

        headers = [_to_string(e, ul=use_latex) for e in elements]
        index_headers = [_to_string(e, ul=use_latex) for e in basis_elements]

        data = []
        for left in basis_elements:
            row = [_to_string(left * right, ul=use_latex) for right in elements]
            data.append(row)

        loc_style = get_style(style_key)

        border_style = "1px solid #ccc"
        for sd in loc_style:
            if sd.get("selector") == "table":
                for prop_name, prop_value in sd.get("props", []):
                    if prop_name == "border":
                        border_style = prop_value
                        break
                break

        col_heading_props = []
        for sd in loc_style:
            if sd.get("selector") == "th.col_heading.level0":
                col_heading_props = sd.get("props", [])
                break

        row_heading_visual = [(k, v) for (k, v) in col_heading_props if not k.startswith("border")]

        parts = border_style.split()
        thickness = parts[0] if parts else "1px"
        color = parts[-1] if parts else "#ccc"
        solid = border_style
        dashed = f"{thickness} dashed {color}"

        additional_styles = [
            {"selector": "table", "props": [("border-collapse", "collapse")]},
            {"selector": "thead th:not(:first-child)", "props": [("border-bottom", solid)]},
            {"selector": "tbody th", "props": [("border-right", solid)]},
            {"selector": "th.row_heading", "props": row_heading_visual},
            {"selector": "thead th:first-child", "props": [("border-right", dashed), ("border-bottom", dashed)]},
        ]        
        table_styles = loc_style + additional_styles
        table = build_matrix_table(
            index_labels=index_headers,
            columns=headers,
            rows=data,
            caption="Multiplication Table",
            theme_styles=table_styles,
            extra_styles=None,
            table_attrs='style="max-width:900px; table-layout:fixed; overflow-x:auto;"',
            cell_align="center",
            escape_cells=False,
            escape_headers=False,
            escape_index=False,
            nowrap=True,
            ul=0,
            ur=0,
            ll=0,
            lr=0
        )

        return latex_in_html(table)

    def subalgebra(
        self,
        basis,
        grading=None,
        span_warning=True,
        simplify_basis=False,
        simplify_products_by_default=None,
    ):
        from .algebras_secondary import subalgebra_class

        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if get_dgcv_category(basis) in {"algebra_subspace","algebra"}:
            basis = basis.basis
        use_slices = True
        subIndices=[]
        index_map = dict()
        for count,elem in enumerate(basis):
            try:
                idx=self.basis.index(elem)
                index_map[idx]=count
                subIndices.append(idx)
            except ValueError:
                use_slices = False
                break
        if use_slices:
            def truncateBySubInd(li, check_compl=False):
                if check_compl is True:
                    new_li = [0]*len(subIndices)
                    for count, elem in enumerate(li):
                        if count in subIndices:
                            new_li[index_map[count]]=elem
                        elif elem != 0:
                            raise TypeError("The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra. Suggestion: use `algebra_class.subspace` instead.") from None
                    return new_li
                return [li[j] for j in subIndices]

            if isinstance(grading, (list, tuple)) and all(
                isinstance(elem, (list, tuple)) for elem in grading
            ):
                gradings = grading
            else:
                if grading is not None:
                    warnings.warn(
                        "The `gradings` keyword given to `algebra_class.subalgebra` was in an unsupported format (i.e., not list of lists), so a valid alternate gradings vector was computed instead inherited from the parent algebra."
                    )
                gradings = [truncateBySubInd(grading) for grading in self.grading]
            structureData = truncateBySubInd(self.structureData)
            structureData = [truncateBySubInd(plane) for plane in structureData]
            structureData = [
                [truncateBySubInd(li, check_compl=True) for li in plane]
                for plane in structureData
            ]
            return subalgebra_class(
                basis,
                self,
                grading=gradings,
                _compressed_structure_data=structureData,
                _internal_lock=retrieve_passkey(),
            )
        testStruct = self.is_subspace_subalgebra(basis, return_structure_data=True)
        if testStruct["closed_under_product"] is not True:
            raise TypeError("The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra. Suggestion: use `algebra_class.subspace` instead.") from None
        return subalgebra_class(
            basis,
            self,
            grading=grading,
            _compressed_structure_data=testStruct["structure_data"],
            _internal_lock=retrieve_passkey(),
            span_warning=span_warning,
            simplify_basis=simplify_basis,
            simplify_products_by_default=simplify_products_by_default,
        )

    def new_alg_from_subalgebra(
        self,
        basis,
        grading=None,
        span_warning=True,
        simplify_basis=False,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
    ):
        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if get_dgcv_category(basis) == "subalgebra_class" and basis.ambient == self:
            alg = basis
        else:
            alg = self.subalgebra(
                basis,
                grading=grading,
                span_warning=span_warning,
                simplify_basis=simplify_basis,
                simplify_products_by_default=simplify_products_by_default,
            )
        return alg.copy(
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
        )

    def subspace(self, basis:list|tuple=[], grading=None, span_warning=True):
        if grading is None:
            grading = self.grading
        return algebra_subspace_class(basis, parent_algebra=self, test_weights=grading, span_warning=span_warning)

    def killing_form_product(self, elem1, elem2, assume_Lie_algebra=False):
        kf = killingForm(self, assume_Lie_algebra=assume_Lie_algebra)
        vec1 = sp.Matrix(elem1.coeffs)
        vec2 = sp.Matrix(elem2.coeffs)
        return (vec2.transpose() * kf * vec1)[0]

    def radical(self, from_subalg=None, assume_Lie_algebra=False,):
        if get_dgcv_category(from_subalg) == "subalgebra":
            refAlg = from_subalg
            amb_basis = from_subalg.basis_in_ambient_alg
            parent = from_subalg.ambient
        else:
            refAlg = self
            amb_basis = self.basis
            parent = self
        if refAlg._radical_cache is None:
            da = refAlg.compute_derived_algebra()
            pref = "v" + create_key()
            variableProcedure(pref, refAlg.dimension, _tempVar=retrieve_passkey())
            vars = _cached_caller_globals[pref]
            terms = [var * elem for var, elem in zip(vars, amb_basis)]
            genElem = sum(terms[1:], terms[0])
            eqns = []
            for elem in da.basis_in_ambient_alg:
                eqns.append(
                    parent.killing_form_product(
                        genElem, elem, assume_Lie_algebra=assume_Lie_algebra
                    )
                )
            sol = solve_dgcv(eqns, vars)
            if len(sol) == 0:
                raise RuntimeError("failed to compute radical.")
            else:
                genSol = genElem.subs(sol[0])
            freeVars = getattr(genSol, "free_symbols", {})
            if len(freeVars) != 0:
                radSpanners = []
                for var in freeVars:
                    radSpanners.append(
                        genSol.subs({var: 1}).subs({v: 0 for v in freeVars})
                    )
            else:
                radSpanners = []
            refAlg._radical_cache = refAlg.subalgebra(radSpanners, span_warning=False)
        clearVar(*listVar(temporary_only=True),report=False)
        return refAlg._radical_cache

    def compute_simple_subalgebras(self,verbose=False):
        self.Levi_decomposition(decompose_semisimple_fully=True,verbose=verbose)
        return self._Levi_deco_cache['simple_ideals']

    def Levi_decomposition(
        self,
        from_subalg=None,
        decompose_semisimple_fully = False,
        _bust_cache=False,
        assume_Lie_algebra=False,
        _try_multiple_times=None,
        verbose=False
    ):
        if (isinstance(_try_multiple_times, numbers.Integral) and _try_multiple_times > 0):
            attempts = int(_try_multiple_times)
            _bust_cache = True
        elif _try_multiple_times is True:
            attempts = 100  # testing... cap to avoid infinite loop... !!! to remove eventually, but harmless for now
            _bust_cache = True
        else:
            attempts = 1

        loop = 0
        last_err = None

        while loop < attempts:
            if loop > 0 and loop % 20 == 0:
                print(f"Trying loop #{loop}...")
            try:
                refAlg = self if get_dgcv_category(from_subalg) != "subalgebra" else from_subalg
                if _bust_cache:
                    refAlg._radical_cache = None
                    refAlg._derived_series_cache = None
                    refAlg._lower_central_series_cache = None
                    refAlg._derived_subalg_cache = None
                if refAlg._Levi_deco_cache is None:
                    if refAlg._educed_properties.get('special_type',None) in {'simple','semisimple'}:
                        refAlg._Levi_deco_cache = {'LD_components':(refAlg, refAlg.subalgebra([])),'simple_ideals':None}
                    elif refAlg._educed_properties.get('special_type',None) in {'nilpotent','solvable','abelian'}:
                        refAlg._Levi_deco_cache = {'LD_components':(refAlg.subalgebra([]),refAlg),'simple_ideals':None}
                    else:
                        if verbose is True:
                            print('Deriving (or retrieving) maximal solvable ideal...')
                        rad = refAlg.radical(assume_Lie_algebra=assume_Lie_algebra)
                        if len(rad.basis)>0:
                            if verbose is True:
                                print('Finding a semisimple complement to the max. solvable ideal...')
                            rad_seq = rad.derived_series(align_nested_bases=True)
                            discrep = refAlg.dimension - len(rad_seq[0])
                            naiveBasis = []
                            augment_NB = list(rad_seq[0])
                            for elem in refAlg.basis:
                                if len(naiveBasis) == discrep:
                                    break
                                if _indep_check(augment_NB, elem):
                                    augment_NB.append(elem)
                                    naiveBasis.append(elem)
                            ss_dim = len(naiveBasis)
                            for idx in range(len(rad_seq)):
                                if idx == len(rad_seq) - 1:
                                    compare_set = rad_seq[idx]
                                    quot_set = []
                                else:   # relying on nesting alignment
                                    rad_discrep = len(rad_seq[idx]) - len(rad_seq[idx + 1])
                                    compare_set = rad_seq[idx][:rad_discrep]
                                    quot_set = rad_seq[idx][rad_discrep:]
                                compLen = len(compare_set)
                                tailLen = len(quot_set)
                                pref = create_key(prefix="v")
                                vars = []
                                basis_modifiers = []
                                for count, w in enumerate(naiveBasis):
                                    w_vars = [sp.Symbol(f"{pref}_{count}_{j}") for j in range(compLen)]
                                    vars += w_vars
                                    w_modifiers = [var * el for var, el in zip(w_vars, compare_set)]
                                    if compLen > 1:
                                        basis_modifiers.append(sum(w_modifiers[1:], w_modifiers[0]))
                                    elif compLen > 0:
                                        basis_modifiers.append(w_modifiers[0])
                                    else:   ###!!! review
                                        basis_modifiers.append(0 * naiveBasis[0])

                                leading_coeffs = {}
                                trailing_coeffs = {}
                                eqns = []
                                for idx1 in range(ss_dim):
                                    for idx2 in range(idx1 + 1, ss_dim):
                                        w1, w2 = naiveBasis[idx1], naiveBasis[idx2]
                                        lb = w1 * w2
                                        lb_decomp = _indep_check(naiveBasis + rad_seq[idx],lb,return_decomp_coeffs=True)    ###!!! subalgebras bug here
                                        lb_decomp=lb_decomp[1][0]
                                        leading_coeffs[(idx1, idx2)] = lb_decomp[:ss_dim]
                                        trailing_coeffs[(idx1, idx2)] = lb_decomp[ss_dim : ss_dim + compLen]
                                for idxs in leading_coeffs:
                                    oldV = [coe * el for coe, el in zip(trailing_coeffs[idxs], compare_set)]
                                    vTerms = [-coe * el for coe, el in zip(leading_coeffs[idxs], basis_modifiers)]
                                    newV = (naiveBasis[idxs[0]] * basis_modifiers[idxs[1]] - naiveBasis[idxs[1]] * basis_modifiers[idxs[0]])
                                    t_vars = [sp.Symbol(f"t{pref}_{idxs[0]}_{idxs[1]}_{j}") for j in range(tailLen)]
                                    vars += t_vars
                                    qTerms = [var * el for var, el in zip(t_vars, quot_set)]
                                    eqns.append(sum(oldV + vTerms + qTerms, newV))
                                sol = solve_dgcv(eqns, vars)
                                if len(sol) == 0:
                                    if not all(getattr(eqn, "is_zero", True) for eqn in eqns):
                                        raise RuntimeError(
                                            "solver failed; This is likely related to an unresolved known bug in the dgcv Levi decomposition algorithm. The following work-around sometimes works and will be available until the bug is fixed in a future dgcv patch: re-run Levi_decomposition with the optional keyword setting `_bust_cache=True`, i.e., run [algebra_class_instance].Levi_decomposition(_bust_cache=True). This clears the cached computations that an algebra_class instance stores, forcing many values to be re-computed. If the workaround fails on the first attempt then (surprisingly) it can still succeed on subsequent attempts. The root of this bug is that somewhere an un-ordered set is being processed by a solve algorithm in somewhat unpredictable ways. Repeating the method with _bust_cache=True seems to shuffle the processing ordering, which sometimes results in success."
                                        )
                                new_basis = [
                                    (w + v).subs(sol[0])
                                    for w, v in zip(naiveBasis, basis_modifiers)
                                ]
                                free_variables = set()
                                for nb in new_basis:
                                    free_variables |= set.union(
                                        *[getattr(j, "free_symbols", set()) for j in nb.coeffs]
                                    )
                                if len(free_variables) > 0:
                                    target = next(iter(free_variables))
                                    new_basis = [
                                        bv.subs({target: 1}).subs(
                                            {var: 0 for var in free_variables}
                                        )
                                        for bv in new_basis
                                    ]
                                naiveBasis = new_basis
                            Levi_component = self.subalgebra(naiveBasis, span_warning=True, simplify_basis=True)
                        else:
                            Levi_component=refAlg
                        refAlg._Levi_deco_cache = {'LD_components':(Levi_component, rad),'simple_ideals':None}
                if decompose_semisimple_fully is True and refAlg._Levi_deco_cache.get('LD_components', None) is not None and refAlg._Levi_deco_cache.get('simple_ideals',1) is None:
                    if verbose is True:
                        print('Decomposing semisimple subalgebra into simple subalgebras...')
                    Levi_component,rad=refAlg._Levi_deco_cache.get('LD_components', None)
                    simples = decompose_semisimple_algebra(Levi_component,format_as_lists_of_elements=True)
                    new_basis = []
                    simple_ideals = []
                    for comp in simples:
                        new_basis+=comp
                        simple_ideals.append(Levi_component.subalgebra(comp,simplify_basis=True))
                    Levi_component=Levi_component.subalgebra(new_basis)
                    refAlg._Levi_deco_cache['LD_components'] = (Levi_component, rad)
                    refAlg._Levi_deco_cache['simple_ideals'] = tuple(simple_ideals)
                return refAlg._Levi_deco_cache.get('LD_components', None)
            except Exception as e:
                last_err = e
                loop += 1
        raise RuntimeError(
            f"Levi_decomposition failed after {attempts} attempt(s)."
        ) from last_err

    @property
    def graded_components(self):
        if self._graded_components is None:
            gradings = sorted(list(set([tuple(j) for j in zip(*self.grading)])))
            gc = {}
            for key in gradings:
                gc[key] = self.weighted_component(key)
            self._graded_components=gc
        return self._graded_components

    def compute_graded_component_wrt_weight_index(self,idx=0):
        if idx not in range(len(self.grading)):
            warnings.warn('The provided index is out of range. `compute_graded_component_wrt_weight_index` is using 0 instead.')
            idx=0
        wc = dict()
        for idxs,comp in self.graded_components.items():
            wc[idxs[idx]]=wc.get(idxs[idx],self.subspace([]))+comp 
        return wc

    def subalgebra_from_grading_conditions(self, 
                                           callable_bool_condition, 
                                           sort_basis_by_grading_weights:bool=False,
                                           index_priority_for_lex_sort:list|tuple=[], 
                                           reverse_sort_order:bool=False, 
                                           label=None,
                                           basis_labels=None,
                                           register_in_vmf=False,
                                           initial_basis_index=None,
                                           simplify_products_by_default=None
                                           ):
        basis = [elem for elem in self.basis if callable_bool_condition(elem.check_element_weight()) is True]
        if sort_basis_by_grading_weights is True:
            grad_len=len(self.grading)
            idx_order = [j for j in index_priority_for_lex_sort if j<grad_len]
            if len(idx_order)<len(index_priority_for_lex_sort):
                warnings.warn('Some indices provided in the `index_priority_for_lex_sort` parameter were out of range, and ignored.')
            if len(idx_order)==0:
                idx_order = range(grad_len)
            basis = sorted(basis,key=lambda elem:[elem.check_element_weight()[idx] for idx in idx_order],reverse=reverse_sort_order)
        grad = list(zip(*[elem.check_element_weight() for elem in basis]))
        return self.new_alg_from_subalgebra(basis,
                                            grading=grad,
                                            span_warning=False,
                                            label=label,
                                            basis_labels=basis_labels,
                                            register_in_vmf=register_in_vmf,
                                            initial_basis_index=initial_basis_index,
                                            simplify_products_by_default=simplify_products_by_default,
                                            )

    def representation(self,rep_space=None,representation_basis=None,use_matrix_rep_instead_of_tensor=None):
        if rep_space is None:
            rep_space=self
        elif get_dgcv_category(rep_space) not in {'vector_space','algebra','subalgebra'}:
            raise('`rep_space` must be a `dgcv` class type representing a vector space or algebra.')
        if any(isinstance(elem,sp.Matrix) for elem in representation_basis):
            use_matrix_rep_instead_of_tensor=True
        if use_matrix_rep_instead_of_tensor is None and representation_basis is None:
            representation_basis = self.preferred_representation
            use_matrix_rep_instead_of_tensor = True if self._preferred_rep_type=='matrix' else False
        if use_matrix_rep_instead_of_tensor is True:
            if representation_basis is None:
                if isinstance(self._mat_rep,(list,tuple)):
                    representation_basis = self.matrix_representation
                elif self._preferred_rep_type=='matrix':
                    representation_basis = self.preferred_representation
                else:
                    representation_basis = adjointRepresentation(self)
            elif isinstance(representation_basis,(list,tuple)):
                if len(representation_basis)!=self.dimension:
                    raise TypeError('`representation_basis` should be a list of matrix/tensor elements matching the length of the represented algebra\'s basis.')
                for elem in representation_basis:
                    if not isinstance(elem, sp.Matrix):
                        raise TypeError(f'If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of matrices. But an element in the given list was of type {type(elem)}')
                    if elem.shape[0]!=elem.shape[1]:
                        raise TypeError(f'If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of square matrices. Recieved a matrix of shape {elem.shape}')
                    if rep_space.dimension!=elem.shape[0]:
                        raise TypeError(f'If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of (d,d) matrices where d is the dimension of the reprentation space (defaults to `self`). Recieved a matrix of shape {elem.shape} and rep. space of dimension {rep_space.dimension}')
            t_rep = [_mat_to_tensor(j,rep_space.dual(),rep_space) for j in representation_basis]
        else:
            if representation_basis is None:
                if isinstance(self._tensor_rep,(list,tuple)):
                    representation_basis = self.tensor_representation
                elif self._preferred_rep_type=='tensor':
                    representation_basis = self.preferred_representation
                else:
                    raise TypeError('`representation_basis` was not provided and no cached representation is currently stored in the algebra to fall back to.')
            if len(representation_basis)!=self.dimension:
                raise TypeError('`representation_basis` should be a list of matrix/tensor elements matching the length of the represented algebra\'s basis.')
            for elem in representation_basis:
                if not get_dgcv_category(elem)=="tensorProduct":
                    raise TypeError(f'If not setting `representation_basis` to a list of matrices or setting `use_matrix_rep_instead_of_tensor==True` then `representation_basis` should be a list of tensor products. But an element in the given list was of type {type(elem)}')
            t_rep = representation_basis
        hom = homomorphism(self,[rep_space,rep_space.dual()],t_rep)
        return linear_representation(hom)

    def structure_equations(self,variable_label:str=None,abreviate_for_skew_struct:bool=True,initial_index:int=1):
        if variable_label is None:
            variable_label='x'
        str_str="["
        for c1,e1 in enumerate(self.basis):
            start=c1+1 if self.is_skew_symmetric and abreviate_for_skew_struct else 0
            for pre_c2, e2 in enumerate(self.basis[start:]):
                c2=pre_c2+start
                prod = e1*e2
                prod_str = ""
                for idx,c in enumerate(prod.coeffs):
                    if c==0:
                        continue
                    if prod_str=="":
                        if c==1:
                            prod_str=f"{variable_label}{idx+initial_index}"
                        elif c==-1:
                            prod_str=f"-{variable_label}{idx+initial_index}"
                        else:
                            prod_str=f"{c}*{variable_label}{idx+initial_index}"
                    elif isinstance(c<0,bool) and c<0:
                        if c==-1:
                            prod_str+=f"{variable_label}{idx+initial_index}"
                        else:
                            prod_str+=f"{c}*{variable_label}{idx+initial_index}"
                    else:
                        if c==1:
                            prod_str+=f"+{variable_label}{idx+initial_index}"
                        else:
                            prod_str+=f"+{c}*{variable_label}{idx+initial_index}"
                if prod_str:
                    str_str +=f'({variable_label}{c1+initial_index},{variable_label}{c2+initial_index}):{prod_str}, '
        if len(str_str)>1:
            str_str=str_str[:-2]+"]"
        else:
            str_str+="]"
        return str_str

    def grading_summary(self):
        from .._dgcv_display import show

        gradingNumber = len(self.grading)
        graded_components = self.graded_components
        # gradings = sorted(list(set([tuple(j) for j in zip(*self.grading)])))
        # graded_components = {}
        # for key in gradings:
        #     graded_components[key] = self.weighted_component(key)
        pref = self._repr_latex_(abbrev=True).replace("$", "")
        if "_" in pref:
            prefi = f"\\left({pref} \\right)_"
        else:
            prefi = f"{pref}_"
        strings = []
        for k, v in graded_components.items():
            inner = ''.join(str(j) for j in k)
            latex = v._repr_latex_().replace('$', '').replace(r'\displaystyle', '')
            strings.append(f"$$ {prefi}{{{inner}}} = {latex},$$")
        if len(strings) > 1:
            strings.insert(-1, "and")
        strings[-1] = strings[-1][:-3] + ".$$"
        if gradingNumber == 0:
            show(f"The algebra ${pref}$ has no assigend grading.")
        else:
            if gradingNumber == 1:
                gradPhrase = "graded"
            elif gradingNumber == 2:
                gradPhrase = "bi-graded"
            elif gradingNumber == 3:
                gradPhrase = "tri-graded"
            else:
                gradPhrase = f"{gradingNumber}-graded"
            show(
                f"The algebra ${pref}$ has {gradPhrase} components: {' '.join(strings)}"
            )

    def direct_sum(self,other,grading=None,label=None,basis_labels=None,register_in_vmf=False,initial_basis_index=None,simplify_products_by_default=None,build_all_gradings=False):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}:
            _markers={'sum':True,'lockKey':retrieve_passkey()}
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0]*self.dimension]
                grad2 = other.grading[:1] or [[0]*other.dimension]
            else:
                grad1 = self.grading or [[0]*self.dimension]
                grad2 = other.grading or [[0]*other.dimension]
            builtG=[]
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append(list(gl1)+list(gl2))
            if not isinstance(grading,(list,tuple)):
                grading=[]
            if isinstance(grading, (list,tuple)):
                if all(isinstance(elem,(list,tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading]+builtG
                elif all(isinstance(elem,_get_expr_num_types()) for elem in grading):
                    grading = [list(grading)]+builtG
                elif grading is not None:
                    warnings.warn('The supplied grading data format is incompatible, and was ignored.')
                    grading=builtG
                else:
                    grading=builtG


            if label is None:
                label = f'{self.label}_plus_{other.label}'
                _markers['_tex_label']=f'{self._repr_latex_(raw=True,abbrev=True)}\\oplus {other._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.basis]+[elem.__repr__() for elem in other.basis]
                _markers['_tex_basis_labels']=[elem._repr_latex_(raw=True) for elem in self.basis]+[elem._repr_latex_(raw=True) for elem in other.basis]

            return linear_representation(homomorphism(self,other.endomorphism_algebra)).semidirect_sum(grading=grading,label=label,basis_labels=basis_labels,register_in_vmf=register_in_vmf,initial_basis_index=initial_basis_index,simplify_products_by_default=simplify_products_by_default,_markers=_markers)
        else: 
            return NotImplemented

    def __add__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}:
            return self.direct_sum(other)
        return NotImplemented

    def __radd__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        return NotImplemented

    def tensor_product(self,other,grading=None,label=None,basis_labels=None,register_in_vmf=False,initial_basis_index=None,simplify_products_by_default=None,build_all_gradings=False):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}: 
            if simplify_products_by_default is None:
                simplify_products_by_default = getattr(self,'simplify_products_by_default',False)
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0]*self.dimension]
                grad2 = other.grading[:1] or [[0]*other.dimension]
            else:
                grad1 = self.grading or [[0]*self.dimension]
                grad2 = other.grading or [[0]*other.dimension]
            builtG=[]
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append([w1+w2 for w1 in gl1 for w2 in gl2])
            if not isinstance(grading,(list,tuple)):
                grading=[]
            if isinstance(grading, (list,tuple)):
                if all(isinstance(elem,(list,tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading]+builtG
                elif all(isinstance(elem,_get_expr_num_types()) for elem in grading):
                    grading = [list(grading)]+builtG
                elif grading is not None:
                    warnings.warn('The supplied grading data format is incompatible, and was ignored.')
                    grading=builtG
                else:
                    grading=builtG

            if isinstance(basis_labels, (tuple, list)):
                if (
                    not all(isinstance(elem, str) for elem in basis_labels)
                    or len(basis_labels) != self.dimension*other.dimension
                ):
                    warnings.warn(
                        f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers={'prod':True,'lockKey':retrieve_passkey(),'tensor_decomposition':(self,other)}
            if label is None:
                label = f'{self.label}_tensor_{other.label}'
                _markers['_tex_label']=f'{self._repr_latex_(raw=True,abbrev=True)}\\otimes {other._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None or not isinstance(basis_labels,str):
                basis_labels = [f'{elem1.__repr__()}_tensor_{elem2.__repr__()}' for elem1 in self.basis for elem2 in other.basis]
                _markers['_tex_basis_labels']=[f'{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}' for elem1 in self.basis for elem2 in other.basis]
            if isinstance(basis_labels,str):
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [f"{pref}{i+IIdx}" for i in range(self.dimension*other.dimension)]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .algebras_secondary import createAlgebra
                return createAlgebra(
                    self.dimension*other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_obj=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers
                )
            else:
                _markers['registered'] = False
                return algebra_class(
                    self.dimension*other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers
                )
        elif isinstance(other,_get_expr_num_types()):
            return self._convert_to_tp().__matmul__(other)
        else:
            return NotImplemented

    def __matmul__(self,other):
        return self.tensor_product(other)

    def __rmatmul__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return algebra_class({})
        if isinstance(other,_get_expr_num_types()):
            return self._convert_to_tp().__rmatmul__(other)

    def dual(self,invert_grad_weights=True):
        return algebra_dual(self,invert_grad_weights=invert_grad_weights)

    def approximate_rank(self,check_semisimple=False,assume_semisimple=False,_use_cache=False,from_subalg=None):
        if get_dgcv_category(from_subalg) == "subalgebra":
            refAlg = from_subalg
        else:
            refAlg = self
        if refAlg.dimension==0:
            refAlg._rank_approximation=0
            return 0
        if check_semisimple is True:
            ssc=refAlg.is_semisimple()
            if ssc is True:
                assume_semisimple = True
            elif assume_semisimple is True:
                print('approximate_rank recieved parameters `check_semisimple=True` and `assume_semisimple=True`, but the semisimple check returned false. The algorithm is proceeding with the `assume_semisimple` logic applied, but this is likely not wanted, and should be prevented by setting those parameters differently. Note, just setting `check_semisimple=True` is enough to use optimized algorithms in the event that the semisimple check returns true, whereas `assume_semisimple` should only be used in applications where forgoing the semisimple check entirely is wanted.')
        if _use_cache and refAlg._rank_approximation is not None:
            return refAlg._rank_approximation
        power=1 if (assume_semisimple or refAlg._is_semisimple_cache is True) else refAlg.dimension
        elem = sp.Matrix(refAlg.structureData[0])    # test element
        bound=max(100,10*refAlg.dimension)
        for elem2 in refAlg.structureData[1:]:
            elem+=random.randint(0,bound)*sp.Matrix(elem2)
        rank = refAlg.dimension-fast_rank(elem**power)
        if not isinstance(refAlg._rank_approximation,numbers.Integral) or refAlg._rank_approximation>rank:
            refAlg._rank_approximation=rank
        return refAlg._rank_approximation

    def summary(self, generate_full_report=False, style=None, use_latex=None, _from_subalg=None, _IL=None):
        dgcvSR = get_dgcv_settings_registry()
        _apply_VScode_display_workaround_with_JS_deliver = bool(
            dgcvSR.get("apply_awkward_workarounds_to_fix_VSCode_display_issues")
        )
        if get_dgcv_category(_from_subalg) == "subalgebra" and _IL==retrieve_passkey():
            subAlg=True
            refAlg=_from_subalg
        else:
            subAlg=False
            refAlg=self
        parentAlg=self
        if style is None:
            style = dgcvSR['theme']
        if use_latex is None:
            use_latex = dgcvSR['use_latex']

        loc_style = get_style(style)

        if generate_full_report:
            print("Progress updates log while introspecting properties (honoring `generate_full_report=True`):")
            refAlg.is_Lie_algebra(verbose=True)
            if subAlg:
                print('The Levi decomposition algorithm is not fully supported for subalgebras yet and is being omitted from the current report. Suggestion: convert the subalgebra to an algebra_class with its `copy` method.')
            else:
                refAlg.compute_simple_subalgebras(verbose=True)
            if not refAlg.is_abelian():
                if refAlg.is_semisimple(verbose=True):
                    refAlg.is_simple(verbose=True)
                elif refAlg.is_solvable(verbose=True):
                    refAlg.is_nilpotent(verbose=True)
            print('Computing ranks of simple subalgebras (with non-deterministic algorithm that works almost always)...')
            if not subAlg:
                for alg in refAlg._Levi_deco_cache['simple_ideals']:
                    alg.approximate_rank(assume_semisimple=True)
                print('Computing derived series of the maximal solvable ideal...')
                refAlg._Levi_deco_cache['LD_components'][1].derived_series()
                print('Computing the lower central series of the maximal solvable ideal...')
                refAlg._Levi_deco_cache['LD_components'][1].lower_central_series()

        if use_latex:
            try:
                algebra_name = f'${parentAlg._repr_latex_(abbrev=True, raw=True)}$'
                algebra_name_cap=algebra_name
            except Exception:
                algebra_name = parentAlg.label if getattr(parentAlg, "label", None) else "the algebra"
                algebra_name_cap=parentAlg.label if getattr(parentAlg, "label", None) else "The algebra"
        else:
            algebra_name = parentAlg.label if getattr(parentAlg, "label", None) else "the algebra"
            algebra_name_cap = parentAlg.label if getattr(parentAlg, "label", None) else "The algebra"

        class _HTMLWrapper:
            def __init__(self, html): self._html = html
            def to_html(self, *args, **kwargs): return self._html
            def _repr_html_(self): return self._html

        def _stack_many(blocks, container_id: str = "dgcv-alg-summary") -> str:
            inner = "\n".join(f'<div class="section">{b}</div>' for b in blocks)
            return f"""
    <div id="{container_id}">
    <style>
        #{container_id} .stack {{
        display: flex; flex-direction: column; gap: 16px;
        align-items: stretch; width: 100%; margin: 0;
        }}
        #{container_id} .section {{ width: 100%; }}
        #{container_id} .section table {{ width: 100%; table-layout: fixed; }}
    </style>
    <div class="stack">
        {inner}
    </div>
    </div>
    """.strip()

        def _get_prop(sel, prop):
            for sd in loc_style:
                if sd.get("selector") == sel:
                    for k, v in sd.get("props", []):
                        if k == prop:
                            return v
            return None

        header_bg  = _get_prop("thead th", "background-color") or _get_prop("th.col_heading.level0", "background-color")
        header_col = _get_prop("thead th", "color")            or _get_prop("th.col_heading.level0", "color")
        header_ff  = _get_prop("thead th", "font-family")      or _get_prop("th.col_heading.level0", "font-family")
        header_fs  = _get_prop("thead th", "font-size")        or _get_prop("th.col_heading.level0", "font-size")
        col_heading_color = _get_prop("th.col_heading.level0", "color")
        col_heading_ff    = _get_prop("th.col_heading.level0", "font-family")
        col_heading_fs    = _get_prop("th.col_heading.level0", "font-size")
        col_heading_bg    = _get_prop("th.col_heading.level0", "background-color")

        border_val = None
        for sd in loc_style:
            if sd.get("selector") == "table":
                for k, v in sd.get("props", []):
                    if k in ("border-bottom","border-right","border-left","border-top","border"):
                        border_val = v
                        break
            if border_val: 
                break
        parts = (border_val or "1px solid #ccc").split()
        thickness = parts[0] if parts else "1px"
        border_color = parts[-1] if parts else "#ccc"
        header_border_raw = (
            _get_prop("thead th", "border")
            or _get_prop("th.col_heading.level0", "border")
        )

        if header_border_raw:
            hb_parts = header_border_raw.split()
            header_base_thickness = hb_parts[0]
            header_border_color = hb_parts[-1]
        else:
            header_base_thickness = None
            header_border_color = border_color

        if header_base_thickness and border_val is None:
            m = re.match(r"(\d*\.?\d+)(px)$", header_base_thickness)
            if m:
                val = float(m.group(1))
                unit = m.group(2)
                header_thickness_for_bottom = f"{2*val}{unit}"
            else:
                header_thickness_for_bottom = header_base_thickness
        else:
            header_thickness_for_bottom = header_base_thickness or thickness

        def _panel_extra():
            return [
                {
                    "selector": ".dgcv-panel",
                    "props": [
                        ("border", f"{thickness} solid {border_color}"),
                        ("background-color", col_heading_bg or header_bg or "transparent"),
                        ("color", header_col or "inherit"),
                        ("padding", "4px 4px"),
                        ("margin", "0"),
                        ("overflow-y", "visible"),
                    ],
                },
                {
                    "selector": ".dgcv-panel-body",
                    "props": [
                        ("overflow-x", "auto"),
                        ("overflow-y", "visible"),
                        ("width", "100%"),
                        ("box-sizing", "border-box"),
                    ],
                },
                {"selector": ".dgcv-panel *", "props": [("color", header_col or "inherit")]},
                {"selector": ".dgcv-panel h3", "props": [
                    ("margin", "0"),
                    ("color", col_heading_color or header_col or "inherit"),
                    ("font-family", col_heading_ff or header_ff or "inherit"),
                    ("font-size", col_heading_fs or header_fs or "inherit"),
                    ("font-weight", "bold"),
                ]},
                {"selector": ".dgcv-panel hr", "props": [
                    ("border", "0"),
                    ("border-top", f"{thickness} solid {border_color}"),
                    ("margin", "6px 0 8px"),
                ]},
                {"selector": ".dgcv-panel ul", "props": [("margin", "8px 0 0 18px"), ("padding", "0")]},
                {"selector": ".dgcv-panel li::marker", "props": [("color", col_heading_color or header_col or border_color)]},
            ]

        caption_ff = _get_prop("th.col_heading.level0", "font-family") or _get_prop("thead th", "font-family") or "inherit"
        caption_fs = _get_prop("th.col_heading.level0", "font-size")   or _get_prop("thead th", "font-size")   or "inherit"

        def _table_extra():
            return [
                {"selector": "table","props": [("border-collapse", "collapse"),("width", "100%"),("table-layout", "fixed")]},
                {"selector": "td", "props": [("text-align", "left")]},
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "thead th.col_heading.level0","props": [("border-bottom", f"{header_thickness_for_bottom} solid {header_border_color}")]},
                {"selector": "tbody tr:first-child td, tbody tr:first-child th","props": [("border-top", "0")]},
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("text-align", "left"),
                        ("margin", "0 0 6px 0"),
                        ("font-family", caption_ff),
                        ("font-size", caption_fs),
                        ("font-weight", "bold"),
                    ],
                },
                {"selector": ".dgcv-table-wrap","props": [("overflow-x", "visible"),("max-width", "100%"),("width", "100%")]},
                {"selector": ".dgcv-table-wrap > table.dgcv-data-table","props": [("min-width", "40rem")]},
            ]

        def _corners_for(i: int, total: int):
            if total <= 1:
                return {}
            if i == 0:
                return {"lr": 0, "ll": 0}
            if i == total - 1:
                return {"ur": 0, "ul": 0}
            return {"ur": 0, "ul": 0, "lr": 0, "ll": 0}

        items = [f"Subalgebra contained in {algebra_name}",f"Dimension: {refAlg.dimension}"] if subAlg else [f"Dimension: {refAlg.dimension}"]
        lie = getattr(refAlg, "_lie_algebra_cache", None)
        if lie is True:
            items.append("Lie algebra: true")
            special_property=getattr(refAlg,'_educed_properties', dict()).get('special_type',None)
            if special_property is not None:
                items.append(f'special properties: {special_property}')
            elif getattr(refAlg,'_is_semisimple_cache', None) is False and getattr(refAlg,'_is_solvable_cache', None) is False:
                items.append('special properties: neither solvable nor semisimple')
            else:
                items.append('special properties: not yet evaluated')
        elif lie is False:
            items.append("Lie algebra: false")
            def _fmt(v): return "true" if v is True else ("false" if v is False else "not yet evaluated")
            items.append(f"Skew symmetric: {_fmt(getattr(refAlg, '_skew_symmetric_cache', 'not yet evaluated'))}")
            items.append(f"Jacobi identity satisfied: {_fmt(getattr(refAlg, '_jacobi_identity_cache', 'not yet evaluated'))}")
        else:
            items.append("Lie algebra: not yet evaluated")

        if refAlg.dimension == 0:
            pv0 = panel_view(
                header="Basic properties of the subalgebra" if subAlg else f"Basic properties of {algebra_name}",
                itemized_text=[f"This is the trivial 0-dimensional subalgebra in {algebra_name}."] if subAlg else ["This is the trivial 0-dimensional algebra."],
                theme_styles=loc_style,
                extra_styles=_panel_extra(),
            ).to_html()
            return latex_in_html(_HTMLWrapper(_stack_many([pv0])), apply_VSCode_workarounds=_apply_VScode_display_workaround_with_JS_deliver)

        basis_elems = getattr(refAlg, "basis", ())
        if use_latex:
            try:
                basis_labels = [f'${b._repr_latex_(raw=True)}$' for b in basis_elems]
            except Exception:
                basis_labels = [repr(b) for b in basis_elems]
        else:
            basis_labels = [repr(b) for b in basis_elems]

        rows = []
        grad_index_labels = []
        rows.append(list(basis_labels))
        grad_index_labels.append("Basis")

        warn_msgs = []
        grad = getattr(refAlg, "grading", None)
        def _fmt_weight(x):
            if use_latex and hasattr(x, "_repr_latex_"):
                try:
                    s = x._repr_latex_()
                    if s.startswith("$") and s.endswith("$"):
                        s = s[1:-1]
                    s = s.replace(r"\displaystyle", "").replace(r"\\displaystyle", "").strip()
                    return f"${s}$"
                except Exception:
                    pass
            return str(x)
        if isinstance(grad, (list, tuple)) and grad:
            for gi, g in enumerate(grad, start=1):
                if isinstance(g, (list, tuple)) and len(g) == len(basis_labels):
                    rows.append([_fmt_weight(x) for x in g])
                    grad_index_labels.append(f"Grading {gi}")
                else:
                    warn_msgs.append(f"grading {gi} invalid or length mismatch")
        if len(basis_labels) != refAlg.dimension:
            warn_msgs.append(f"dimension {refAlg.dimension} does not match basis length {len(basis_labels)}")

        footer_rows = None
        if warn_msgs:
            msg = " | ".join(warn_msgs)
            footer_rows = [[{"html": f"<em>{_esc(msg)}</em>", "attrs": {"colspan": len(basis_labels)}}]]

        sections = []

        # panel: Basic properties panel
        def _build_basic_panel(corner_kwargs):
            return panel_view(
                header="Basic properties of the subalgebra" if subAlg else f"Basic properties of {algebra_name}",
                itemized_text=items,
                theme_styles=loc_style,
                extra_styles=_panel_extra(),
                **corner_kwargs
            ).to_html()
        sections.append(("panel", _build_basic_panel))

        # panel: Basis and gradings table
        def _build_basis_panel(corner_kwargs):
            table_view = build_matrix_table(
                show_headers=False,
                index_labels=grad_index_labels,
                columns=[],
                rows=rows,
                caption="",
                theme_styles=loc_style,
                extra_styles=_table_extra(),
                footer_rows=footer_rows,
                table_attrs='style="table-layout:auto;"',
                cell_align=None,
                escape_cells=False,
                escape_headers=True,
                nowrap=False,
                truncate_chars=None,
                dashed_corner=False,
            )

            return panel_view(
                header="Basis and assigned grading(s)",
                primary_text=table_view,
                itemized_text=None,
                theme_styles=loc_style,
                extra_styles=_panel_extra(),
                **corner_kwargs,
            ).to_html()

        sections.append(("panel", _build_basis_panel))

        if refAlg._lie_algebra_cache is True and refAlg._Levi_deco_cache is not None:
            simples=refAlg._Levi_deco_cache.get('simple_ideals',None)
            def _LD_panel(corner_kwargs):
                IT=[]
                if refAlg.is_solvable():
                    if subAlg is True:
                        PT = "The subalgebra equals its own maximal solvable ideal."
                    else:
                        PT = f"{algebra_name_cap} equals its own maximal solvable ideal."
                elif refAlg.is_semisimple():
                    if simples is None:
                        if subAlg is True:
                            PT = "The subalgebra is semisimple and the number of simple ideals has not been evaluated yet."
                        else:
                            PT = f"{algebra_name_cap} is semisimple and the number of simple ideals has not been evaluated yet."
                    elif len(simples)==1:
                        if subAlg is True:
                            PT = "The subalgebra is simple."
                        else:
                            PT = f"{algebra_name_cap} is simple."
                    else:
                        if subAlg is True:
                            PT = "The subalgebra is a direct sum of the following simple ideals:"
                        else:
                            PT = f"{algebra_name_cap} is a direct sum of the following simple ideals:"
                        for alg in simples:
                            label = f'${alg._repr_latex_(raw=True,abbrev=True)}$' if use_latex else alg.__repr__()
                            IT.append(label)
                else:
                    if subAlg is True:
                        PT="The subalgebra is a semidirect sum of the following (respectively) semisimple and solvable subalgebras:"
                    else:
                        PT=f"{algebra_name_cap} is a semidirect sum of the following (respectively) semisimple and solvable subalgebras:"
                    for alg in refAlg._Levi_deco_cache['LD_components']:
                        label = f'${alg._repr_latex_(raw=True,abbrev=True)}$' if use_latex else alg.__repr__()
                        IT.append(label)

                return panel_view(
                    header="Levi decomposition of the subalgebra" if subAlg else f"Levi decomposition of {algebra_name}",
                    primary_text=PT,
                    itemized_text=IT,
                    theme_styles=loc_style,
                    extra_styles=_panel_extra(),
                    **corner_kwargs
                ).to_html()
            sections.append(("panel", _LD_panel))
            if refAlg._Levi_deco_cache['LD_components'][0].dimension !=0 and simples is not None:
                def _ss_compl_panel(corner_kwargs):
                    cols = ['Ideal #','Dimension','Rank','Iso. Class','Basis']
                    rows= []
                    for idx,alg in enumerate(simples):
                        if use_latex:
                            BL=', '.join([f'${elem._repr_latex_(raw=True)}$' for elem in alg.basis])
                        else:
                            BL=', '.join([f'{elem.__repr__()}' for elem in alg.basis])
                        rank=alg.approximate_rank(_use_cache=True,assume_semisimple=True)
                        dim=alg.dimension
                        if (rank+1)**2-1==dim:
                            IC = f'$\\mathfrak{{sl}}_{{{rank+1}}}$' if use_latex else f'A_{rank}'
                        elif (2*rank+1)*rank==dim:
                            IC = f'$\\mathfrak{{so}}_{{{2*rank+1}}}$ or $\\mathfrak{{sp}}_{{{2*rank}}}$' if use_latex else f'B_{rank} or C_{rank}'
                        elif (2*rank-1)*rank==dim:
                            IC = f'$\\mathfrak{{so}}_{{{2*rank}}}$' if use_latex else f'D_{rank}'
                        elif rank==2 and dim==14:
                            IC = '$\\operatorname{{Lie}}(G_2)$' if use_latex else 'G_2'
                        elif rank==4 and dim==52:
                            IC = '$\\operatorname{{Lie}}(F_4)$' if use_latex else 'F_4'
                        elif rank==6 and dim==78:
                            IC = '$\\operatorname{{Lie}}(E_6)$' if use_latex else 'E_6'
                        elif rank==7 and dim==133:
                            IC = '$\\operatorname{{Lie}}(E_7)$' if use_latex else 'E_7'
                        elif rank==8 and dim==248:
                            IC = '$\\operatorname{{Lie}}(E_8)$' if use_latex else 'E_8'

                        rows.append([f'subalgebra {idx+1}',f'{dim}',f'{rank}',IC,BL])

                    table_view = build_matrix_table(
                        index_labels=None,
                        columns=cols,
                        rows=rows,
                        caption="",  # header moves to panel
                        theme_styles=loc_style,
                        extra_styles=_table_extra(),
                        footer_rows=None,
                        table_attrs='style="table-layout:auto;"',
                        cell_align=None,
                        escape_cells=False,
                        escape_headers=True,
                        nowrap=False,
                        dashed_corner=False,
                        truncate_chars=None,
                    )
                    return panel_view(
                        header="Simple ideals in semisimple complement to the max. solvable ideal.",
                        primary_text=table_view,
                        itemized_text=None,
                        theme_styles=loc_style,
                        extra_styles=_panel_extra(),
                        **corner_kwargs,
                    ).to_html()
                sections.append(("panel", _ss_compl_panel))

            if refAlg._Levi_deco_cache['LD_components'][1].dimension !=0 and refAlg._Levi_deco_cache['LD_components'][1]._lower_central_series_cache is not None:
                def _lcs_panel(corner_kwargs):
                    cols = ['Filtration Level (1 = top)','Dimension','Basis']
                    rows= []
                    for idx,alg in enumerate(refAlg._Levi_deco_cache['LD_components'][1]._lower_central_series_cache[0]):
                        alg=getattr(alg,'basis',alg)
                        if use_latex:
                            BL=', '.join([f'${elem._repr_latex_(raw=True)}$' for elem in alg])
                        else:
                            BL=', '.join([f'{elem.__repr__()}' for elem in alg])
                        dim = 0 if len(alg)==1 and getattr(alg[0],'is_zero',alg[0]==0) else len(alg)
                        rows.append([f'Level {idx+1}',f'{dim}',BL])

                    table_view = build_matrix_table(
                        index_labels=None,
                        columns=cols,
                        rows=rows,
                        caption="",
                        theme_styles=loc_style,
                        extra_styles=_table_extra(),
                        footer_rows=None,
                        table_attrs='style="table-layout:auto;"',
                        cell_align=None,
                        escape_cells=False,
                        escape_headers=True,
                        nowrap=False,
                        dashed_corner=False,
                        truncate_chars=None,
                    )

                    return panel_view(
                        header="Lower central series in the maximal solvable ideal.",
                        primary_text=table_view,
                        itemized_text=None,
                        theme_styles=loc_style,
                        extra_styles=_panel_extra(),
                        **corner_kwargs,
                    ).to_html()

                sections.append(("panel", _lcs_panel))

            if refAlg._Levi_deco_cache['LD_components'][1].dimension !=0 and refAlg._Levi_deco_cache['LD_components'][1]._derived_series_cache is not None:
                def _ds_panel(corner_kwargs):
                    cols = ['Filtration Level (1 = top)','Dimension','Basis']
                    rows= []
                    for idx,alg in enumerate(refAlg._Levi_deco_cache['LD_components'][1]._derived_series_cache[0]):
                        alg=getattr(alg,'basis',alg)
                        if use_latex:
                            BL=', '.join([f'${elem._repr_latex_(raw=True)}$' for elem in alg])
                        else:
                            BL=', '.join([f'{elem.__repr__()}' for elem in alg])
                        dim = 0 if len(alg)==1 and getattr(alg[0],'is_zero',alg[0]==0) else len(alg)
                        rows.append([f'Level {idx+1}',f'{dim}',BL])

                    table_view = build_matrix_table(
                        index_labels=None,
                        columns=cols,
                        rows=rows,
                        caption="",
                        theme_styles=loc_style,
                        extra_styles=_table_extra(),
                        footer_rows=None,
                        table_attrs='style="table-layout:auto;"',
                        cell_align=None,
                        escape_cells=False,
                        escape_headers=True,
                        nowrap=False,
                        dashed_corner=False,
                        truncate_chars=None,
                    )

                    return panel_view(
                        header="Derived series in the maximal solvable ideal.",
                        primary_text=table_view,
                        itemized_text=None,
                        theme_styles=loc_style,
                        extra_styles=_panel_extra(),
                        **corner_kwargs,
                    ).to_html()

                sections.append(("panel", _ds_panel))


        # # panel template: Placeholder panel example
        # def _build_new_panel(corner_kwargs):
        #     return panel_view(
        #         header="New panel (placeholder)",
        #         primary_text="Content to be filled in later.",
        #         itemized_text=[],
        #         theme_styles=loc_style,
        #         extra_styles=_panel_extra(),
        #         **corner_kwargs
        #     ).to_html()
        # sections.append(("panel", _build_new_panel))

        # # table template: Placeholder table example
        # def _build_new_table(corner_kwargs):
        #     return build_matrix_table(
        #         show_headers=False,
        #         index_labels=["placeholder"],
        #         columns=[],
        #         rows=[["—" for _ in range(max(1, len(basis_labels)))]],
        #         caption="New table (placeholder)",
        #         theme_styles=loc_style,
        #         extra_styles=_table_extra(),
        #         footer_rows=None,
        #         table_attrs='style="table-layout:auto; overflow-x:auto;"',
        #         cell_align=None,
        #         escape_cells=False,
        #         escape_headers=True,
        #         nowrap=False,
        #         truncate_chars=None,
        #         dashed_corner=False,
        #         **corner_kwargs
        #     ).to_html()
        # sections.append(("table", _build_new_table))

        built_blocks = []
        total = len(sections)
        for i, (_, builder) in enumerate(sections):
            corner_kwargs = _corners_for(i, total)
            built_blocks.append(builder(corner_kwargs))

        return latex_in_html(
            _HTMLWrapper(_stack_many(built_blocks)),
            apply_VSCode_workarounds=_apply_VScode_display_workaround_with_JS_deliver
        )

class algebra_dual:
    def __init__(self, alg, invert_grad_weights=True):
        object.__setattr__(self, "dual_algebra", alg)
        object.__setattr__(self, 'basis', tuple([elem.dual() for elem in alg.basis]))
        object.__setattr__(self, 'label', alg.label+'_dual')
        if invert_grad_weights is not False:
            object.__setattr__(self, 'grading', [tuple(-j for j in elem) for elem in alg.grading])
        object.__setattr__(self, '_terms', None)
        object.__setattr__(self, '_dgcv_categories', {"algebra_dual"})


    def __getattr__(self, name):
        return getattr(self.dual_algebra, name)

    def __setattr__(self, name, value):
        if name == "dual_algebra":
            object.__setattr__(self, name, value)
        else:
            setattr(self.dual_algebra, name, value)

    def __delattr__(self, name):
        if name == "dual_algebra":
            raise AttributeError("Cannot delete 'dual_algebra'")
        delattr(self.dual_algebra, name)

    def __dir__(self):
        # Merge proxy attributes with algebra_class attributes
        return sorted(set(dir(type(self))) | set(self.__dict__.keys()) | set(dir(self.dual_algebra)))

    def dual(self):
        return self.dual_algebra

    def __repr__(self):
        return f"algebra_dual({self.dual_algebra!r})"

    def _repr_latex_(self,raw=False,abbrev=False):
        texS=self.dual_algebra._repr_latex_(raw=True,abbrev=True)
        if '^' in texS:
            texS=f'\\left({texS}\\right)'
        texS=f'{texS}^{{*}}'
        if abbrev is True:
            return texS if raw else f'$\\displaystyle {texS}$'
        else:
            bL=r'=\langle '+', '.join([elem._repr_latex_(raw=True) for elem in self.basis])+r'\rangle'
            return texS+bL if raw else f'$\\displaystyle {texS+bL}$'

    def direct_sum(self,other,grading=None,label=None,basis_labels=None,register_in_vmf=False,initial_basis_index=None,simplify_products_by_default=None,build_all_gradings=False):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}:
            _markers={'sum':True,'lockKey':retrieve_passkey()}
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0]*self.dimension]
                grad2 = other.grading[:1] or [[0]*other.dimension]
            else:
                grad1 = self.grading or [[0]*self.dimension]
                grad2 = other.grading or [[0]*other.dimension]
            builtG=[]
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append(list(gl1)+list(gl2))
            if not isinstance(grading,(list,tuple)):
                grading=[]
            if isinstance(grading, (list,tuple)):
                if all(isinstance(elem,(list,tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading]+builtG
                elif all(isinstance(elem,_get_expr_num_types()) for elem in grading):
                    grading = [list(grading)]+builtG
                elif grading is not None:
                    warnings.warn('The supplied grading data format is incompatible, and was ignored.')
                    grading=builtG
                else:
                    grading=builtG


            if label is None:
                label = f'{self.label}_plus_{other.label}'
                _markers['_tex_label']=f'{self._repr_latex_(raw=True,abbrev=True)}\\oplus {other._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.basis]+[elem.__repr__() for elem in other.basis]
                _markers['_tex_basis_labels']=[elem._repr_latex_(raw=True) for elem in self.basis]+[elem._repr_latex_(raw=True) for elem in other.basis]

            return linear_representation(homomorphism(self,other.endomorphism_algebra)).semidirect_sum(grading=grading,label=label,basis_labels=basis_labels,register_in_vmf=register_in_vmf,initial_basis_index=initial_basis_index,simplify_products_by_default=simplify_products_by_default,_markers=_markers)
        else: 
            return NotImplemented

    def __add__(self,other):
        return self.direct_sum(other)

    def tensor_product(self,other,grading=None,label=None,basis_labels=None,register_in_vmf=False,initial_basis_index=None,simplify_products_by_default=None,build_all_gradings=False):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}: 
            if simplify_products_by_default is None:
                simplify_products_by_default = getattr(self,'simplify_products_by_default',False)
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0]*self.dimension]
                grad2 = other.grading[:1] or [[0]*other.dimension]
            else:
                grad1 = self.grading or [[0]*self.dimension]
                grad2 = other.grading or [[0]*other.dimension]
            builtG=[]
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append([w1+w2 for w1 in gl1 for w2 in gl2])
            if not isinstance(grading,(list,tuple)):
                grading=[]
            if isinstance(grading, (list,tuple)):
                if all(isinstance(elem,(list,tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading]+builtG
                elif all(isinstance(elem,_get_expr_num_types()) for elem in grading):
                    grading = [list(grading)]+builtG
                elif grading is not None:
                    warnings.warn('The supplied grading data format is incompatible, and was ignored.')
                    grading=builtG
                else:
                    grading=builtG

            if isinstance(basis_labels, (tuple, list)):
                if (
                    not all(isinstance(elem, str) for elem in basis_labels)
                    or len(basis_labels) != self.dimension*other.dimension
                ):
                    warnings.warn(
                        f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers={'prod':True,'lockKey':retrieve_passkey(),'tensor_decomposition':(self,other)}
            if label is None:
                label = f'{self.label}_tensor_{other.label}'
                _markers['_tex_label']=f'{self._repr_latex_(raw=True,abbrev=True)}\\otimes {other._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None or not isinstance(basis_labels,str):
                basis_labels = [f'{elem1.__repr__()}_tensor_{elem2.__repr__()}' for elem1 in self.basis for elem2 in other.basis]
                _markers['_tex_basis_labels']=[f'{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}' for elem1 in self.basis for elem2 in other.basis]
            if isinstance(basis_labels,str):
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [f"{pref}{i+IIdx}" for i in range(self.dimension*other.dimension)]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .algebras_secondary import createAlgebra
                return createAlgebra(
                    self.dimension*other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_obj=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers
                )
            else:
                _markers['registered'] = False
                return algebra_class(
                    self.dimension*other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers
                )

        else:
            return NotImplemented

    def __matmul__(self,other):
        return self.tensor_product(other)

    def terms(self):
        if self._terms is None:
            terms=[]
            for idx,c in enumerate(self.coeffs):
                if c==0:
                    continue
                terms.append(c*self.algebra.basis[idx])
            self._terms=[self] if len(terms)<2 else terms
        return self._terms

class algebra_element_class:
    def __init__(self, alg, coeffs, valence, format_sparse=False):
        if not isinstance(alg, algebra_class):
            raise TypeError(
                "`algebra_element_class` expects the first argument to be an instance of the `algebra` class."
            ) from None
        if valence not in {0, 1}:
            raise TypeError(
                "vector_space_element expects third argument to be 0 or 1."
            ) from None
        coeffs = tuple(coeffs)
        if len(coeffs)<alg.dimension:
            coeffs = tuple(coeffs[j] if j<len(coeffs) else 0 for j in range(alg.dimension))
        self.algebra = alg
        self.vectorSpace = alg
        self.valence = valence
        self.is_sparse = format_sparse
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra_element"
        self.dgcv_vs_id=self.vectorSpace.dgcv_vs_id
        if isinstance(coeffs, (list, tuple)):
            self.coeffs = tuple(coeffs)
        else:
            raise TypeError(
                "algebra_element_class expects coeffs to be a list or tuple."
            ) from None
        self._tensor_rep = None
        self._natural_weight = None

    @property
    def tensor_representation(self):
        if self._tensor_rep is None and self.algebra.tensor_representation is not None:
            self._tensor_rep = sum([c*tp for c,tp in zip(self.coeffs,self.algebra.tensor_representation)],0*self.algebra.tensor_representation[0])
        return self._tensor_rep


    def __eq__(self, other):
        if not isinstance(other, algebra_element_class):
            return NotImplemented
        return (
            self.algebra == other.algebra
            and self.coeffs == other.coeffs
            and self.valence == other.valence
            and self.is_sparse == other.is_sparse
        )

    def __hash__(self):
        return hash((self.algebra, self.coeffs, self.valence, self.is_sparse))

    def __str__(self):
        if self.algebra.basis_labels is None:
            # Fallback to __str__ when basis_labels is None
            return 'elem'

        terms = []
        for coeff, basis_label in zip(self.coeffs, self.algebra.basis_labels):
            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence == 1:
                    terms.append(f"{basis_label}")
                else:
                    terms.append(f"{basis_label}^''")
            elif coeff == -1:
                if self.valence == 1:
                    terms.append(f"-{basis_label}")
                else:
                    terms.append(f"-{basis_label}^''")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(f"({coeff}) * {basis_label}")
                    else:
                        terms.append(f"({coeff}) * {basis_label}^''")
                else:
                    if self.valence == 1:
                        terms.append(f"{coeff} * {basis_label}")
                    else:
                        terms.append(f"{coeff} * {basis_label}^''")
        if not terms:
            return "0"
        return " + ".join(terms).replace("+ -", "- ")

    def _class_builder(self, coeffs, valence, format_sparse=False):
        return algebra_element_class(
            self.algebra, coeffs, valence, format_sparse=format_sparse
        )

    def _repr_latex_(self, verbose=False, raw=False):
        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                warnings.warn(self.algebra._child_print_warning, UserWarning)
            else:
                warnings.warn(
                    "This algebra_element_class's parent vector space (algebra_class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra_class` objects with dgcv creator functions like `createAlgebra` instead.",
                    UserWarning,
                )

        terms = []

        if self.algebra._tex_basis_labels is not None:
            llist = self.algebra._tex_basis_labels
            proDone = True
        elif self.algebra.basis_labels is not None:
            llist = self.algebra.basis_labels
            proDone = False
        else:
            llist = [f"_e_{{{i+1}}}" for i in range(self.algebra.dimension)]
            proDone = False

        def latexify_base(base: str) -> str:
            if not base:
                return base
            if "_" not in base and base and base[-1].isdigit():
                m = re.match(r"^(.+?)(\d+)$", base)
                if m:
                    name, digits = m.groups()
                    return f"{name}_{{{digits}}}"
            if base.startswith("_"):
                return r"\_" + format_latex_subscripts(base[1:])
            return format_latex_subscripts(base)

        def _format_basis_label_with_hi_low(basis_label: str) -> str:
            if not basis_label:
                return basis_label

            decoration_cmd = None
            core = basis_label
            ###!!! Modularize
            prefix_map = {
                "tilde_": r"\tilde",
                "hat_": r"\hat",
                "widehat_": r"\widehat",
                "bar_": r"\bar",
                "overline_": r"\overline",
                "overline": r"\overline",
                "bar": r"\bar",
            }
            for prefix in sorted(prefix_map, key=len, reverse=True):
                if core.startswith(prefix):
                    decoration_cmd = prefix_map[prefix]
                    core = core[len(prefix):]
                    break

            if "_low_" not in core and "_hi_" not in core:
                out = latexify_base(core)
                if decoration_cmd is not None:
                    out = f"{decoration_cmd}{{{out}}}"
                return out

            idx_low = core.find("_low_")
            idx_hi = core.find("_hi_")
            index_start_candidates = [i for i in (idx_low, idx_hi) if i != -1]
            if index_start_candidates:
                index_start = min(index_start_candidates)
                base_part = core[:index_start]
                index_part = core[index_start:]
            else:
                base_part = core
                index_part = ""

            base_tex = latexify_base(base_part)

            lower_indices = []
            upper_indices = []
            primes_count = 0

            upper_part = ""
            if "_low_" in index_part:
                lower_part = index_part.split("_low_")[1]
                if "_hi_" in lower_part:
                    lower_part, upper_part = lower_part.split("_hi_", 1)
                lower_indices = [tok for tok in lower_part.split("_") if tok]
            if upper_part == "" and "_hi_" in index_part:
                upper_part = index_part.split("_hi_")[1]

            if upper_part:
                tokens = [t for t in upper_part.split("_") if t]
                while tokens and re.fullmatch(r"p+", tokens[0]):
                    primes_count += len(tokens[0])
                    tokens.pop(0)
                upper_indices = tokens

            prime_tex = r"\prime" * primes_count
            if upper_indices:
                if prime_tex:
                    upper_tex = prime_tex + "," + ",".join(upper_indices)
                else:
                    upper_tex = ",".join(upper_indices)
            else:
                upper_tex = prime_tex

            indices_str = ""
            if upper_tex:
                indices_str += f"^{{{upper_tex}}}"
            if lower_indices:
                indices_str += f"_{{{','.join(lower_indices)}}}"

            out = base_tex + indices_str

            if decoration_cmd is not None:
                out = f"{decoration_cmd}{{{out}}}"

            return out

        def starred_label(label: str) -> str:
            if "^" in label:
                return rf"\left({label}\right)^*"
            return label + "^*"

        for coeff, basis_label in zip(self.coeffs, llist):
            labelProc = proDone

            if proDone is False and self.algebra._basis_labels_parent is True:
                match = re.search(r"(_\d+)$", basis_label)
                if match:
                    tail = f"_{{{match.group(1)[1:]}}}"
                    basis_label = (
                        f"\\left({self.algebra._repr_latex_(abbrev=True, raw=True)}\\right)"
                        + tail
                    )
                    labelProc = True

            if labelProc is False:
                basis_label = _format_basis_label_with_hi_low(basis_label)

            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence == 1:
                    terms.append(rf"{basis_label}")
                else:
                    terms.append(starred_label(basis_label))
            elif coeff == -1:
                if self.valence == 1:
                    terms.append(rf"-{basis_label}")
                else:
                    terms.append(rf"-{starred_label(basis_label)}")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}")
                    else:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {starred_label(basis_label)}")
                else:
                    if self.valence == 1:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}")
                    else:
                        terms.append(rf"{sp.latex(coeff)} \cdot {starred_label(basis_label)}")

        if not terms:
            if verbose:
                return rf"$0 \cdot {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}$"
            else:
                return "0" if raw else "$0$"

        result = " + ".join(terms).replace("+ -", "- ")

        if get_dgcv_settings_registry()["verbose_label_printing"] is False:
            m = get_dgcv_settings_registry()["VLP"].match(result)
            if not m:
                pass
            elif m.group("content") is not None:
                result = m.group("content")
            else:
                result = result[: result.rfind("_{\\operatorname{v.}")]

        return result if raw else rf"$\displaystyle {result}$"

    def _latex(self, printer=None):
        return self._repr_latex_()

    def _sympystr(self):
        """
        SymPy string representation for algebra_element_class.
        Handles unregistered parent algebra by raising a warning.
        """
        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                warnings.warn(
                    self.algebra._child_print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra_element_class's parent algebra (`algebra` class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra` class objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        coeffs_str = ", ".join(map(str, self.coeffs))
        if self.algebra.label:
            return f"algebra_element_class({self.algebra.label}, coeffs=[{coeffs_str}])"
        else:
            return f"algebra_element_class(coeffs=[{coeffs_str}])"

    def _latex_verbose(self, printer=None):
        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                warnings.warn(
                    self.algebra._child_print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra_element_class's parent vector space (an `algebra` class instance) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra` class objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        terms = []
        for coeff, basis_label in zip(
            self.coeffs,
            self.algebra.basis_labels
            or [f"e_{i+1}" for i in range(self.algebra.dimension)],
        ):
            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence == 1:
                    terms.append(rf"{basis_label}")
                else:
                    terms.append(rf"{basis_label}^*")
            elif coeff == -1:
                if self.valence == 1:
                    terms.append(rf"-{basis_label}")
                else:
                    terms.append(rf"-{basis_label}^*")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}")
                    else:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}^*")
                else:
                    if self.valence == 1:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}")
                    else:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}^*")

        if not terms:
            return rf"0 \cdot {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}"

        result = " + ".join(terms).replace("+ -", "- ")

        def format_algebra_label(label):
            r"""
            Wrap the vector space label in \mathfrak{} if lowercase, and add subscripts for numeric suffixes or parts.
            """
            if "_" in label:
                main_part, subscript_part = label.split("_", 1)
                if main_part.islower():
                    return rf"\mathfrak{{{main_part}}}_{{{subscript_part}}}"
                return rf"{main_part}_{{{subscript_part}}}"
            elif label[-1].isdigit():
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label.islower():
                return rf"\mathfrak{{{label}}}"
            return label

        return rf"\text{{Element of }} {format_algebra_label(self.algebra.label)}: {result}"

    def __repr__(self):
        if self.algebra.basis_labels is None:
            # Fallback to __str__ when basis_labels is None
            return str(self)

        terms = []
        for coeff, basis_label in zip(self.coeffs, self.algebra.basis_labels):
            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence == 1:
                    terms.append(f"{basis_label}")
                else:
                    terms.append(f"{basis_label}^''")
            elif coeff == -1:
                if self.valence == 1:
                    terms.append(f"-{basis_label}")
                else:
                    terms.append(f"-{basis_label}^''")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(f"({coeff}) * {basis_label}")
                    else:
                        terms.append(f"({coeff}) * {basis_label}^''")
                else:
                    if self.valence == 1:
                        terms.append(f"{coeff} * {basis_label}")
                    else:
                        terms.append(f"{coeff} * {basis_label}^''")

        if not terms:
            if self.valence == 1:
                return f"0*{self.algebra.basis_labels[0]}"
            else:
                return f"0*{self.algebra.basis_labels[0]}^''"

        return " + ".join(terms).replace("+ -", "- ")

    @property
    def label(self):
        return self.__repr__()

    @property
    def is_zero(self):
        for j in self.coeffs:
            if sp.simplify(j) != 0:
                return False
        else:
            return True

    def subs(self, subsData):
        newCoeffs = [sp.sympify(j).subs(subsData) for j in self.coeffs]
        return algebra_element_class(
            self.algebra, newCoeffs, self.valence, format_sparse=self.is_sparse
        )

    @property
    def ambient_rep(self):
        return self

    def _eval_simplify(self,*args,**kwargs):
        return algebra_element_class(
            self.algebra,
            [simplify_dgcv(c) for c in self.coeffs],
            self.valence,
            format_sparse=self.is_sparse,
        )

    def dual(self):
        return algebra_element_class(
            self.algebra,
            self.coeffs,
            (self.valence + 1) % 2,
            format_sparse=self.is_sparse,
        )

    def _convert_to_tp(self):
        return tensorProduct(
            tuple([self.dgcv_vs_id]),
            {(j,self.valence,self.dgcv_vs_id): self.coeffs[j] for j in range(self.algebra.dimension)},
        )

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def _si_wrap(self, obj):
        if self.algebra.simplify_products_by_default is True:
            return simplify_dgcv(obj)
        else:
            return obj

    def _fast_add(self, other):
        """
        Internal-only: assumes `other` is an algebra_element_class
        with the same algebra and valence.
        No type or safety checks.
        No simplification.
        """
        coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
        return algebra_element_class(
            self.algebra,
            coeffs,
            self.valence,
            format_sparse=self.is_sparse,
        )

    def __add__(self, other):
        if getattr(other, "is_zero", False) or other==0:
            return self
        if get_dgcv_category(other) == "subalgebra_element":
            if other.algebra.ambient.dgcv_vs_id==self.dgcv_vs_id and self.valence == other.valence:
                other = other.ambient_rep
            else:
                other=other._convert_to_tp()
        if get_dgcv_category(other)=='algebra_element':
            if self.algebra == other.algebra and self.valence == other.valence:
                coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
                return algebra_element_class(
                    self.algebra,
                    coeffs,
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                other=other._convert_to_tp()
        if get_dgcv_category(other)=='vector_space_element':
            other=other._convert_to_tp()
        if isinstance(other,_get_expr_num_types()):
            other=tensorProduct('_',{tuple():other})
        if isinstance(other, tensorProduct):
            return self._convert_to_tp()+other
        return NotImplemented

    def __radd__(self, other):
        if getattr(other, "is_zero", False) or other==0:
            return self
        if isinstance(other,_get_expr_num_types()):
            return tensorProduct('_',{tuple():other})+self
        return NotImplemented

    def __sub__(self, other):
        if getattr(other, "is_zero", False) or other==0:
            return self
        if get_dgcv_category(other) == "subalgebra_element": 
            if other.algebra.ambient.dgcv_vs_id==self.dgcv_vs_id and self.valence == other.valence:
                other = other.ambient_rep
            else:
                other=other._convert_to_tp()
        if get_dgcv_category(other) == 'algebra_element':
            if self.algebra == other.algebra and self.valence == other.valence:
                return algebra_element_class(
                    self.algebra,
                    [
                        self._si_wrap(self.coeffs[j] - other.coeffs[j])
                        for j in range(len(self.coeffs))
                    ],
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                other=other._convert_to_tp()
        if get_dgcv_category('other')=='vector_space_element':
            other=other._convert_to_tp()
        if isinstance(other,_get_expr_num_types()):
            other=tensorProduct('_',{tuple():other})
        if isinstance(other, tensorProduct):
            return self._convert_to_tp()-other
        return NotImplemented

    def __rsub__(self, other):
        if getattr(other, "is_zero", False) or other==0:
            return -self
        if isinstance(other,_get_expr_num_types()):
            return tensorProduct('_',{tuple():other})-self
        return NotImplemented

    def __mul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if other.algebra.ambient.dgcv_vs_id==self.dgcv_vs_id and self.valence == other.valence:
                other = other.ambient_rep
            else:
                other=other._convert_to_tp()
        if isinstance(other, algebra_element_class):
            if self.algebra == other.algebra and self.valence == other.valence:
                sign = 1 if self.valence == 1 else -1
                alg = self.algebra
                dim = alg.dimension
                coeffs1 = self.coeffs
                coeffs2 = other.coeffs
                struct = alg.structureData

                raw_result = [0] * dim

                for i in range(dim):
                    ci = coeffs1[i]
                    if ci == 0:
                        continue
                    for j in range(dim):
                        cj = coeffs2[j]
                        if cj == 0:
                            continue
                        scalar = sign * ci * cj
                        row = struct[i][j]
                        for k in range(dim):
                            c_ijk = row[k]
                            if c_ijk != 0:
                                raw_result[k] += scalar * c_ijk

                if self.algebra.simplify_products_by_default:
                    result_coeffs = [self._si_wrap(c) for c in raw_result]
                else:
                    result_coeffs = raw_result

                return algebra_element_class(
                    self.algebra,
                    result_coeffs,
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                other=other._convert_to_tp()
        elif isinstance(other, tensorProduct):
            return self._si_wrap((self._convert_to_tp()) * other)
        elif isinstance(other, _get_expr_num_types()):
            new_coeffs = [self._si_wrap(coeff * other) for coeff in self.coeffs]
            return algebra_element_class(self.algebra, new_coeffs, self.valence, format_sparse=self.is_sparse)
        return NotImplemented

    def __rmul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if other.algebra.ambient.dgcv_vs_id==self.dgcv_vs_id and self.valence == other.valence:
                return other.ambient_rep * self
        if isinstance(other, _get_expr_num_types()) or get_dgcv_category(other) in {"subalgebra_element",'algebra_element','tensorProduct'}:
            return self._si_wrap(self * other)
        return NotImplemented

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if get_dgcv_category(other)=='tensorProduct':
            return self._convert_to_tp() @ other
        if isinstance(other,_get_expr_num_types()):
            return other*self
        if get_dgcv_category(other) not in {'algebra_element','subalgebra_element','vector_space_element'}:
            raise TypeError(
                f"unsuported operand types for `@`. Types {type(self)} and {type(other)}"
            ) from None
        new_dict = {
            (j, k, self.valence, other.valence, self.dgcv_vs_id, other.dgcv_vs_id): self.coeffs[j] * other.coeffs[k]
            for j in range(self.algebra.dimension)
            for k in range(other.algebra.dimension)
        }
        return self._si_wrap(tensorProduct(mergeVS([self.dgcv_vs_id],[other.dgcv_vs_id]), new_dict))

    def __rmatmul__(self,other):
        if isinstance(other,_get_expr_num_types()):
            return other*self
        return self._convert_to_tp().__rmatmul__(other) 

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self._si_wrap(sp.Rational(1, other) * self)
        elif isinstance(other, _get_expr_num_types()):
            return self._si_wrap((1 / other) * self)
        else:
            raise TypeError(
                f"True division `/` of algebra elements by another object is only supported for scalars, not {type(other)}"
            ) from None

    def __neg__(self):
        return -1 * self

    def __xor__(self, other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    def __call__(self, other, **kwds):
        if get_dgcv_category(other) == "subalgebra_element":
            other = other.ambient_rep
        if get_dgcv_category(other) == "algebra_element":
            if other.algebra == self.algebra and other.valence != self.valence:
                return sum([j * k for j, k in zip(self.coeffs, other.coeffs)])
            elif self.tensor_representation is not None:
                return self.tensor_representation(other)
            else:
                raise TypeError(
                    f"`algebra_element_class` call can only be applied to elements from the same algebra pairing one element with another of complementary valence, or applying elements from an endomorphism_space subclass. Recieved self: {self} and other: {other} belonging to {self.algebra} and {other.algebra} with valences {self.valence} and {other.valence}"
                )
        else:
            raise TypeError(
                f"`algebra_element_class` call cannot be applies objects of type {type(other)}"
            )

    def check_element_weight(self, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of this algebra_element_class with respect to its algebra' grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.
            Each entry is either an integer, variable representing a weight, the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parent algebra' check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """

        return self.algebra.check_element_weight(
            self, test_weights=test_weights, flatten_weights=flatten_weights
        )

    def weighted_decomposition(self, test_weights=None, flatten_weights=False):
        weighted_components = {}
        for idx, coeff in enumerate(self.coeffs):
            if coeff != 0:
                elem = self.algebra.basis[idx]
                w = elem.check_element_weight(
                    test_weights=test_weights, flatten_weights=flatten_weights
                )
                if isinstance(w, list):
                    w = tuple(w)
                weighted_components[w] = weighted_components.get(w, 0) + coeff * elem
        return weighted_components

    def coproduct(self):
        if self.valence!=0:
            return print('The algebra co-product is only defined on dual Lie algebra elements as it is dual to the algebra product map.')
        terms = []
        for c,elem in zip(self.coeffs,self.algebra.basis):
            if c!=0:
                if self.algebra._coproduct.get(elem,None) is None:
                    tensor_terms = []
                    for idx,e1 in enumerate(self.algebra.basis):
                        if self.algebra.is_skew_symmetric:
                            skew=True
                            start=idx+1
                        else:
                            skew=False
                            start=0
                        for e2 in self.algebra.basis[start:]:
                            if skew:
                                tensor_terms.append(self(e1*e2) * (e1.dual() @ e2.dual() - e2.dual() @ e1.dual()))
                            else:
                                tensor_terms.append(self(e1*e2) * (e1.dual() @ e2.dual()))
                    self.algebra._coproduct[elem]=sum(tensor_terms)
                terms.append(c*self.algebra._coproduct[elem])
        return sum(terms)

    @property
    def free_symbols(self):
        fs = set()
        for c in self.coeffs:
            fs |= getattr(c, "free_symbols", set())
        return fs

    def dual_pairing(self,other):
        return self._convert_to_tp().dual_pairing(other)

    def decompose(self):
        if self.valence==1:
            return self.coeffs,self.algebra.basis
        return self.coeffs,[j.dual() for j in self.algebra.basis]

    ###!!! CHECK
    # @property
    # def terms(self):
    #     return self._convert_to_tp().terms

class algebra_subspace_class:
    def __init__(
        self,
        basis,
        parent_algebra=None,
        test_weights=None,
        _grading=None,
        _internal_lock=None,
        span_warning=True,
        simplify_basis=False,
        **kwargs,
    ):
        # From former __new__: validate inputs and compute subspace attributes
        if not isinstance(basis, (list, tuple)):
            raise TypeError(
                "algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances"
            ) from None
        typeCheck = {"subalgebra_element", "algebra_element"}
        if not all(get_dgcv_category(j) in typeCheck for j in basis):
            raise TypeError(
                "algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances"
            ) from None
        if parent_algebra is None:
            if len(basis) > 0:
                if get_dgcv_category(basis[0].algebra) != "algebra":
                    if all(j.algebra == basis[0].algebra for j in basis[1:]):
                        parent_alg = basis[0].algebra.ambient
                    else:
                        parent_alg = basis[0].algebra.ambient
                else:
                    parent_alg = basis[0].algebra
            else:
                parent_alg = None
        elif get_dgcv_category(parent_algebra) in {"subalgebra", "algebra_subspace"}:
            parent_alg = parent_algebra.ambient
        elif get_dgcv_category(parent_algebra) == "algebra":
            parent_alg = parent_algebra
        else:
            raise TypeError(
                "algebra_subspace_class expects second argument to be an algebra instance or algebra subspace or subalgebra."
            ) from None
        # Filter independent elements
        filtered_basis = parent_alg.filter_independent_elements(
            basis, apply_light_basis_simplification=simplify_basis
        )
        if len(filtered_basis) < len(basis):
            basis = filtered_basis
            if span_warning:
                wmessage = ' This can result in incorrect weighting assignements from the manual assignment provided to the `test_weights` parameter. To avoid this issue, provided a linearly independent spanning set instead.' if test_weights is None else ''
                warnings.warn(
                    "The given list for `basis` was not linearly independent, so the algebra_subspace_class initializer computed a basis for its span to use instead."+wmessage
                )
        # Assign to self
        self.filtered_basis = tuple(filtered_basis)
        self.basis = tuple(filtered_basis)
        self.dimension = len(filtered_basis)
        self.ambient = parent_alg
        # Compute grading for subspace
        grading_per_elem = []
        if (
            _internal_lock == retrieve_passkey()
            and test_weights is None
            and _grading is not None
        ):
            self.grading = _grading
        else:
            for elem in filtered_basis:
                weight = parent_alg.check_element_weight(elem, test_weights=test_weights)
                grading_per_elem.append(weight)
            self.grading = [
                elem for elem in zip(*grading_per_elem) if "NoW" not in elem
            ]
        self.original_basis = basis
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra_subspace"

        # immutables
        self._grading = tuple(self.grading)
        self._gradingNumber = len(self._grading)

        vsr=get_vs_registry()
        self.dgcv_vs_id = len(vsr)
        vsr.append(self)

        # attribute caches
        self._is_subalgebra = None

    @property
    def zero_element(self):
        return algebra_element_class(self,(0,)*self.ambient.dimension,1)

    def __eq__(self, other):
        if not isinstance(other, algebra_subspace_class):
            return NotImplemented
        return (self.dgcv_vs_id == other.dgcv_vs_id)

    def __hash__(self):
        return hash(self.dgcv_vs_id)

    def __contains__(self, item):
        return item in self.basis

    def update_grading(self,new_weight_vectors_list,replace_instead_of_add=False):
        if isinstance(new_weight_vectors_list,(list,tuple)):
            if all(isinstance(elem,(list,tuple)) for elem in new_weight_vectors_list):
                   if replace_instead_of_add is True:
                       self.grading = [tuple(elem) for elem in new_weight_vectors_list]
                   else:
                       grad=list(self.grading)+[tuple(elem) for elem in new_weight_vectors_list]
                       self.grading=grad
            else:
                raise TypeError(f'update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}')
        else:
            raise TypeError(f'update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}')

    def check_element_weight(self, element, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of an algebra_element_class with respect to the grading vectors assigned to an algebra_subspace_class. Weight can be instead computed against another grading vector passed a list of weights as the keyword `test_weights`.

        Parameters
        ----------
        element : (sub)algebra_element_class
            The (sub)algebra_element_class to analyze.
        test_weights : list of int or sympy.Expr, optional (default: None)
        flatten_weights : (default: False) If True, returns contents of a list if otherwise would have returned a length 1 list

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of this algebra subspace (or test_weights if provided).
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' (i.e., All Weights) if the element is the zero element,
            or 'NoW' (i.e., No Weights) if the element is not homogeneous.
            If the list is length 1 and flatten_weights=True then only the contents of the list is returned.

        Notes
        -----
        - 'AllW' (meaning, All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (meaning, No Weights) is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if not isinstance(element, algebra_element_class) or element.algebra != self:
            if (
                get_dgcv_category(element) == "subalgebra_element"
                and element.algebra.ambient == self.ambient
            ):
                pass
            else:
                raise TypeError(
                    "Input in `algebra_subspace_class.check_element_weight` must be an `(sub)algebra_element_class` instance belonging to the `(sub)algebra_class` instance whose `check_element_weight` is being called."
                ) from None
        if not test_weights:
            if self._gradingNumber == 0:
                raise ValueError(
                    "This algebra subspace instance has no assigned grading vectors to test weighting w.r.t.."
                ) from None
        if all(coeff == 0 for coeff in element.coeffs):
            return tuple(["AllW"] * self._gradingNumber)
        if test_weights:
            if not isinstance(test_weights, (list, tuple)):
                raise TypeError(
                    f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, etc.). Revieved {test_weights}"
                ) from None
            for weight in test_weights:
                if not isinstance(weight, (list, tuple)):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, etc.). Revieved {test_weights}"
                    ) from None
                if self.dimension != len(weight) or not all(
                    [isinstance(j, _get_expr_num_types()) for j in weight]
                ):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or symbolic expresion). Revieved {test_weights}"
                    ) from None
            GVs = test_weights
        else:
            GVs = self.grading
        weights = []
        for grading_vector in GVs:
            non_zero_indices = [
                i for i, coeff in enumerate(element.coeffs) if coeff != 0
            ]
            basis_weights = [grading_vector[i] for i in non_zero_indices]
            if len(set(basis_weights)) == 1:
                weights.append(basis_weights[0])
            else:
                weights.append("NoW")
        if flatten_weights and len(weights) == 1:
            return weights[0]
        return tuple(weights)

    def contains(self, items, return_basis_coeffs=False,strict_types=False):
        if isinstance(items, (list, tuple)):
            return [self.contains(item,return_basis_coeffs=return_basis_coeffs) for item in items]
        if strict_types is False and items==0:
            if return_basis_coeffs is True:
                return [0]*self.dimension
            return True
        item = items
        if get_dgcv_category(item) == "subalgebra_element":
            if item.dgcv_vs_id==self.dgcv_vs_id:
                if return_basis_coeffs is True:
                    return list(item.coeffs)
                else:
                    return True
            item = item.ambient_rep
        if (not isinstance(item, algebra_element_class) or item.algebra != self.ambient):
            return False
        if item not in self.basis:
            if self.dimension==0:
                return False
            tempVarLabel = "T" + retrieve_public_key()
            vars=variableProcedure(tempVarLabel, len(self.basis), _tempVar=retrieve_passkey(),return_created_object=True)[0]
            genElement = sum([vars[j + 1] * elem for j, elem in enumerate(self.basis[1:])], vars[0] * (self.basis[0]))
            sol = solve_dgcv(item - genElement, vars)
            if len(sol) == 0:
                clearVar(*listVar(temporary_only=True),report=False)
                return False
        else:
            if return_basis_coeffs is True:
                idx = (self.basis).index(item)
                return [1 if _ == idx else 0 for _ in range(len(self.basis))]
        if return_basis_coeffs is True:
            vec = [var.subs(sol[0]) for var in vars]
            clearVar(*listVar(temporary_only=True),report=False)
            return vec
        clearVar(*listVar(temporary_only=True),report=False)
        return True

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, index):
        return self.basis[index]

    def is_subalgebra(self, return_structure_data=False):
        if self._is_subalgebra is None:
            self._is_subalgebra = self.ambient.is_subspace_subalgebra(
                self.filtered_basis, return_structure_data=return_structure_data
            )
        return self._is_subalgebra

    def __str__(self):
        return f"span{{{', '.join(elem.__str__() for elem in self.basis)}}}"

    def __repr__(self):
        return self.__str__()

    def _repr_latex_(self,**kwargs):
        formatted_str = f"\\langle{', '.join(elem._repr_latex_() for elem in self.basis)}\\rangle".replace(
            "$", ""
        ).replace(
            "\\displaystyle", ""
        )
        return f"$\\displaystyle {formatted_str}$"

    def _latex(self, printer=None):
        return self._repr_latex_().replace("$", "").replace("\\displaystyle", "")

    def __add__(self, other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        if get_dgcv_category(other) in {'algebra_subspace', 'subalgebra'}:
            if other.dimension==0:
                return self
            if self.dimension==0:
                return other
            new_basis = list(self.basis)
            for elem in getattr(other,'basis_in_ambient_alg',other.basis):
                new_basis = _basis_builder(new_basis,elem)
            return algebra_subspace_class(new_basis,self.ambient)
        return NotImplemented

    def __radd__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        return NotImplemented

    def append(self,item,recompute_gradings_and_return_new=False):
        if item == 0 or getattr(item,'is_zero',False):
            pass
        elif recompute_gradings_and_return_new:
            bas=list(self.filtered_basis)
            bas.append(item)
            return self.ambient.subspace(bas)
        elif not self.contains(item):
            self.original_basis = list(self.original_basis)+[item]
            self.basis = tuple(self.basis)+(item,)
            self.filtered_basis = tuple(self.filtered_basis)+(item,)
            self.dimension+=1
            self.grading = [(0,)*self.dimension]


############## algebra tools

def decompose_semisimple_algebra(alg, assume_semisimple=False, format_as_lists_of_elements=False):
    assert get_dgcv_category(alg) in {'algebra','subalgebra'}
    if alg.dimension==0:
        return [alg]
    if assume_semisimple is False and not alg.is_semisimple():
        raise TypeError('decompose_semisimple_algebra was given a non-semisimple algebra to decompose.')
    mbasis = [sp.Matrix(j).transpose() for j in alg.structureData]
    pref = create_key('_var')
    vars = [sp.Symbol(f'{pref}{j}') for j in range((alg.dimension)**2)]
    vMat = sp.Matrix(alg.dimension,alg.dimension,vars)
    mats=sum([[*(vMat*mat-mat*vMat)] for mat in mbasis],[])
    sol = solve_dgcv(mats,vars)
    solMat = vMat.subs(sol[0])
    free_vars = set()
    for entry in solMat:
        free_vars|=entry.free_symbols
    if len(free_vars)<2:
        simples = [alg]
    else:
        bound = max(100, 10*alg.dimension)
        _cached_caller_globals['DEGUBfv']=free_vars,solMat
        solMat = solMat.subs({var:random.randint(1,bound) for var in free_vars})
        bases = [j[2] for j in solMat.eigenvects()]
        simples = []
        for base in bases:
            new_basis = []
            for vec in base:
                new_basis.append(sum([c*elem for c,elem in zip(vec,alg.basis)]))
            if format_as_lists_of_elements is True:
                new_alg = new_basis
            else:
                new_alg = alg.subalgebra(new_basis)
            simples.append(new_alg)
    return simples

def killingForm(alg, list_processing=False, assume_Lie_algebra=False):
    if get_dgcv_category(alg) in {"algebra", "subalgebra"}:
        if alg._killing_form is None:
            if assume_Lie_algebra is False and not alg.is_Lie_algebra():
                raise Exception(
                    "killingForm expects argument to be a Lie algebra instance of the algebra"
                ) from None
            if list_processing:
                aRepLoc = alg.structureData
                return [
                    [
                        trace_matrix(multiply_matrices(aRepLoc[j], aRepLoc[k]))
                        for k in range(alg.dimension)
                    ]
                    for j in range(alg.dimension)
                ]
            else:
                aRepLoc = adjointRepresentation(
                    alg, assume_Lie_algebra=assume_Lie_algebra
                )
                alg._killing_form = sp.Matrix(
                    alg.dimension,
                    alg.dimension,
                    lambda j, k: (aRepLoc[j] * aRepLoc[k]).trace(),
                )
        return alg._killing_form
    else:
        raise Exception(
            "killingForm expected to receive an algebra instance."
        ) from None

def adjointRepresentation(alg, list_format=False, assume_Lie_algebra=False):
    if get_dgcv_category(alg) in {"algebra", "subalgebra"}:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            warnings.warn("Caution: The algebra passed to `adjointRepresentation` is not a Lie algebra.")
        if list_format:
            return alg.structureData
        return [sp.Matrix(j).transpose() for j in alg.structureData]
    else:
        raise Exception("adjointRepresentation expected to receive an algebra instance.") from None


############## misc


def convert_to_greek(var_name):
    for name, greek in greek_letters.items():
        if var_name.lower().startswith(name):
            return var_name.replace(name, greek, 1)
    return var_name


def format_latex_subscripts(var_name, nest_braces=False):
    """for use_latex branches"""
    if var_name[-1] == "_":
        var_name = var_name[:-1]
    if var_name=='':
        return r'\_'
    if var_name[0]=='_':
        return format_latex_subscripts(var_name=var_name[1:],nest_braces=nest_braces)
    parts = var_name.split("_")
    if len(parts) == 1:
        return convert_to_greek(var_name)
    base = convert_to_greek(parts[0])
    subscript = ", ".join(parts[1:])
    if nest_braces is True:
        return f"{{{base}_{{{subscript}}}}}"
    else:
        return f"{base}_{{{subscript}}}"


############## linear algebra for list/tuple processing
def multiply_matrices(A, B):
    """
    Multiplies two matrices A and B, represented as lists of lists.

    Parameters
    ----------
    A : list of lists
        The first matrix (m x n).
    B : list of lists
        The second matrix (n x p).

    Returns
    -------
    list of lists
        The resulting matrix (m x p) after multiplication.

    Raises
    ------
    ValueError
        If the number of columns in A is not equal to the number of rows in B.
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(
            "Incompatible matrix dimensions: A is {}x{}, B is {}x{}".format(
                rows_A, cols_A, rows_B, cols_B
            )
        ) from None
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def fast_rank(mat: sp.Matrix) -> int:
    if mat.rows == 0 or mat.cols == 0:
        return 0
    fast_types = _get_fast_scalar_types()
    if all(isinstance(a, fast_types) for a in mat):
        lcm = 1
        for a in mat:
            num, den = a.as_numer_denom()
            lcm = math.lcm(lcm, int(den))
        M = (mat * lcm).applyfunc(int) if lcm != 1 else mat.applyfunc(int)
        _, pivots = M.rref(iszerofunc=lambda x: x == 0, simplify=False)
        return len(pivots)
    _, pivots = mat.rref(iszerofunc=lambda x: x == 0, simplify=False)
    return len(pivots)


def trace_matrix(A):
    """
    Computes the trace of a square matrix A (sum of the diagonal elements).

    Parameters
    ----------
    A : list of lists
        The square matrix.

    Returns
    -------
    trace_value
        The trace of the matrix (sum of the diagonal elements).

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    rows_A, cols_A = len(A), len(A[0])
    if rows_A != cols_A:
        raise ValueError(
            "Trace can only be computed for square matrices. sp.Matrix is {}x{}.".format(
                rows_A, cols_A
            )
        ) from None
    trace_value = sum(A[i][i] for i in range(rows_A))
    return trace_value


def _indep_check(elems, newE, return_decomp_coeffs=False, print_solve_stats=False, method="linsolve", _solve_variables=None):

    if not isinstance(elems, (list,tuple)) or len(elems) == 0:
        if return_decomp_coeffs:
            return True, []
        return True
    if newE == 0 or getattr(newE, "is_zero", False):
        if return_decomp_coeffs:
            return False, [[0] * len(elems)]
        return False
    pref = create_key(prefix="var")
    if _solve_variables is None or len(_solve_variables)<len(elems):
        vars = [sp.Symbol(f"{pref}{j}") for j in range(len(elems))]
    else:
        vars = _solve_variables[:len(elems)]
    terms = [var * elem for var, elem in zip(vars, elems)]
    eqn = newE - sum(terms[1:], terms[0])
    sol = solve_dgcv([eqn], vars, print_solve_stats=print_solve_stats, method=method)
    if len(sol) == 0:
        if return_decomp_coeffs:
            return True, []
        return True
    if return_decomp_coeffs:
        coeffs = [var.subs(sol[0]) for var in vars]
        free_vars = set()
        for c in coeffs:
            free_vars |= getattr(c, "free_symbols", set())
        free_vars = {var for var in free_vars if var in vars}  ###!!!
        if len(free_vars) == 0:
            coeffs = [coeffs]
        else:
            coeffs = [
                [c.subs({v: 1}).subs({var: 0 for var in free_vars}) for c in coeffs]
                for v in free_vars
            ]
        return False, coeffs
    return False


def _elem_scale(elem):
    coeffs = getattr(elem, "coeffs", None)
    if isinstance(coeffs, (list, tuple)):
        for c in coeffs:
            if c != 0:
                try:
                    return elem / c
                except Exception:
                    return elem
    return elem

def _basis_builder(elems, newE, ALBS=False, print_solve_stats=False, method="linsolve", _solve_variables=None):
    if newE == 0 or getattr(newE, "is_zero", False):
        return list(elems)
    if ALBS is True:
        newE = _elem_scale(newE)
    if not isinstance(elems,(list,tuple)):
        raise TypeError(f'_basis_builder expects `elems` to be a list, recieved {elems} of type {type(elems)}')
    if len(elems) == 0:
        return [newE]
    if _indep_check(elems, newE, print_solve_stats=print_solve_stats, method=method, _solve_variables=_solve_variables) is True:
        return list(elems) + [newE]
    else:
        return list(elems)


def _extract_basis(element_list, ALBS=False, print_solve_stats=False, method="linsolve", _solve_variables = None):
    basis = []
    for newE in element_list:
        basis = _basis_builder(basis, newE, ALBS=ALBS, print_solve_stats=print_solve_stats, method=method, _solve_variables=_solve_variables)
    return basis


def _generate_gl_structure_data(vs):
    n=len(vs.basis)-1
    matrix_dim = n + 1

    # Basis elements
    hBasis = {"elems": dict(), "grading": dict()}
    offDiag = {"elems": dict(), "grading": dict()}

    repMatrix = [[0] * matrix_dim for _ in range(matrix_dim)]

    def elemWeights(idx1, idx2):
        wVec = []
        for idx in range(n):
            if idx1 <= idx:
                if idx2 <= idx:
                    wVec.append(0)
                else:
                    wVec.append(1)
            else:
                if idx2 <= idx:
                    wVec.append(-1)
                else:
                    wVec.append(0)
        return wVec

    for j in range(n + 1):
        for k in range(j, n + 1):
            if j == k and j < n:
                M = [row[:] for row in repMatrix]
                for idx in range(n + 1):
                    if idx > j:
                        M[idx][idx] = -sp.Rational(j+1,n+1)
                    else:
                        M[idx][idx] = 1-sp.Rational(j+1,n+1)
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
            elif j==n and k==n:
                M = [row[:] for row in repMatrix]
                for idx in range(n + 1):
                    M[idx][idx] = 1
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n             
            elif j != k:
                MPlus = [row[:] for row in repMatrix]
                MMinus = [row[:] for row in repMatrix]
                MPlus[j][k] = 1
                MMinus[k][j] = 1
                offDiag["elems"][(j, k, 1)] = MPlus
                offDiag["grading"][(j, k, 1)] = elemWeights(j, k)
                offDiag["elems"][(k, j, 1)] = MMinus
                offDiag["grading"][(k, j, 1)] = elemWeights(k, j)

    indexingKey = dict(
        enumerate(list(hBasis["grading"].keys()) + list(offDiag["grading"].keys()))
    )
    indexingKeyRev = {j: k for k, j in indexingKey.items()}
    LADimension = len(indexingKey)
    def _structureCoeffs(idx1, idx2):
        coeffs = [0] * LADimension
        if idx2 == idx1:
            return coeffs
        if idx2 < idx1:
            reSign = -1
            idx2, idx1 = idx1, idx2
        else:
            reSign = 1
        p10, p11, p12 = indexingKey[idx1]
        p20, p21, p22 = indexingKey[idx2]
        if p12 == 0:
            if p22 == 1 and (p10!=n or p11!=n):
                coeffs[idx2] += reSign * (
                    int(p10 == p20)
                    - int(p10 == p21)
                    + int(p10 + 1 == p21)
                    - int(p10 + 1 == p20)
                )
        elif p12 == 1:
            if p22 == 1:
                if p11 == p20:
                    if p10 == p21:
                        if p10 < p11:
                            for idx in range(p10, p11):
                                coeffs[indexingKeyRev[(idx, idx, 0)]] = reSign
                        else:
                            for idx in range(p11, p10):
                                coeffs[indexingKeyRev[(idx, idx, 0)]] = -reSign
                    else:
                        coeffs[indexingKeyRev[(p10, p21, 1)]] = reSign
                elif p10 == p21:
                    coeffs[indexingKeyRev[(p20, p11, 1)]] = -reSign
        return coeffs

    _structure_data = [
        [_structureCoeffs(k, j) for j in range(LADimension)]
        for k in range(LADimension)
    ]
    CartanSubalg = list(hBasis["elems"].values())
    matrixBasis = CartanSubalg + list(offDiag["elems"].values())
    def obGen(j,k):
        if j==k:
            if j<n:
                tp = (1-sp.Rational(j+1,n+1))*vs.basis[0]@(vs.basis[0].dual())
                for idx in range(1,n + 1):
                    if idx > j:
                        tp += -sp.Rational(j+1,n+1)*vs.basis[idx]@(vs.basis[idx].dual())
                    else:
                        tp += (1-sp.Rational(j+1,n+1))*vs.basis[idx]@(vs.basis[idx].dual())
                return tp
            return sum([vs.basis[j]@(vs.basis[j].dual()) for j in range(n)],vs.basis[n]@(vs.basis[n].dual()))
        else:
            return vs.basis[j]@(vs.basis[k].dual())
    operatorBasis = [obGen(indexingKey[idx][0],indexingKey[idx][1]) for idx in range(LADimension)]
    gradingVecs = list(hBasis["grading"].values()) + list(offDiag["grading"].values())
    return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis, operatorBasis


class vector_space_endomorphisms(algebra_class):
    def __init__(self, vector_space):
        self.domain = vector_space
        self._dgcv_categories = {"endomorphism_space"}
        structure_data, grading, _, matrix_representation, tensor_representation = _generate_gl_structure_data(vector_space)
        super().__init__(structure_data, grading=grading, format_sparse=False, process_matrix_rep=False, preferred_representation=None, simplify_products_by_default=None, assume_skew=False, matrix_representation=matrix_representation,tensor_representation=tensor_representation,_basis_labels_parent=None, _label=None, _basis_labels=None, _calledFromCreator=retrieve_passkey(), _callLock=retrieve_passkey(), _print_warning=None, _child_print_warning=None, _exclude_from_VMF=retrieve_passkey(), _markers={'endo':True,'endo_label':vector_space.label,'endo_tex':vector_space._repr_latex_(raw=True,abbrev=True)})

class linear_representation:
    def __init__(self, hom:homomorphism):
        self.structureData,self.antihomomorphism,self.parameters=self._validate_hom(hom)
        self.homomorphism=hom
        self.domain = hom.domain
        self.representation_space = hom.codomain.domain

    @classmethod
    def _validate_hom(cls,hom):
        params=set()
        assert query_dgcv_categories(hom.codomain,{'endomorphism_space','tensor_proxy'})
        skew = getattr(hom.domain,'is_skew_symmetric',False)
        amb_dim = hom.domain.dimension+hom.codomain.domain.dimension
        dom_dim = hom.domain.dimension
        anti=None
        is_zero_map = getattr(hom, '_zero_map', False)
        def _equal(a,b):
            return getattr(a-b,'is_zero',False) or a==b
        for c,e1 in enumerate(hom.domain.basis):
            lidx = c+1 if skew else 0
            for e2 in hom.domain.basis[lidx:]:
                if is_zero_map:
                    anti = False
                    break
                else:
                    p1=hom(e1*e2)
                    p2=hom(e1)*hom(e2)
                    if anti is None and _equal(p1,p2) and not _equal(p1,0*p1):
                        anti=False
                    if not _equal(p1,p2):
                        if anti is None and _equal(p1,-p2):
                            anti=True
                        elif (anti is True and _equal(p1,-p2)):
                            pass
                        else:
                            raise ValueError(f'The `hom` parameter given to the `linear_representation` initializer does not define an algebra homomorphism. The identity hom(v*w)=hom(v)*hom(w) fails for basis elements {e1} and {e2}, producing hom(v*w)={p1} and hom(v)*hom(w)={p2}')
                    if anti is None:
                        anti = False
        def _extract_update(elems,idx,par):
            elem=elems[idx]
            if hasattr(elem,'free_symbols'):
                par|=elem.free_symbols
            return elem
        def sd_gen(j,k,sd1,sd2,assume_skew=True):
            if assume_skew and j==k:
                return tuple(0 for _ in range(amb_dim))
            if j<dom_dim:
                if k<dom_dim:
                    start=sd1[j][k]
                    return tuple(_extract_update(start,jj,params) if jj<dom_dim else 0 for jj in range(amb_dim))
                else:
                    if is_zero_map:
                        end = (0,)*(amb_dim-dom_dim)
                    else:
                        endterm=hom(hom.domain.basis[j])(hom.codomain.domain.basis[k-dom_dim])
                        if endterm==0:
                            end = (0,)*(amb_dim-dom_dim)
                        else:
                            end=endterm.coeffs
                    return tuple(0 if jj<dom_dim else _extract_update(end,jj-dom_dim,params) for jj in range(amb_dim))
            else:
                if k<dom_dim:
                    if is_zero_map:
                        end = (0,)*(amb_dim-dom_dim)
                    else:
                        endterm=hom(hom.domain.basis[k])(hom.codomain.domain.basis[j-dom_dim])
                        if endterm==0:
                            end = (0,)*(amb_dim-dom_dim)
                        else:
                            end=endterm.coeffs
                    return tuple(0 if jj<dom_dim else -end[jj-dom_dim] for jj in range(amb_dim))
                else:
                    end=sd2[j-dom_dim][k-dom_dim]
                    return tuple(0 if jj<dom_dim else end[jj-dom_dim] for jj in range(amb_dim))
        sd_dom=getattr(hom.domain,'structureData',tuple(tuple((0,)*dom_dim for _ in range(dom_dim)) for __ in range(dom_dim)))
        sd_targ=getattr(hom.codomain.domain,'structureData',tuple(tuple((0,)*(amb_dim-dom_dim) for _ in range(amb_dim-dom_dim)) for __ in range(amb_dim-dom_dim)))
        new_sd=tuple(tuple(sd_gen(j,k,sd_dom,sd_targ) for k in range(amb_dim)) for j in range(amb_dim))
        return new_sd, anti, params

    def semidirect_sum(self,
                       grading=None,
                       label=None,
                       basis_labels=None,
                       register_in_vmf=False,
                       initial_basis_index=None,
                       simplify_products_by_default=None,
                       _markers=None):
        if simplify_products_by_default is None:
            simplify_products_by_default = getattr(self.domain,'simplify_products_by_default',False)
        if grading is None:
            g1=tuple(next(iter(self.domain.grading)))
            g2=tuple(next(iter(self.representation_space.grading)))
            grading = [g1+g2]
        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels) != self.domain.dimension+self.representation_space.dimension
            ):
                warnings.warn(
                    f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.domain.dimension}+{self.representation_space.dimension}"
                )
                basis_labels = None
        def _pref(el):
            if el[0]=='_':
                return '_I'+el
            return '_'+el
        def _preftex(el):
            if el[:2]==r'\_':
                return '\\_|'+el
            return '\\_'+el
        if _markers is None:
            _markers={'sum':True,'lockKey':retrieve_passkey()}
            if label is None:
                label = f'{self.domain.label}_semidir_{self.representation_space.label}'
                _markers['_tex_label']=f'{self.domain._repr_latex_(raw=True,abbrev=True)}\\ltimes {self.representation_space._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.domain.basis]+[elem.__repr__() for elem in self.representation_space.basis]
                _markers['_tex_basis_labels']=[elem._repr_latex_(raw=True) for elem in self.domain.basis]+[elem._repr_latex_(raw=True) for elem in self.representation_space.basis]
        elif not isinstance(basis_labels, (tuple, list)):
            if not isinstance(basis_labels,str):
                basis_labels=[elem.__repr__() for elem in self.domain.basis]+[elem.__repr__() for elem in self.representation_space.basis]
            else:
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [f"{pref}{i+IIdx}" for i in range(self.domain.dimension+self.representation_space.dimension)]
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()

        _markers['semidirect_decomposition']=(self.domain,self.representation_space,self.homomorphism)
        _markers['parameters']=self.parameters
        if register_in_vmf is True:
            from .algebras_secondary import createAlgebra
            return createAlgebra(
                self.structureData,
                label,
                basis_labels=basis_labels,
                grading=grading,
                return_created_obj=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers
            )
        else:
            _markers['registered'] = False
            return algebra_class(
                self.structureData,
                grading=grading,
                simplify_products_by_default=simplify_products_by_default,
                _label=label,
                _basis_labels=basis_labels,
                _calledFromCreator=retrieve_passkey(),
                _markers=_markers
            )

    def __call__(self, *args, **kwds):
        return self.homomorphism.__call__(*args, **kwds)

class algebra_structure_data:
    """
    Lazy algebra structure data container.

    shape: None (default), "skew", or "symmetric".
      - "skew": call gen only for i<j, mirror with minus; diagonal forced 0.
      - "symmetric": call gen only for i<=j, mirror with plus.

    cache_max_pairs: LRU size for cached (i,j) results.
    """

    def __init__(self, dim, gen, *, shape=None, cache_max_pairs=20000):
        if dim <= 0: 
            raise ValueError("dim must be positive")
        if shape not in (None, "skew", "symmetric"):
            raise ValueError("shape must be None, 'skew', or 'symmetric'")
        self.dim = dim
        self._gen = gen
        self._shape = shape

        @lru_cache(maxsize=cache_max_pairs)
        def _pair_cache(i, j):
            i0, j0, sign = self._normalize_pair(i, j)
            if sign == 0: 
                return ()  # skew-diagonal short-circuit
            payload = self._normalize_output(self._gen(i0, j0))
            if sign == -1: 
                return tuple((k, -v) for (k, v) in payload)
            return payload
        self._pair_cache = _pair_cache

        @lru_cache(maxsize=cache_max_pairs)
        def _vector_cache(i, j):
            if self._normalize_pair(i, j)[2] == 0: 
                return tuple(0 for _ in range(self.dim))
            vec = [0]*self.dim
            for k, v in self._pair_cache(i, j):
                if 0 <= k < self.dim and v != 0: 
                    vec[k] = v
            return tuple(vec)
        self._vector_cache = _vector_cache

    def pair(self, i, j):
        self._check_bounds(i, j)
        return dict(self._pair_cache(i, j))

    def coeff(self, i, j, k):
        self._check_bounds(i, j, k)
        return self._vector_cache(i, j)[k]

    def vector(self, i, j):
        self._check_bounds(i, j)
        return list(self._vector_cache(i, j))

    def clear_cache(self):
        self._pair_cache.cache_clear()
        self._vector_cache.cache_clear()

    def stats(self):
        return {"dim": self.dim, "shape": self._shape, "pairs_cached": self._pair_cache.cache_info().currsize, "pair_cache_info": self._pair_cache.cache_info(), "vectors_cached": self._vector_cache.cache_info().currsize, "vector_cache_info": self._vector_cache.cache_info()}

    def _check_bounds(self, i, j, k=None):
        if not (0 <= i < self.dim and 0 <= j < self.dim): 
            raise IndexError(f"(i,j)=({i},{j}) out of bounds for dim={self.dim}")
        if k is not None and not (0 <= k < self.dim): 
            raise IndexError(f"k={k} out of bounds for dim={self.dim}")

    def _normalize_output(self, raw):
        if isinstance(raw, dict): 
            return tuple((k, v) for k, v in raw.items() if v != 0)
        if isinstance(raw, list):
            if len(raw) != self.dim: 
                raise ValueError("Dense generator output length != dim")
            return tuple((k, v) for k, v in enumerate(raw) if v != 0)
        return tuple((k, v) for (k, v) in raw if v != 0)

    def _normalize_pair(self, i, j):
        if self._shape is None: 
            return i, j, 1
        if self._shape == "skew":
            if i == j: 
                return i, j, 0
            return (i, j, 1) if i < j else (j, i, -1)
        return (i, j, 1) if i <= j else (j, i, 1)

    # Back-compat: sd[i][j][k]
    def __getitem__(self, i):
        self._check_bounds(i, 0)
        return _RowView(self, i)

class _RowView:
    __slots__ = ("_sd", "_i")
    def __init__(self, sd, i): self._sd, self._i = sd, i
    def __getitem__(self, j): 
        self._sd._check_bounds(self._i, j)
        return _ColumnView(self._sd, self._i, j)

class _ColumnView:
    __slots__ = ("_sd", "_i", "_j")
    def __init__(self, sd, i, j): self._sd, self._i, self._j = sd, i, j
    def __getitem__(self, k): return self._sd.coeff(self._i, self._j, k)
    def __iter__(self): return iter(self._sd.vector(self._i, self._j))
    def __len__(self): return self._sd.dim
    def __repr__(self): return f"ColumnView({self._i},{self._j}): {self._sd.vector(self._i, self._j)}"

class algebra_rep_data:
    def __init__(self, dim, matrix_gen, operator_gen, *, cache_max=50000):
        self.dim = dim
        self._matrix_gen = matrix_gen
        self._operator_gen = operator_gen
        @lru_cache(maxsize=cache_max)
        def _cache(mode, i):
            if not (0 <= i < self.dim): 
                raise IndexError(f"index {i} out of bounds for dim={self.dim}")
            if mode == "matrix":
                return self._matrix_gen(i)
            if mode == "operator": 
                return self._operator_gen(i)
            raise KeyError("mode must be 'matrix' or 'operator'")
        self._cache = _cache

    def __getitem__(self, mode):
        if mode not in ("matrix","operator"): 
            raise KeyError("use ['matrix'] or ['operator']")
        return _RepModeView(self, mode)

    def clear_cache(self): self._cache.cache_clear()
    def stats(self): return {"dim": self.dim, "items_cached": self._cache.cache_info().currsize, "cache_info": self._cache.cache_info()}

class _RepModeView:
    __slots__ = ("_rep","_mode")
    def __init__(self, rep, mode): 
        self._rep = rep
        self._mode = mode
    def __getitem__(self, i): return self._rep._cache(self._mode, i)
    def __len__(self): return self._rep.dim
    def __iter__(self): 
        for i in range(self._rep.dim):
            yield self[i]
    def __repr__(self): return f"algebra_rep_view(mode={self._mode}, dim={self._rep.dim})"

class _lazy_SD(Mapping):
    def __init__(self, structure_data):
        self._data = structure_data
        self._cache = {}

    def __getitem__(self, key):
        # key is expected to be (i, j, k)
        if key in self._cache:
            return self._cache[key]
        i, j, k = key
        val = self._data[i][j][k]
        self._cache[key] = val
        return val

    def __iter__(self):
        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                for k in range(len(self._data[i][j])):
                    yield (i, j, k)

    def __len__(self):
        return sum(len(middle) * len(middle[0]) for middle in self._data)

    def values(self):
        # Overrides Mapping.values() to lazily iterate through values
        for key in self:
            yield self[key]

    def items(self):
        for key in self:
            yield key, self[key]


def _mat_to_tensor(mat,domain,codomain):
    if not isinstance(mat,sp.Matrix):
        return mat
    if domain.dimension!=mat.shape[0] or codomain.dimension!=mat.shape[1]:
        raise TypeError('`mat` should be a r-by-s matrix where domain and codomain have dimensions r and s.')
    tp=0
    for j in range(domain.dimension):
        for k in range(codomain.dimension):
            tp+=mat[j,k]*codomain.basis[k]@domain.basis[j]
    return tp

def _generate_gl_structure_data_caching(vs):
    n = len(vs.basis) - 1
    N = n + 1

    h_keys = []        # (j,j,0)
    od_keys = []       # (j,k,1) and (k,j,1)
    grading = {}       # key -> weight vector

    def elemWeights(i1, i2):
        w = []
        for idx in range(n):
            if i1 <= idx:
                w.append(0 if i2 <= idx else 1)
            else:
                w.append(-1 if i2 <= idx else 0)
        return w

    # Build keys + grading only (no matrices here)
    for j in range(N):
        for k in range(j, N):
            if j == k:
                h_keys.append((j, j, 0))
                grading[(j, j, 0)] = [0] * n
            else:
                od_keys.append((j, k, 1))
                grading[(j, k, 1)] = elemWeights(j, k)
                od_keys.append((k, j, 1))
                grading[(k, j, 1)] = elemWeights(k, j)

    # Indexing
    ordering = h_keys + od_keys
    indexingKey = dict(enumerate(ordering))
    indexingKeyRev = {t: i for i, t in indexingKey.items()}
    LADimension = len(indexingKey)

    # Lazy structure constants
    def _structureCoeffs(i1, i2):
        if i2 == i1: 
            return {}
        reSign = 1
        if i2 < i1: 
            reSign = -1
            i2, i1 = i1, i2
        p10,p11,p12 = indexingKey[i1]
        p20,p21,p22 = indexingKey[i2]
        out = {}
        if p12 == 0:
            if p22 == 1 and p10 != n:  # (n,n,0) central in gl(N)
                val = reSign*(int(p10==p20) - int(p10==p21) + int(p10+1==p21) - int(p10+1==p20))
                if val: 
                    out[i2] = val
        elif p12 == 1 and p22 == 1:
            if p11 == p20:
                if p10 == p21:
                    if p10 < p11:
                        for idx in range(p10, p11): 
                            out[indexingKeyRev[(idx,idx,0)]] = reSign
                    else:
                        for idx in range(p11, p10): 
                            out[indexingKeyRev[(idx,idx,0)]] = -reSign
                else:
                    out[indexingKeyRev[(p10,p21,1)]] = reSign
            elif p10 == p21:
                out[indexingKeyRev[(p20,p11,1)]] = -reSign
        return out

    sd = algebra_structure_data(LADimension, _structureCoeffs, shape="skew", cache_max_pairs=50000)

    # Eager pieces you asked to keep
    # CartanSubalg = diagonal H_j; gradingVecs_T = zip(*grading.values())
    def H_diag(j):
        # j < n: diag = 1 - (j+1)/(n+1) on indices <= j, and -(j+1)/(n+1) on > j
        # j = n: identity
        M = [[0]*N for _ in range(N)]
        if j == n:
            for a in range(N): 
                M[a][a] = 1
        else:
            frac = sp.Rational(j+1, n+1)
            for a in range(N): 
                M[a][a] = (1-frac) if a <= j else -frac
        return M

    CartanSubalg = [H_diag(j) for (j,_,_) in h_keys]  # eager by request
    gradingVecs_T = list(zip(*[grading[key] for key in ordering]))

    # Fully lazy representation data
    class algebra_rep_data:
        def __init__(self, dim, *, cache_max=50000):
            self.dim = dim
            @lru_cache(maxsize=cache_max)
            def _matrix(i):
                j,k,t = indexingKey[i]
                if t == 0: 
                    return H_diag(j)
                M = [[0]*N for _ in range(N)]
                M[j][k] = 1
                return M
            @lru_cache(maxsize=cache_max)
            def _operator(i):
                j,k,t = indexingKey[i]
                if t == 0:
                    if j < n:
                        frac = sp.Rational(j+1, n+1)
                        tp = (1-frac)*vs.basis[0]@(vs.basis[0].dual())
                        for a in range(1, N):
                            tp += (-(frac) if a>j else (1-frac))*vs.basis[a]@(vs.basis[a].dual())
                        return tp
                    return sum([vs.basis[a]@(vs.basis[a].dual()) for a in range(n)], vs.basis[n]@(vs.basis[n].dual()))
                return vs.basis[j]@(vs.basis[k].dual())
            self._matrix = _matrix
            self._operator = _operator
        def __getitem__(self, mode):
            if mode == "matrix": 
                return _RepModeView(self, "matrix")
            if mode == "operator": 
                return _RepModeView(self, "operator")
            raise KeyError("use ['matrix'] or ['operator']")
        def clear_cache(self): 
            self._matrix.cache_clear()
            self._operator.cache_clear()
        def stats(self): return {"dim": self.dim, "matrix_cached": self._matrix.cache_info().currsize, "operator_cached": self._operator.cache_info().currsize}
    class _RepModeView:
        __slots__ = ("_rep","_mode")
        def __init__(self, rep, mode): self._rep, self._mode = rep, mode
        def __getitem__(self, i): 
            if not (0 <= i < self._rep.dim):
                raise IndexError(f"index {i} out of bounds for dim={self._rep.dim}")
            return self._rep._matrix(i) if self._mode == "matrix" else self._rep._operator(i)
        def __len__(self): return self._rep.dim
        def __iter__(self): 
            for i in range(self._rep.dim): 
                yield self[i]
        def __repr__(self): return f"algebra_rep_view(mode={self._mode}, dim={self._rep.dim})"

    rep = algebra_rep_data(LADimension, cache_max=50000)

    return sd, gradingVecs_T, CartanSubalg, rep
