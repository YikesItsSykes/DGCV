"""
DGCV: Differential Geometry with Complex Variables

This module defines the core classes and functions for the DGCV package, handling the creation 
and manipulation of vector fields, differential forms, tensor fields, algebras, and more. It includes 
tools for managing relationships between real and complex coordinate systems, object creation function,
and basic operations for its classes.

Key Classes:
    - DFClass: Represents differential forms.
    - VFClass: Represents vector fields.
    - STFClass: Represents symmetric tensor fields.
    - TFClass: Represents general tensor fields.
    - DGCVPolyClass: Hooks DGCV's complex variable handling into SymPy's Poly tools.

Key Functions:

Object Creation:
    - createVariables(): Initializes and labels coordinate systems of various types and 
    registers them within DGCV's Variable Management Framework.

Coordinate Conversion:
    - holToReal(): Converts holomorphic coordinates to their real counterparts.
    - realToSym(): Converts real coordinates to holomorphic with symbolic conjugate representations.
    - symToHol(): Converts symbolic conjugate coordinate representations to standard holomorphic coordinates.
    - holToSym(): Converts holomorphic coordinates to holomorphic with symbolic conjugate representations.
    - realToHol(): Converts real coordinates to holomorphic coordinates.
    - symToReal(): Converts symbolic conjugate coordinate representations to real coordinates.
    - allToReal(): Converts all types of coordinates to real coordinates.
    - allToHol(): Converts all types of coordinates to holomorphic coordinates.
    - allToSym(): Converts all types of coordinates to holomorphic with symbolic conjugate representations.

Structure Operations:
    - complex_struct_op(): Applies the complex structure operator to a vector field.
    - changeVFBasis(): Changes the coordinates space of a vector field.
    - changeDFBasis(): Changes the coordinates space of a differential form.
    - changeTensorFieldBasis(): Changes the coordinates space of a tensor field.
    - changeSTFBasis(): Changes the coordinates space of a symmetric tensor field.
    - VF_bracket(): Computes the Lie bracket of two vector fields.
    - exteriorProduct(): Computes the exterior product of differential forms.
    - tensor_product(): Computes the tensor product of tensor fields.
    - scaleDF(): Scales a differential form.
    - addDF(): Adds two differential forms.
    - addVF(): Adds two vector fields.
    - scaleVF(): Scales a vector field.
    - addSTF(): Adds two symmetric tensor fields.
    - addTF(): Adds two tensor fields.
    - scaleTF(): Scales a tensor field.

Coefficients and Decompositions:
    - VF_coeffs(): Returns the coefficients of a vector field w.r.t. given coordinate vector 
    fields.
    - holVF_coeffs(): Returns the holomorphic coefficients of a vector field w.r.t. given 
    coordinate vector fields.
    - antiholVF_coeffs(): Returns the antiholomorphic coefficients of a vector field w.r.t. 
    given coordinate vector fields.
    - complexVFC(): Returns the holomorphic and antiholomorphic coefficients of a vector 
    field w.r.t. given coordinate vector fields.
    - realPartOfVF(): Extracts the real part of a vector field.

Variable Management:
    - listVar(): Lists the "parent names" of objects currently tracked by the DGCV VMF.
    - clearVar(): Clears the variables from the DGCV registry and deletes them from caller's globals().
    - DGCV_snapshot(): Takes a snapshot of the current DGCV VMF and reports a summary in a
    Pandas table.

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy
    - pandas

License:
    MIT License
"""

############## dependencies
import itertools
import re
import warnings

import pandas as pd
import sympy as sp
from IPython.display import display
from pandas import DataFrame
from sympy import I

from ._safeguards import (
    protected_caller_globals,
    retrieve_passkey,
    retrieve_public_key,
    validate_label,
)
from ._tensor_field_printers import tensor_field_latex, tensor_field_printer
from .combinatorics import carProd, permSign
from .config import _cached_caller_globals, get_variable_registry, greek_letters
from .styles import get_style
from .vmf import _coeff_dict_formatter

############## classes

class TFClass:
    def __init__(self):
        pass


class tensorField(sp.Basic):
    def __new__(cls, varSpace, coeff_dict, valence=None, data_shape="general",
                DGCVType="standard", _simplifyKW=None):
        varSpace = tuple(varSpace)

        # Validate coeff_dict
        if not isinstance(coeff_dict, dict):
            raise TypeError("`coeff_dict` must be a dictionary.")

        # Determine valence if not provided
        if valence is None:
            if not coeff_dict:
                valence = tuple()
            else:
                first_key = next(iter(coeff_dict))
                if not isinstance(first_key, tuple):
                    raise TypeError("Keys in `coeff_dict` must be tuples representing tensor indices.")
                valence = (0,) * len(first_key)  # Default to all covariant

        valence = tuple(valence)

        # Validate valence structure
        if not all(v in (0, 1) for v in valence):
            raise ValueError("`valence` must contain only 0s and 1s.")

        # Validate data_shape compatibility
        if len(set(valence)) > 1 and data_shape in ['symmetric', 'skew']:
            raise ValueError(f"Mixed type/valence and `data_shape={data_shape}` is not supported.")

        # Default handling for _simplifyKW
        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }

        # Process coefficient dictionary
        total_degree = len(valence)
        processed_coeff_dict = cls._process_coeffs_dict(coeff_dict, data_shape, total_degree)

        # Create immutable instance and set valence as an attribute
        obj = sp.Basic.__new__(cls, varSpace, processed_coeff_dict, valence, data_shape, DGCVType, _simplifyKW)
        obj.valence = valence  # Now valence is set in __new__

        return obj

    def __init__(self, varSpace, coeff_dict, valence=None, data_shape="general",
                 DGCVType="standard",
                 _simplifyKW=None):
        self.varSpace = varSpace
        self.data_shape = data_shape
        self.DGCVType = DGCVType

        self.total_degree = len(self.valence)
        self.contravariant_degree = sum(self.valence)
        self.covariant_degree = self.total_degree - self.contravariant_degree

        self.coeff_dict = coeff_dict  

        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }
        self._simplifyKW = _simplifyKW

        self._preferred_basis_element = self._compute_preferred_basis_element()

        # Determine variable space type for complex cases
        self._set_varSpace_type()

        # Initialize caches
        self._coeffArray = None
        self._expanded_coeff_dict = None
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._cd_formats = None

    def _set_varSpace_type(self):
        """Determine the type of variable space (real, complex, or standard)."""
        variable_registry = get_variable_registry()
        if self.DGCVType == "complex":
            if all(var in variable_registry["conversion_dictionaries"]["realToSym"] for var in self.varSpace):
                self._varSpace_type = "real"
            elif all(var in variable_registry["conversion_dictionaries"]["symToReal"] for var in self.varSpace):
                self._varSpace_type = "complex"
            else:
                raise KeyError(
                    f"To initialize a `tensorField` instance with `DGCVType='complex'`, `varSpace` must contain only variables from DGCV's complex variable systems. All variables in `varSpace` must be either simultaneously among the real and imaginary types, or simultaneously among the holomorphic and antiholomorphic types. \n Recieved: {self.varSpace}\n Use `createVariables` to easily create DGCV\'s complex coordinate systems."
                )
        else:
            self._varSpace_type = "standard"

    def _compute_preferred_basis_element(self, _simplifyKW=None):
        """
        Compute the preferred basis element for printing trivial tensor fields.

        Parameters
        ----------
        _simplifyKW : dict, optional
            Dictionary containing "preferred_basis_element" key.

        Returns
        -------
        list of str
        """
        # Determine preferred basis indices
        if _simplifyKW is None or _simplifyKW.get("preferred_basis_element") is None:
            if self.varSpace:
                preferred_basis_indices = [i % len(self.varSpace) for i in range(self.total_degree)]
            else:
                preferred_basis_indices = [0 for _ in range(self.total_degree)]
        else:
            preferred_basis_indices = _simplifyKW["preferred_basis_element"]
            if not (isinstance(preferred_basis_indices, (list, tuple)) and len(preferred_basis_indices) == self.total_degree):
                raise TypeError(
                    "`preferred_basis_element` must be None or a list/tuple matching the total degree of the tensor field."
                )
            if not all(0 <= j < len(self.varSpace) for j in preferred_basis_indices):
                raise ValueError(
                    "`preferred_basis_element` contains invalid indices."
                )

        # Generate labels for contravariant and covariant components
        if self.varSpace:
            frameLabels = [f"D_{j}" for j in self.varSpace]
            coframeLabels = [f"d_{j}" for j in self.varSpace]
        else:
            frameLabels = ["NULL"]
            coframeLabels = ["NULL"]


        # Choose labels based on valence
        return [
            frameLabels[preferred_basis_indices[i]] if self.valence[i] == 1 else coframeLabels[preferred_basis_indices[i]]
            for i in range(self.total_degree)
        ]

    @staticmethod
    def _process_coeffs_dict(data, shape, total_degree):
        """
        Process the data_dict based on the specified shape.
        For recognized shapes, use the appropriate formatter.
        Defaults to general handling for unrecognized shapes.
        """
        if shape == "symmetric":
            return tensorField._format_symmetric_data(data,total_degree)
        elif shape == "skew":
            return tensorField._format_skew_data(data,total_degree)
        else:
            # Remove key-value pairs where value == 0
            processed_data = {key: value for key, value in data.items() if value != 0}

            # If all values are removed, fallback to default 0 dict
            if not processed_data:
                return {(0,) * total_degree: 0}

            return processed_data

    @staticmethod
    def _format_symmetric_data(data,total_degree):
        coeffs = {}
        for key, value in data.items():
            # Sort the key to enforce symmetry
            sorted_key = tuple(sorted(key))
            if sorted_key in coeffs:
                # Check for consistency in values
                if sp.simplify(coeffs[sorted_key]) != sp.simplify(value):
                    raise TypeError(
                        "The tensorField initializer was given non-symmetric data"
                        "but it was called with `data_shape='symmetric'`."
                    )
            else:
                coeffs[sorted_key] = value

        # If empty fallback to a default zero dict
        if not coeffs:
            coeffs = {(0,) * total_degree: 0}

        return coeffs

    @staticmethod
    def _format_skew_data(data, total_degree):
        formatted_data = {}
        for key, value in data.items():
            # Determine the sign and sorted form of the key
            perm_sign, sorted_key = permSign(key, returnSorted=True)
            sorted_key = tuple(sorted_key)

            # Check for consistency in values
            if sorted_key in formatted_data:
                if sp.simplify(formatted_data[sorted_key]) != sp.simplify(perm_sign * value):
                    raise TypeError(
                        "The tensorField initializer was given non-skew-symmetric data"
                        " but it was called with `data_shape='skew'`."
                    )
            else:
                if len(set(sorted_key)) < len(sorted_key) and value != 0:
                    raise TypeError(
                        "The tensorField initializer was given non-skew-symmetric data"
                        " but it was called with `data_shape='skew'`."
                    )
                formatted_data[sorted_key] = perm_sign * value

        # For empty list fallback to a default zero value
        if not formatted_data:
            formatted_data = {(0,) * total_degree: 0}

        return formatted_data

    @property
    def expanded_coeff_dict(self):
        if self._expanded_coeff_dict:
            return self._expanded_coeff_dict
        if self.data_shape == 'symmetric':

            def expand_method_A(coeff_dict):
                expanded_dict = {}
                for key, value in coeff_dict.items():
                    for perm in set(itertools.permutations(key)):  # Use set to avoid duplicates
                        expanded_dict[perm] = value
                return expanded_dict
            def expand_method_B(coeff_dict, dimension, total_degree):
                expanded_dict = {}
                for index_tuple in itertools.product(range(dimension), repeat=total_degree):
                    sorted_tuple = tuple(sorted(index_tuple))
                    if sorted_tuple in coeff_dict:
                        expanded_dict[index_tuple] = coeff_dict[sorted_tuple]
                return expanded_dict

            # Use different method depending on data density
            if len(self.coeff_dict) * sp.factorial(self.total_degree) < len(self.varSpace)**(self.total_degree):
                # Use Method A
                self._expanded_coeff_dict = expand_method_A(self.coeff_dict)
            else:
                # Use Method B
                self._expanded_coeff_dict = expand_method_B(self.coeff_dict, len(self.varSpace), self.total_degree)
            return self._expanded_coeff_dict

        if self.data_shape == 'skew':

            def expand_method_A(coeff_dict):
                expanded_dict = {}
                for key, value in coeff_dict.items():
                    for perm in set(itertools.permutations(key)):  # Use set to avoid duplicates
                        expanded_dict[perm] = permSign(perm)*value
                return expanded_dict
            def expand_method_B(coeff_dict, dimension, total_degree):
                expanded_dict = {}
                for index_tuple in itertools.product(range(dimension), repeat=total_degree):
                    sorted_tuple = tuple(sorted(index_tuple))
                    if sorted_tuple in coeff_dict:
                        expanded_dict[index_tuple] = permSign(index_tuple)*coeff_dict[sorted_tuple]
                return expanded_dict

            # Use different method depending on data density
            if len(self.coeff_dict) * sp.factorial(self.total_degree) < len(self.varSpace)**(self.total_degree):
                # Use Method A
                self._expanded_coeff_dict = expand_method_A(self.coeff_dict)
            else:
                # Use Method B
                self._expanded_coeff_dict = expand_method_B(self.coeff_dict, len(self.varSpace), self.total_degree)
            return self._expanded_coeff_dict

        self._expanded_coeff_dict = self.coeff_dict
        return self._expanded_coeff_dict

    @property
    def coeffArray(self):
        """
        Returns the tensor coefficients as a SymPy Immutable Sparse Array.
        """
        if self._coeffArray is None:
            def entry_rule(indexTuple):
                """
                Retrieves the coefficient for a given index tuple.
                sorts if symmetric
                """
                sortedTuple = tuple(sorted(indexTuple)) if self.data_shape == "symmetric" else indexTuple
                return self.expanded_coeff_dict.get(sortedTuple, 0)

            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [
                        (i,) + t
                        for i in range(shape[0])
                        for t in generate_indices(shape[1:])
                    ]

            # Tensor shape is determined by the number of variables and tensor rank (valence length)
            shape = (len(self.varSpace),) * len(self.valence)

            # Build the sparse data dictionary
            sparse_data = {
                indices: entry_rule(indices) for indices in generate_indices(shape)
            }

            # Store as immutable sparse array
            self._coeffArray = sp.ImmutableSparseNDimArray(sparse_data, shape)

        return self._coeffArray

    def __str__(self):
        return tensor_field_printer(self)

    def __repr__(self):
        return f"tensorField(varSpace={self.varSpace}, valence={self.valence}, coeffs={dict(self.coeff_dict)})"

    def _repr_latex_(self):
        """
        Define how the tensorField is displayed in LaTeX in IPython.
        """
        return tensor_field_latex(self)

    def _sympystr(self, printer):
        return self.__repr__()

    def _latex(self, printer=None):
        return self._repr_latex_()

    @property
    def realVarSpace(self):
        if self.DGCVType == "standard": # Do nothing for DGCVType == "standard"
            return self._realVarSpace
        if self._realVarSpace is None or self._imVarSpace is None:
            self.cd_formats
            return self._realVarSpace + self._imVarSpace
        return self._realVarSpace + self._imVarSpace

    @property
    def holVarSpace(self):
        if self.DGCVType == "standard": # Do nothing for DGCVType == "standard"
            return self._holVarSpace
        if self._holVarSpace is None:
            self.cd_formats
            return self._holVarSpace
        return self._holVarSpace

    @property
    def antiholVarSpace(self):
        if self.DGCVType == "standard": # Do nothing for DGCVType == "standard"
            return self._antiholVarSpace
        if self._antiholVarSpace is None:
            self.cd_formats
            return self._antiholVarSpace
        return self._antiholVarSpace

    @property
    def compVarSpace(self):
        if self.DGCVType == "standard": # Do nothing for DGCVType == "standard"
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace is None or self._antiholVarSpace is None:
            self.cd_formats
            return self._holVarSpace + self._antiholVarSpace
        return self._holVarSpace + self._antiholVarSpace

    @property
    def cd_formats(
        self,
    ):  # Retrieves coeffs in different variable formats and updates *VarSpace and _cd_formats caches if needed
        if self._cd_formats:
            return self._cd_formats

        if self.DGCVType == "standard" or all(
            j is not None
            for j in [
                self._realVarSpace,
                self._holVarSpace,
                self._antiholVarSpace,
                self._imVarSpace,
                self._cd_formats,
            ]
        ):
            return self._cd_formats
        populate,self._realVarSpace,self._holVarSpace,self._antiholVarSpace,self._imVarSpace = _coeff_dict_formatter(self.varSpace,self.coeff_dict,self.valence,self.total_degree,self._varSpace_type,self.data_shape)

        if self._cd_formats is None:
            self._cd_formats = populate
            return populate
        else:
            return self._cd_formats

    def _eval_simplify(self, **kwargs):
        """
        Applies simplification rules based on the current `_simplifyKW` settings.

        Returns
        -------
        tensorField
            A simplified tensorField object.
        """
        if self._simplifyKW["simplify_rule"] is None:
            # Apply standard simplification to all tensor coefficients
            simplified_coeffs = {
                key: sp.simplify(value, **kwargs)
                for key, value in self.expanded_coeff_dict.items()
            }

        elif self._simplifyKW["simplify_rule"] == "holomorphic":
            # Convert coefficients to holomorphic form before simplifying
            simplified_coeffs = {
                key: sp.simplify(
                    allToHol(value, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for key, value in self.expanded_coeff_dict.items()
            }

        elif self._simplifyKW["simplify_rule"] == "real":
            # Convert coefficients to real form before simplifying
            simplified_coeffs = {
                key: sp.simplify(
                    allToReal(value, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for key, value in self.expanded_coeff_dict.items()
            }

        elif self._simplifyKW["simplify_rule"] == "symbolic_conjugate":
            # Convert coefficients to symbolic conjugate before simplifying
            simplified_coeffs = {
                key: sp.simplify(
                    allToSym(value, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for key, value in self.expanded_coeff_dict.items()
            }

        else:
            warnings.warn(
                f"_eval_simplify received an unsupported simplify_rule: {self._simplifyKW['simplify_rule']}."
                " Recommended values: None, 'holomorphic', 'real', or 'symbolic_conjugate'."
            )
            simplified_coeffs = {
                key: sp.simplify(value, **kwargs)
                for key, value in self.expanded_coeff_dict.items()
            }

        # Return a new instance of tensorField with simplified coefficients
        return tensorField(
            self.varSpace,
            simplified_coeffs,
            valence=self.valence,
            data_shape=self.data_shape,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW,
        )

    def __eq__(self, other):
        """
        Checks if two tensorField instances are equal.

        Two tensorFields are considered equal if:
        - They have the same variable space (`varSpace`).
        - They have the same valence.
        - They have the same data shape.
        - They have the same DGCVType.
        - Their coefficient dictionaries simplify to the same values.
        """
        if not isinstance(other, tensorField):
            return False

        return (
            self.varSpace == other.varSpace
            and self.valence == other.valence
            and self.data_shape == other.data_shape
            and self.DGCVType == other.DGCVType
            and all(
                sp.simplify(allToReal(self.coeff_dict[key])) == sp.simplify(allToReal(other.coeff_dict.get(key, 0)))
                for key in set(self.coeff_dict.keys()).union(set(other.coeff_dict.keys()))
            )
        )


    def __hash__(self):
        """
        Computes a hash value for the tensorField instance.

        The hash is based on:
        - `varSpace` (tuple of variables)
        - `valence` (tuple indicating index types)
        - `data_shape` (symmetric, skew, etc.)
        - `DGCVType` (standard, complex, etc.)
        - The **simplified** coefficient dictionary as a tuple of (key, value) pairs.
        """
        simplified_coeffs = tuple(
            sorted(
                (key, sp.simplify(allToReal(value)))
                for key, value in self.coeff_dict.items()
            )
        )

        return hash((self.varSpace, self.valence, self.data_shape, self.DGCVType, simplified_coeffs))


    def is_zero(self):
        return all(sp.simplify(allToReal(value)) == 0 for value in self.coeff_dict.values())

    def subs(self, substitutions):
        substituted_coeff_dict = {
            key: sp.sympify(value).subs(substitutions) for key, value in self.coeff_dict.items()
        }
        return tensorField(
            self.varSpace,
            substituted_coeff_dict,
            self.valence,
            data_shape=self.data_shape,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW
        )

    def tensor_product(self, *others):
        """
        Compute the tensor product of self with anynumber of tensorField instances.
        """
        return tensor_product(self, *others)

    def tp(self, *others):
        """
        Alias for tensor_product for easier typing.
        """
        return self.tensor_product(*others)

    def __matmul__(self, other):
        """
        Overload `@` for the tensor product.
        """
        return tensor_product(self,other)

    def __add__(self, other):
        """
        Adds two tensorField instances using addTensorFields.
        """
        if isinstance(other, tensorField):
            return addTensorFields(self, other)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: `tensorField' and '{type(other).__name__}'")

    def __sub__(self, other):
        """
        Subtracts another tensorField instance from the current instance using addTensorFields.
        """
        if isinstance(other, tensorField):
            return addTensorFields(self, scaleTensorField(-1,other))
        else:
            raise TypeError(f"Unsupported operand type(s) for -: `tensorField' and '{type(other).__name__}'")

    def __neg__(self):
        """
        Returns
        -------
        tensorField
            A new tensorField instance representing the negation of the original.
        """
        return scaleTensorField(-1, self)

    def __mul__(self, other):
        """
        Defines multiplication for tensorField objects.

        - If `other` is a scalar (int, float, sympy expression), the tensor is scaled.
        - If `other` is another `tensorField` their tensor product is returned
        """
        if isinstance(other, (int, float, sp.Expr)):
            return scaleTensorField(other,self)

        elif isinstance(other,tensorField):
            if self.data_shape == 'skew' and other.data_shape == 'skew':
                warnings.deprecated('`*` for tensor products is depricated. Use `@` instead. `*` was applied to multiply a pair of skew symmetric tensorField instances of which at least one was not a DFClass. A usual tensor product was therefore returned. If a wedge product was intended then make sure to initialize arguments as DFClass instances (a TFClass subclass).')
            return self.__matmul__(other)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: `tensorField` and `{type(other).__name__}`")

    def __rmul__(self, scalar):
        """
        Allows scalar multiplication on the left-hand side.
        """
        return self.__mul__(scalar)

    def __call__(self, *tfList):
        """
        Pair tensor field with a list of tensor fields or dual valence

        Parameters
        ----------
        tfList : list of tensorField instances
            The tensor fields to pair with. Must have length equal to len(self.valence),
            and each tfList[j] must have the opposite valence of self.valence[j].

        Returns
        -------
        scalar : sympy expression or tensorField
        """
        if len(tfList) != len(self.valence):
            raise ValueError(
                f"Expected {len(self.valence)} tensor fields for contraction, but got {len(tfList)}."
            )

        # tfList should contain only degree-1 tensors
        for j, tf in enumerate(tfList):
            if not isinstance(tf, tensorField):
                raise TypeError(f"Expected a tensorField at position {j}, got {type(tf).__name__}.")
            if len(tf.valence) != 1:
                raise ValueError(
                    f"tensorField contracts only with degree-1 tensors, but got degree {len(tf.valence)} at position {j}."
                )
            if self.valence[j] == tf.valence[0]:  # tf.valence is a tuple, so access index 0
                raise ValueError(
                    f"Tensor valence mismatch: expected opposite valence at position {j}."
                )

        # Determine if varSpaces are different
        all_varSpaces = [self.varSpace] + [tf.varSpace for tf in tfList]
        if len(set(all_varSpaces)) > 1:  # If there are different varSpaces, reformat
            # Compute minimal new_varSpace
            new_varSpace = list(self.varSpace)
            for tf in tfList:
                for var in tf.varSpace:
                    if var not in new_varSpace:
                        new_varSpace.append(var)
            new_varSpace = tuple(new_varSpace)

            # Format TF w.r.t. new_varSpace
            new_self = changeTensorFieldBasis(self, new_varSpace)
            new_tfList = [changeTensorFieldBasis(tf, new_varSpace) for tf in tfList]

            # Recursively call __call__ on transformed TF
            return new_self.__call__(*new_tfList)

        # Initialize result
        contracted_result = 0  # This will accumulate the sum of contractions

        # Use expanded_coeff_dict instead of coeff_dict
        for key_self, coeff_self in self.expanded_coeff_dict.items():
            # Compute contraction term-by-term
            term_value = coeff_self
            contraction_indices = list(key_self)

            for j, tf in enumerate(tfList):
                if term_value == 0:
                    break  # Stop processing early if multiplication will result in zero

                tf_coeff_dict = tf.expanded_coeff_dict
                key_to_lookup = (contraction_indices[j],)

                if key_to_lookup in tf_coeff_dict:
                    term_value *= tf_coeff_dict[key_to_lookup]
                else:
                    term_value *= 0

            contracted_result += term_value  # Accumulate the sum

        return contracted_result

# differential form class
class DFClass(tensorField):
    """
    A class representing symbolic differential forms in the DGCV package, with support for standard and complex differential forms.

    The DFClass represents differential forms where each component is a symbolic expression (using SymPy expressions)
    corresponding to a differential form in the given variable space. It supports standard differential forms,
    as well as complex differential forms with automatic handling of holomorphic and anti-holomorphic variables.

    Parameters:
    -----------
    varSpace : list
        The variables (coordinates) with respect to which the differential form is defined.

    coeffs : dict
        The coefficients for each variable combination in the differential form. The keys of the dictionary are tuples
        representing the indices of the variables, and the values are the coefficients.

    degree : int
        The degree of the differential form (number of variables in each wedge product).

    DGCVType : str, optional, default='standard'
        The type of differential form: 'standard' for real differential forms or 'complex' for complex differential forms.

    holVarSpace : list, optional
        For complex differential forms, specifies the holomorphic variable space. If not provided, it will be inferred
        from `varSpace` using the `variable_registry`.

    compVarSpace : list, optional
        For complex differential forms, specifies the full variable space including both holomorphic and conjugate variables.
        If not provided, it will be inferred from `varSpace`.

    Examples
    --------
    >>> from DGCV import DFClass, variableProcedure, DGCV_init_printing()
    >>> variableProcedure(['x', 'y'])
    >>> df = DFClass([x, y], {(0,): 1, (1,): -1}, 1)
    >>> df
    DFClass([x, y], {(0,): 1, (1,): -1}, 1)

    >>> df_latex = df._repr_latex_()  # Custom LaTeX formatting
    >>> print(df_latex) # display copy-able LaTex
    >>> DGCV_init_printing() # Nicely formats all DGCV and sympy objects displayed in Jupyter (or an iPython shell)
    >>> display(df) # renders nicely format differential form display in Jupyter (or an iPython shell)

    Attributes:
    -----------
    varSpace : list
        The variables with respect to which the differential form is defined.

    coeffs : dict
        The coefficients for each variable combination in the differential form.

    degree : int
        The degree of the differential form.

    DGCVType : str
        Indicates whether the differential form is standard ('standard') or complex ('complex').

    holVarSpace : tuple
        The holomorphic variable space for complex differential forms.

    compVarSpace : tuple
        The complete variable space (holomorphic and conjugate variables) for complex differential forms.

    Methods
    -------
    _repr_latex_ :
        Returns a custom LaTeX representation of the differential form, useful for rendering in Jupyter notebooks.

    __add__ :
        Custom addition method to allow for differential form addition.

    __sub__ :
        Custom subtraction method to allow for differential form subtraction.

    __mul__ and __rmul__ :
        Custom multiplication methods to scale differential forms by a scalar.

    __call__ :
        Apply the differential form to a set of vector fields, with optional handling for complex variables.

    Raises
    ------
    TypeError :
        Raised if operations (e.g., addition, subtraction) are attempted between incompatible types.

    ValueError :
        Raised if the differential form is applied to an unsupported argument type (e.g., incorrect number of arguments).
    """

    def __new__(
        cls,
        varSpace,
        data_dict,
        degree,
        DGCVType="standard",
        _simplifyKW={
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        },
    ):
        # Validate the inputs
        if not isinstance(varSpace, (list, tuple)):
            raise TypeError("`varSpace` must be a list or tuple.")
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("`degree` must be a non-negative integer.")
        if len(varSpace) != len(set(varSpace)):
            raise ValueError("`varSpace` must not have duplicate entries.")
        if not isinstance(data_dict, dict):
            raise TypeError("`data_dict` must be a dictionary.")

        # Set valence to (0, degree) for purely covariant tensors
        valence = (0,) * degree

        # Call the parent class's __new__ method
        return super().__new__(
            cls,
            varSpace,
            data_dict,
            valence,
            data_shape="skew",
            DGCVType=DGCVType,
            _simplifyKW=_simplifyKW,
        )

    def __init__(
        self,
        varSpace,
        data_dict,
        degree,
        DGCVType="standard",
        _simplifyKW={
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        },
    ):
        # Initialize the parent tensorField
        super().__init__(
            varSpace,
            data_dict,
            (0,) * degree,
            data_shape="skew",
            DGCVType=DGCVType,
            _simplifyKW=_simplifyKW,
        )

        # Any DFClass-specific initialization
        self.degree = degree  # For backward compatibility

        self.DFClassDataDict = self.coeff_dict

        self.DFClassDataMinimal = [
            [list(a), b] for a, b in self.DFClassDataDict.items() if b!=0
        ]
        self.coeffsInKFormBasis = [j for _, j in self.DFClassDataMinimal]
        if self.DFClassDataMinimal == []:
            self.DFClassDataMinimal = [
                (
                    [
                        0,
                    ]
                    * degree,
                    0,
                )
            ]

        if self.degree == 0:
            self.coeffsInKFormBasis = [j[1] for j in self.DFClassDataMinimal]
            self.kFormBasisGenerators = [[1]]
        else:
            oneFormsLabelsLoc = ["d_" + str(j) for j in self.varSpace]
            self.coeffsInKFormBasis = [j[1] for j in self.DFClassDataMinimal]
            self.kFormBasisGenerators = [
                [oneFormsLabelsLoc[k] for k in j[0]] for j in self.DFClassDataMinimal
            ]

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the differential dorm for custom simplification.

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.

        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.

        Returns
        -------
        DFClass
            A new DFClass instance with updated simplification settings.
        """
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            warnings.warn(
                "simplify_format() recieved an unsupported first argument. Try None, 'holomorphic', 'real',  or 'symbolic_conjugate' instead."
            )
        return DFClass(
            self.varSpace,
            self.DFClassDataDict,
            self.degree,
            DGCVType=self.DGCVType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
        )


    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self._simplifyKW attribute.

        Returns
        -------
        DFClass
            A simplified DFClass object.
        """
        if self._simplifyKW["simplify_rule"] is None:
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(b, **kwargs) for a, b in self.DFClassDataDict.items()
            }
        elif self._simplifyKW["simplify_rule"] == "holomorphic":
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(
                    allToHol(b, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for a, b in self.DFClassDataDict.items()
            }
        elif self._simplifyKW["simplify_rule"] == "real":
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(
                    allToReal(b, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for a, b in self.DFClassDataDict.items()
            }
        elif self._simplifyKW["simplify_rule"] == "symbolic_conjugate":
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(
                    allToSym(b, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for a, b in self.DFClassDataDict.items()
            }
        else:
            warnings.warn(
                "_eval_simplify recieved an unsupported DFClass._simplifyKW['simplify_rule']. It is recommend to only set the _simplifyKW['simplify_rule'] attribute to None, 'holomorphic', 'real',  or 'symbolic_conjugate'."
            )
            simplified_coeffs = {
                a: sp.simplify(b, **kwargs) for a, b in self.DFClassDataDict.items()
            }

        # Return a new instance of DFClass with simplified coeffs

        # Return a new instance of DFClass with simplified coeffs
        return DFClass(
            self.varSpace,
            simplified_coeffs,
            self.degree,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW,
        )

    def subs(self, subsData):
        newDFData = {
            a: sp.sympify(b).subs(subsData) for a, b in self.DFClassDataDict.items()
        }
        return DFClass(
            self.varSpace,
            newDFData,
            self.degree,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW,
        )

    def __add__(self, other):
        """
        Adds two differential forms (i.e., DFClass instances). The addition is performed
        by combining the coefficients of both forms. If the variable spaces of the forms
        are not the same, the resulting form will have a variable space that is the union
        of both input spaces.

        Parameters
        ----------
        other : DFClass
            Another differential form to add. The variable spaces of the two forms need not
            be identical. The result will use the union of the variable spaces of the two
            forms.

        Returns
        -------
        DFClass
            A new differential form representing the sum of the two input forms.

        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `DFClass`.

        Examples
        --------
        >>> from DGCV import DFClass, creatVariables
        >>> creatVariables('x1','x2')
        >>> d_x1 = DFClass([x1], {(0,): 1}, 1)
        >>> d_x2 = DFClass([x2], {(0,): 1}, 1)
        >>> d_sum = d_x1 + d_x2
        >>> print(d_sum)
        d_x1 + d_x2
        """
        if isinstance(other, DFClass):
            return addDF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with DFClass")

    def __sub__(self, other):
        """
        Subtracts one differential form from another (i.e., DFClass instances). The subtraction
        is performed by negating the second form and then adding it to the first form. If the
        variable spaces of the forms are not the same, the resulting form will have a variable
        space that is the sum of both input spaces.

        Parameters
        ----------
        other : DFClass
            Another differential form to subtract. The variable spaces of the two forms need not
            be identical. The result will use the sum of the variable spaces of the two forms.

        Returns
        -------
        DFClass
            A new differential form representing the difference between the two input forms.

        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `DFClass`.

        Examples
        --------
        >>> from DGCV import DFClass, creatVariables
        >>> creatVariables('x1','x2')
        >>> d_x1 = DFClass([x1], {(0,): 1}, 1)
        >>> d_x2 = DFClass([x2], {(0,): 1}, 1)
        >>> d_diff = d_x1 - d_x2
        >>> print(d_diff)
        d_x1 - d_x2
        """
        if isinstance(other, DFClass):
            return addDF(self, scaleDF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with DFClass")

    def __matmul__(self, other):
        """
        Overload `@` for the tensor product.
        """
        return super().__matmul__(other)

    def __mul__(self, other):
        """
        Multiplication for DFClass objects. If the argument is a scalar (SymPy expression or numeric),
        the form is scaled by the scalar. If the argument is another DFClass object,
        the exterior (wedge) product is computed.

        Parameters
        ----------
        arg1 : sympy expression, numeric, or DFClass
            The object to multiply by. Can be a scalar function (SymPy expression or numeric) or
            another differential form (DFClass instance).

        Returns
        -------
        DFClass
            The result of scaling the form by the scalar or the exterior product of the
            two differential forms.

        Raises
        ------
        TypeError
            If the argument is neither a scalar (SymPy expression, numeric) nor a DFClass instance.
        """
        if isinstance(other, DFClass):
            # If the argument is another differential form, compute the exterior product
            return exteriorProduct(self, other)
        elif isinstance(
            other, (sp.Expr, int, float)
        ):
            return scaleDF(other, self)
        elif isinstance(other,tensorField):
            return super().__mul__(other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: 'DFClass' and '{type(other).__name__}'"
            )

    def __rmul__(self, other):
        """
        Right multiplication for DFClass objects. If the argument is a scalar (SymPy expression or numeric),
        the form is scaled by the scalar. If the argument is another DFClass object,
        the exterior (wedge) product is computed.

        Parameters
        ----------
        arg1 : sympy expression, numeric, or DFClass
            The object to multiply by. Can be a scalar function (SymPy expression or numeric) or
            another differential form (DFClass instance).

        Returns
        -------
        DFClass
            The result of scaling the form by the scalar or the exterior product of the
            two differential forms.

        Raises
        ------
        TypeError
            If the argument is neither a scalar (SymPy expression, numeric) nor a DFClass instance.
        """
        if isinstance(other, DFClass):
            # If the argument is another differential form, compute the exterior product
            return exteriorProduct(other, self)
        elif isinstance(
            other, (sp.Expr, int, float)
        ):
            return scaleDF(other, self)
        elif isinstance(other,tensorField):
            return super().__mul__(other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: 'DFClass' and '{type(other).__name__}'"
            )

    def __neg__(self):
        return scaleDF(-1, self)

    def __call__(self, *VFArgs):
        if self.degree == len(VFArgs):
            if self.degree == 0:
                return self.coeffsInKFormBasis[0]
            else:
                if all(isinstance(var, VFClass) for var in VFArgs):
                    if self.DGCVType == "complex":
                        if self._varSpace_type == "complex":
                            VFArgs = [
                                (
                                    allToSym(vf)
                                    if vf._varSpace_type != self._varSpace_type
                                    else vf
                                )
                                for vf in VFArgs
                            ]
                        else:
                            VFArgs = [
                                (
                                    allToReal(vf)
                                    if vf._varSpace_type != self._varSpace_type
                                    else vf
                                )
                                for vf in VFArgs
                            ]
                    if self.DGCVType == "standard":
                        VFArgs = [_remove_complex_handling(vf) for vf in VFArgs]
                    inputCoeffs = contravariantVFTensorCoeffs(self.varSpace, *VFArgs)
                    # Access the elements of coeffArray directly
                    result = 0
                    for pair in inputCoeffs:
                        index, scalar = pair
                        permS, orderedList = permSign(index, returnSorted=True)
                        index = tuple(orderedList)
                        scalar = permS * scalar
                        if index in self.DFClassDataDict:
                            result += self.DFClassDataDict[index] * scalar
                    return result
                else:
                    return super().__call__(*VFArgs)
        else:
            raise ValueError(
                "A differential form received a number of arguments different from its degree."
            )

# vector field class
class VFClass(tensorField):
    """
    A class representing vector fields in the DGCV package, with support for VF defined w.r.t. standard and
    complex coordinate systems.

    Parameters
    ----------
    varSpace : list or tuple
        Variables (coordinates) defining the vector field's domain.

    coeffs : list
        Coefficients for each variable, matching the length of `varSpace`.

    DGCVType : str, optional
        Specifies the field type: 'standard' for real vector fields or 'complex' for complex ones (default is 'standard').

    _simplifyKW : dict, optional
        Simplification options:
        - 'simplify_rule': Simplification rule ('real', 'holomorphic', 'symbolic_conjugate', or None).
        - 'simplify_ignore_list': Variables to exclude from simplification (list of str labels from VMF).

    Methods
    -------
    simplify_format(format_type=None, skipVar=None)
        Sets simplification rules for the vector field.

    subs(subsData)
        Substitutes variables or expressions into the vector field's coefficients.

    __add__, __sub__
        Component-wise addition and subtraction of vector fields.

    __mul__, __rmul__
        Scalar multiplication of the vector field (supports SymPy expressions).

    __call__(arg, ignore_complex_handling=None)
        Computes the directional derivative of a scalar function.

    Examples
    --------
    >>> variableProcedure(['x', 'y'])
    >>> vf = VFClass(['x', 'y'], ['x**2', 'y**2'], 'standard')
    >>> simplify(vf)  # Simplified representation
    >>> vf('x**2 + y**2')  # Computes the derivative

    Complex Variables:
    >>> complexVarProc('z', 'x', 'y')
    >>> vf2 = VFClass(['z', 'BARz'], ['z**2', 'BARz**2'], 'complex')
    >>> simplify(vf2.simplify_format('real'))  # Converts coefficients to real form
    """
    def __new__(
        cls,
        varSpace,
        coeffs,
        DGCVType="standard",
        _simplifyKW={
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        },
    ):
        # Validate coefficients and varSpace lengths
        if len(varSpace) != len(coeffs):
            raise ValueError("`varSpace` and `coeffs` must have the same length.")

        # Convert coeffs to coeff_dict
        coeff_dict = {(i,): coeff for i, coeff in enumerate(coeffs)}

        # Set valence for vector field (degree 1 contravariant tensor)
        valence = (1,)

        # Call the parent class's __new__ with data_shape="skew"
        return super().__new__(
            cls,
            varSpace=varSpace,
            coeff_dict=coeff_dict,
            valence=valence,
            data_shape="skew",
            DGCVType=DGCVType,
            _simplifyKW=_simplifyKW,
        )

    def __init__(
        self,
        varSpace,
        coeffs,
        DGCVType="standard",
        _simplifyKW={
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        },
    ):
        # Validate that varSpace contains unique elements
        if len(varSpace) != len(set(varSpace)):
            raise TypeError(
                "`VFClass` expects the `varSpace` tuple not to have repeated variables."
            )

        # Store VFClass-specific attributes
        self.coeffs = coeffs

        # Initialize the parent tensorField
        super().__init__(
            varSpace=varSpace,
            coeff_dict={(i,): coeff for i, coeff in enumerate(coeffs)},
            valence=(1,),
            data_shape="skew",
            DGCVType=DGCVType,
            _simplifyKW=_simplifyKW,
        )

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the vector field for custom simplification.

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.

        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.

        Returns
        -------
        VFClass
            A new VFClass instance with updated simplification settings.
        """
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            warnings.warn(
                "simplify_format() recieved an unsupported first argument. Try None, 'holomorphic', 'real',  or 'symbolic_conjugate' instead."
            )
        return VFClass(
            self.varSpace,
            self.coeffs,
            DGCVType=self.DGCVType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
        )

    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self._simplifyKW attribute.

        Returns
        -------
        VFClass
            A simplified VFClass object.
        """
        if self._simplifyKW["simplify_rule"] is None:
            simplified_coeffs = [sp.simplify(j, **kwargs) for j in self.coeffs]
        elif self._simplifyKW["simplify_rule"] == "holomorphic":
            simplified_coeffs = [
                sp.simplify(
                    allToHol(j, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for j in self.coeffs
            ]
        elif self._simplifyKW["simplify_rule"] == "real":
            simplified_coeffs = [
                sp.simplify(
                    allToReal(j, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for j in self.coeffs
            ]
        elif self._simplifyKW["simplify_rule"] == "symbolic_conjugate":
            simplified_coeffs = [
                sp.simplify(
                    allToSym(j, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for j in self.coeffs
            ]
        else:
            warnings.warn(
                "_eval_simplify recieved an unsupported VFClass._simplifyKW['simplify_rule']. It is recommend to only set the _simplifyKW['simplify_rule'] attribute to None, 'holomorphic', 'real',  or 'symbolic_conjugate'."
            )
            simplified_coeffs = [sp.simplify(j, **kwargs) for j in self.coeffs]

        return VFClass(
            self.varSpace,
            simplified_coeffs,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW,
        )

    def subs(self, subsData):
        newCoeffs = [sp.sympify(j).subs(subsData) for j in self.coeffs]
        return VFClass(
            self.varSpace, newCoeffs, DGCVType=self.DGCVType, _simplifyKW=self._simplifyKW
        )

    def __add__(self, other):
        if isinstance(other, VFClass):
            return addVF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with VFClass")

    def __sub__(self, other):
        if isinstance(other, VFClass):
            return addVF(self, scaleVF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with VFClass")

    def __neg__(self):
        return scaleVF(-1, self)

    def __mul__(self, scalar):
        return scaleVF(scalar, self)

    def __rmul__(self, scalar):
        return scaleVF(scalar, self)

    def __call__(self, *args, ignore_complex_handling=None):
        if len(args) == 1:
            if isinstance(args[0], tensorField):  # Check if first arg is a tensorField
                return super().__call__(*args)
            arg = args[0]
            if (
                self._varSpace_type == "complex"
            ):  # this implies self.DGCVType=='complex'
                if isinstance(arg, DFClass) and arg.degree == 0:
                    arg = (arg.coeffsInKFormBasis[0],)
                return sum(
                    self.coeffs[j] * sp.diff(allToSym(arg), self.varSpace[j])
                    for j in range(len(self.coeffs))
                )
            else:
                if isinstance(arg, DFClass) and arg.degree == 0:
                    arg = (arg.coeffsInKFormBasis[0],)
                if self.DGCVType == "standard" or ignore_complex_handling:
                    return sum(
                        self.coeffs[j] * sp.diff(arg, self.varSpace[j])
                        for j in range(len(self.coeffs))
                    )
                elif self.DGCVType == "complex":
                    return sum(
                        self.coeffs[j] * sp.diff(allToReal(arg), self.varSpace[j])
                        for j in range(len(self.coeffs))
                    )

        else:
            raise ValueError(
                "A vector field received a number of arguments different from 1."
            )

# Symmetric tensor field class
class STFClass(tensorField):
    def __new__(
        cls,
        varSpace,
        data_dict,
        degree,
        DGCVType="standard",
        _simplifyKW=None,
    ):
        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }

        if len(varSpace) != len(set(varSpace)):
            raise TypeError("`STFClass` expects `varSpace` to contain unique variables.")

        valence = (0,) * degree

        obj = super().__new__(
            cls,
            varSpace,
            data_dict,
            valence,
            data_shape="symmetric",
            DGCVType=DGCVType,
            _simplifyKW=_simplifyKW,
        )

        return obj

    def __init__(self, varSpace, data_dict, degree, DGCVType="standard", _simplifyKW=None):
        # Call tensorField's initializer
        super().__init__(varSpace, data_dict, (0,) * degree, "symmetric", DGCVType, _simplifyKW)

        self.degree = degree

        # Process the STFClass-specific attributes
        self.STFClassDataDict = self.coeff_dict
        self.STFClassDataMinimal = [[a, b] for a, b in self.STFClassDataDict.items()]
        self.coeffsInKFormBasis = [j for _, j in self.STFClassDataMinimal]

        # Caches
        self._STFClassDataDictFull = None
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._cd_formats = None
        self._coeffArray = None

        # Generate k-form basis representation
        if self.degree == 0:
            self.coeffsInKFormBasis = [
                self.STFClassDataDict[a] for a in self.STFClassDataDict
            ]
            self.kFormBasisGenerators = [[1]]
        else:
            oneFormsLabelsLoc = ["d_" + str(j) for j in self.varSpace]
            self.coeffsInKFormBasis = [j[1] for j in self.STFClassDataMinimal]
            self.kFormBasisGenerators = [
                [oneFormsLabelsLoc[k] for k in j[0]] for j in self.STFClassDataMinimal
            ]

    @property
    def coeffArray(self):
        if self._STFClassDataDictFull is None:
            if self._coeffArray is None:

                def entry_rule(indexTuple):
                    sortedTuple = tuple(sorted(indexTuple))
                    if sortedTuple in self.STFClassDataDict:
                        return self.STFClassDataDict[sortedTuple]
                    else:
                        return 0

                def generate_indices(shape):
                    """Recursively generates all index tuples for an arbitrary dimensional array."""
                    if len(shape) == 1:
                        return [(i,) for i in range(shape[0])]
                    else:
                        return [
                            (i,) + t
                            for i in range(shape[0])
                            for t in generate_indices(shape[1:])
                        ]

                shape = (len(self.varSpace),) * self.degree
                sparse_data = {
                    indices: entry_rule(indices) for indices in generate_indices(shape)
                }
                self._coeffArray = sp.ImmutableSparseNDimArray(sparse_data, shape)
        else:
            if self._coeffArray is None:
                self._coeffArray = sp.ImmutableSparseNDimArray(
                    self._STFClassDataDictFull, shape
                )
            # Create the ImmutableSparseNDimArray

        return self._coeffArray

    @property
    def STFClassDataDictFull(self):
        return self.expanded_coeff_dict

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the differential dorm for custom simplification.

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.

        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.

        Returns
        -------
        DFClass
            A new DFClass instance with updated simplification settings.
        """
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            warnings.warn(
                "simplify_format() recieved an unsupported first argument. Try None, 'holomorphic', 'real',  or 'symbolic_conjugate' instead."
            )
        return STFClass(
            self.varSpace,
            self.STFClassDataDict,
            self.degree,
            DGCVType=self.DGCVType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
        )

    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self._simplifyKW attribute.

        Returns
        -------
        DFClass
            A simplified DFClass object.
        """
        if self._simplifyKW["simplify_rule"] is None:
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(b, **kwargs) for a, b in self.STFClassDataDict.items()
            }
        elif self._simplifyKW["simplify_rule"] == "holomorphic":
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(
                    allToHol(b, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for a, b in self.STFClassDataDict.items()
            }
        elif self._simplifyKW["simplify_rule"] == "real":
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(
                    allToReal(b, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for a, b in self.STFClassDataDict.items()
            }
        elif self._simplifyKW["simplify_rule"] == "symbolic_conjugate":
            # Simplify each element in the coeffs list
            simplified_coeffs = {
                a: sp.simplify(
                    allToSym(b, skipVar=self._simplifyKW["simplify_ignore_list"]),
                    **kwargs,
                )
                for a, b in self.STFClassDataDict.items()
            }
        else:
            warnings.warn(
                "_eval_simplify recieved an unsupported STFClass._simplifyKW['simplify_rule']. It is recommend to only set the _simplifyKW['simplify_rule'] attribute to None, 'holomorphic', 'real',  or 'symbolic_conjugate'."
            )
            simplified_coeffs = {
                a: sp.simplify(b, **kwargs) for a, b in self.STFClassDataDict.items()
            }

        # Return a new instance of DFClass with simplified coeffs

        # Return a new instance of DFClass with simplified coeffs
        return STFClass(
            self.varSpace,
            simplified_coeffs,
            self.degree,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW,
        )

    def subs(self, subsData):
        newSTFData = {
            a: sp.sympify(b).subs(subsData) for a, b in self.STFClassDataDict.items()
        }
        return STFClass(
            self.varSpace,
            newSTFData,
            self.degree,
            DGCVType=self.DGCVType,
            _simplifyKW=self._simplifyKW,
        )

    def __neg__(self):
        return (-1)*self

    def __add__(self, other):
        """
        Adds two symmetric tensor fields (i.e., STFClass instances). If the variable spaces of the fields
        are not the same, the resulting field will have a variable space that is the union
        of both input spaces.

        Parameters
        ----------
        other : STFClass
            Another symmetric tensor field to add. The variable spaces of the two forms need not
            be identical. The result will use the union of the variable spaces of the two
            forms.

        Returns
        -------
        STFClass
            A new symmetric tensor field representing the sum of the two input fields.

        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `STFClass`.
        """
        if isinstance(other, STFClass):
            return addSTF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with DFClass")

    def __sub__(self, other):
        """
        Subtracts one symmetric tensor field from another (i.e., STFClass instances). The subtraction
        is performed by negating the second form and then adding it to the first form. If the
        variable spaces of the forms are not the same, the resulting form will have a variable
        space that is the sum of both input spaces.

        Parameters
        ----------
        other : STFClass
            Another symmetric tensor field to subtract. The variable spaces of the two fields need not
            be identical. The result will use the sum of the variable spaces of the two forms.

        Returns
        -------
        STFClass
            A new symmetric tensor field representing the difference between the two input fields.

        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `STFClass`.
        """
        if isinstance(other, STFClass):
            return addSTF(self, scaleTF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with STFClass")

    def __mul__(self, other):
        if isinstance(other,(int,float,sp.Expr)):
            return STFClass(self.varSpace, {k:other*v for k,v in self.coeff_dict.items()}, self.degree, DGCVType=self.DGCVType, _simplifyKW=self._simplifyKW)
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other,(int,float,sp.Expr)):
            return STFClass(self.varSpace,  {k:other*v for k,v in self.coeff_dict.items()}, self.degree, DGCVType=self.DGCVType, _simplifyKW=self._simplifyKW)
        return super().__mul__(other)


# DGCV polynomial class
class DGCVPolyClass(sp.Basic):
    """
    A polynomial class for handling both standard and complex polynomials, integrating SymPy's polynomial tools
    with DGCV's complex variable handling. The `DGCVPolyClass` wraps SymPy's core polynomial functionality, namely the Poly class objects, in a way that interacts well with the DGCV framework for managing complex variable systems.

    DGCVPolyClass augments the base SymPy `Poly` in a few ways:

    - **Support for multiple variable formats**: The class allows working with polynomials in unformatted,
      real, or complex (holomorphic/antiholomorphic with symbolic conjuagtes) forms through different lazy attributes.
      The unformatted form maintains whatever format the expression given to initialize the DGCVPolyClass was in.
    - **Holomorphic, antiholomorphic, pluriharmonic, and mixed term extraction**: This class has built-in methods to
      decompose polynomials based on these parts, leveraging DGCV's `allToSym` and `allToReal` functions.


    Parameters
    ----------
    polyExpr : sympy.Expr
        The polynomial expression to be handled.
    varSpace : tuple, optional
        The tuple of variables used in the polynomial. If not provided, it defaults to the free symbols in the given expression.
    degreeUpperBound : int, optional
      automatically inferred via sympy Poly class methods if not provided. `degreeUpperBound` is an attribute that a polynomial can be marked with for applications where it is preferable to evaluate only low degree terms (i.e., below the upper bound). It defaults to the polynomial's degree.

    Attributes
    ----------
    polyExpr : sympy.Expr
        The polynomial expression wrapped by this class.
    varSpace : tuple
        The set of variables involved in the polynomial.
    degreeUpperBound : int or None
        The degree upper bound of the polynomial, inferred or provided during initialization.
    poly_obj_unformatted : sympy.Poly
        The unformatted SymPy `Poly` object constructed from the provided expression and variable space.
    poly_obj_complex : sympy.Poly
        The `Poly` object with holomorphic and antiholomorphic variables, constructed via DGCV's complex variable transformations.
    poly_obj_real : sympy.Poly
        The `Poly` object with real variables, constructed using DGCV's real variable transformations.

    Methods
    -------
    get_monomials(min_degree=0, max_degree=None, format='unformatted'):
        Returns the monomials of the polynomial within the specified degree range and format.

    holomorphic_part():
        Lazily computes and returns the holomorphic part of the polynomial.

    antiholomorphic_part():
        Lazily computes and returns the antiholomorphic part of the polynomial.

    pluriharmonic_part():
        Lazily computes and returns the pluriharmonic part of the polynomial (sum of holomorphic and antiholomorphic parts).

    mixed_terms():
        Lazily computes and returns the mixed terms (terms involving both holomorphic and antiholomorphic variables).

    __add__(other):
        Adds two `DGCVPolyClass` instances, combining their variable spaces and degree bounds.

    __mul__(other):
        Multiplies two `DGCVPolyClass` instances, combining their variable spaces and degree bounds.

    __subs__(substitutions):
        Substitutes variables in the polynomial expression and returns a new `DGCVPolyClass` instance with the substitutions applied.

    Example
    -------
    >>> from sympy import symbols
    >>> from dgcv import DGCVPolyClass, complexVarProc
    >>> x, y, z, BARz = symbols('x y z BARz')
    >>> complexVarProc('z', 'x', 'y')  # Initialize complex variables in DGCV
    >>> poly = DGCVPolyClass(y**2 + z**2)
    >>> print(poly.holomorphic_part)  # Extract the holomorphic part
    -z**2
    >>> print(poly.get_monomials(format='complex'))  # Get monomials in complex format
    [-z**2, y**2]
    """

    def __new__(cls, polyExpr, varSpace=None, degreeUpperBound=None):
        # Create a new instance using sp.Basic's __new__
        return sp.Basic.__new__(cls, polyExpr)

    def __init__(self, polyExpr, varSpace=None, degreeUpperBound=None):
        """
        Initializes the DGCVPolyClass with an optional variable space.

        Parameters:
        polyExpr : sympy.Expr
            The polynomial expression.
        varSpace : tuple, optional
            The tuple of variables (default is the free symbols in polyExpr).
        degreeUpperBound : int, optional
            The upper bound on the degree of the polynomial.
        """
        polyExpr = sp.sympify(polyExpr)

        # If varSpace is not provided, infer it from the free symbols in polyExpr
        if varSpace is None:
            if isinstance(polyExpr, (int, float)):  # If polyExpr is a scalar
                self.varSpace = ()
            else:
                self.varSpace = tuple(polyExpr.free_symbols)
        else:
            self.varSpace = tuple(varSpace)

        self.polyExpr = polyExpr
        self.degree = sp.total_degree(polyExpr, *self.varSpace)
        self.degreeUpperBound = degreeUpperBound
        # Initialize private attributes for lazy evaluation
        self._holomorphic_part = None
        self._antiholomorphic_part = None
        self._pluriharmonic_part = None
        self._mixed_terms_part = None

        self._poly_obj_unformatted = None
        self._poly_obj_complex = None
        self._poly_obj_real = None

    @property
    def poly_obj_unformatted(self):
        """Lazily constructs and returns the unformatted Poly object."""
        if self._poly_obj_unformatted is None:
            self._poly_obj_unformatted = sp.Poly(self.polyExpr, *self.varSpace)
        return self._poly_obj_unformatted

    @property
    def poly_obj_complex(self):
        """Lazily constructs and returns the Poly object with complex variable handling."""
        if self._poly_obj_complex is None:
            # Apply allToSym to convert to holomorphic and antiholomorphic variables
            complex_expr = allToSym(self.polyExpr)
            # Get the free symbols of the transformed expression for generators
            complex_varSpace = tuple(
                set.union(*[allToSym(j).free_symbols for j in self.varSpace])
            )
            self._poly_obj_complex = sp.Poly(complex_expr, *complex_varSpace)
        return self._poly_obj_complex

    @property
    def poly_obj_real(self):
        """Lazily constructs and returns the Poly object with real variable handling."""
        if self._poly_obj_real is None:
            # Apply allToReal to convert to real variables
            real_expr = allToReal(self.polyExpr)
            # Get the free symbols of the transformed expression for generators
            real_varSpace = tuple(
                set.union(*[allToReal(j).free_symbols for j in self.varSpace])
            )
            self._poly_obj_real = sp.Poly(real_expr, *real_varSpace)
        return self._poly_obj_real

    def get_monomials(
        self, min_degree=0, max_degree=None, format="unformatted", return_coeffs=False
    ):
        """
        Computes and returns the monomials of the polynomial within the specified degree range,
        using the specified format for the polynomial generators.

        Parameters
        ----------
        min_degree : int, optional
            The minimum degree of monomials to return (default is 0).
        max_degree : int, optional
            The maximum degree of monomials to return (default is the polynomial's total degree).
        format : str, optional
            The format of the polynomial generators. Must be one of:
            - 'unformatted': Use the original variables provided or free symbols from polyExpr.
            - 'complex': Use holomorphic and antiholomorphic variables (z, BARz).
            - 'real': Use real variable representations (x, y).
            (default is 'unformatted').
        return_coeffs : bool, optional
            If true, only a list of coefficients is returned

        Returns
        -------
        list
            A list of monomials within the specified degree range, or just their coefficients if return_coeffs=True is set.
        """
        if format not in ["unformatted", "complex", "real"]:
            raise ValueError(
                "Invalid format. Choose 'unformatted', 'complex', or 'real'."
            )

        # Select the appropriate Poly object based on the format
        if format == "unformatted":
            poly_obj = self.poly_obj_unformatted
        elif format == "complex":
            poly_obj = self.poly_obj_complex
        else:
            poly_obj = self.poly_obj_real

        # Default max_degree to the total degree of the polynomial
        if max_degree is None:
            max_degree = self.degree

        # Extract monomials and coefficients
        monoms = poly_obj.monoms()
        coeffs = poly_obj.coeffs()

        # Filter monomials by degree
        if return_coeffs:
            filtered_monomials = [
                coeff
                for monom, coeff in zip(monoms, coeffs)
                if min_degree <= sum(monom) <= max_degree
            ]
        else:
            filtered_monomials = [
                sp.Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                for monom, coeff in zip(monoms, coeffs)
                if min_degree <= sum(monom) <= max_degree
            ]

        return filtered_monomials

    def get_coeffs(
        self, min_degree=0, max_degree=None, format="unformatted"
    ):
        return self.get_monomials(min_degree=min_degree, max_degree=max_degree, format=format, return_coeffs=True
        )

    @property
    def holomorphic_part(self):
        """Lazily computes and returns the holomorphic part of the polynomial."""
        if self._holomorphic_part is None:
            # Use the 'complex' formatted Poly object
            poly_obj = self.poly_obj_complex
            variable_registry = get_variable_registry()
            conversion_dictionaries = variable_registry["conversion_dictionaries"]

            holomorphic_terms = []

            for monom, coeff in zip(poly_obj.monoms(), poly_obj.coeffs()):
                # Rebuild the monomial from gens and exponents
                term = (
                    sp.Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                )
                free_symbols = term.free_symbols

                # Check if all symbols are holomorphic
                if all(
                    symbol in conversion_dictionaries["holToReal"]
                    for symbol in free_symbols
                ):
                    holomorphic_terms.append(term)

            self._holomorphic_part = sp.Add(*holomorphic_terms) if holomorphic_terms else 0
        return self._holomorphic_part

    @property
    def antiholomorphic_part(self):
        """Lazily computes and returns the antiholomorphic part of the polynomial."""
        if self._antiholomorphic_part is None:
            # Use the 'complex' formatted Poly object
            poly_obj = self.poly_obj_complex
            variable_registry = get_variable_registry()
            conversion_dictionaries = variable_registry["conversion_dictionaries"]

            antiholomorphic_terms = []

            for monom, coeff in zip(poly_obj.monoms(), poly_obj.coeffs()):
                # Rebuild the monomial from gens and exponents
                term = (
                    sp.Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                )
                free_symbols = term.free_symbols

                # Check if all symbols are antiholomorphic
                if all(
                    symbol in conversion_dictionaries["symToHol"]
                    for symbol in free_symbols
                ):
                    antiholomorphic_terms.append(term)

            self._antiholomorphic_part = (
                sp.Add(*antiholomorphic_terms) if antiholomorphic_terms else 0
            )
        return self._antiholomorphic_part

    @property
    def pluriharmonic_part(self):
        """Lazily computes and returns the pluriharmonic part of the polynomial."""
        if self._pluriharmonic_part is None:
            # Sum of holomorphic and antiholomorphic parts, without double-counting the constant term
            holomorphic = self.holomorphic_part
            antiholomorphic = self.antiholomorphic_part

            # Constant term is common between holomorphic and antiholomorphic parts
            constant_term = holomorphic.as_coefficients_dict().get(1, 0)

            # Subtract the constant term to avoid double-counting
            self._pluriharmonic_part = holomorphic + antiholomorphic - constant_term

        return self._pluriharmonic_part

    @property
    def mixed_terms(self):
        """Lazily computes and returns the mixed terms of the polynomial."""
        if self._mixed_terms_part is None:
            # Use the 'complex' formatted Poly object
            poly_obj = self.poly_obj_complex
            variable_registry = get_variable_registry()
            conversion_dictionaries = variable_registry["conversion_dictionaries"]

            mixed_terms = []

            for monom, coeff in zip(poly_obj.monoms(), poly_obj.coeffs()):
                # Rebuild the monomial from gens and exponents
                term = (
                    sp.Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                )
                free_symbols = term.free_symbols

                # Check if the term contains both holomorphic and antiholomorphic variables
                is_holomorphic = all(
                    symbol in conversion_dictionaries["holToReal"]
                    for symbol in free_symbols
                )
                is_antiholomorphic = all(
                    symbol in conversion_dictionaries["symToHol"]
                    for symbol in free_symbols
                )

                if not (is_holomorphic or is_antiholomorphic):
                    mixed_terms.append(term)

            self._mixed_terms_part = sp.Add(*mixed_terms) if mixed_terms else 0
        return self._mixed_terms_part

    # Custom simplify and latex methods
    def simplify_poly(self):
        return DGCVPolyClass(
            sp.simplify(self.polyExpr),
            self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def latex_representation(self):
        return sp.latex(self.polyExpr)

    def _repr_latex_(self):
        return (self.polyExpr)._repr_latex_()

    def _eval_simplify(self, **kwargs):
        return self.simplify_poly()

    def _latex(self, printer=None):
        return self.latex_representation()

    # Additional methods
    def evaluate(self, **values):
        return self.polyExpr.subs(values)

    def get_degree(self):
        return self.degree

    def is_homogeneous(self):
        degrees = [sum(j[1]) for j in self.multiIndexEncoding if j[0] != 0]
        return all(degree == degrees[0] for degree in degrees)

    def pretty_print(self):
        return sp.pretty(self.polyExpr)

    def expand(self, deep=True, modulus=None, **hints):
        """
        Expands the internal polynomial expression and returns a new DGCVPolyClass instance.
        Supports 'deep' and 'modulus' keywords, passed directly to SymPy's expand function.

        Parameters
        ----------
        deep : bool, optional
            Whether to apply expansion recursively to sub-expressions. Default is True.
        modulus : int, optional
            Apply expansion with respect to this modulus (modular arithmetic). Default is None.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the expanded polynomial expression.
        """
        # Pass 'deep', 'modulus', and any additional hints to SymPy's expand method
        expanded_expr = self.polyExpr.expand(deep=deep, modulus=modulus, **hints)
        return DGCVPolyClass(
            expanded_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def factor(self, **hints):
        """
        Factors the internal polynomial expression and returns a new DGCVPolyClass instance.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the factored polynomial expression.
        """
        factored_expr = self.polyExpr.factor(**hints)
        return DGCVPolyClass(
            factored_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def expand_trig(self, **hints):
        """
        Expands trigonometric functions in the polynomial expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with expanded trigonometric expressions.
        """
        expanded_trig_expr = self.polyExpr.expand(trig=True, **hints)
        return DGCVPolyClass(
            expanded_trig_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def cancel(self, **hints):
        """
        Cancels common factors between the numerator and the denominator of the polynomial expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the canceled rational expression.
        """
        canceled_expr = self.polyExpr.cancel(**hints)
        return DGCVPolyClass(
            canceled_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def diff(self, *symbols, **hints):
        """
        Differentiates the polynomial expression with respect to the given symbols.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the differentiated expression.
        """
        differentiated_expr = self.polyExpr.diff(*symbols, **hints)
        return DGCVPolyClass(
            differentiated_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def integrate(self, *symbols, **hints):
        """
        Integrates the polynomial expression with respect to the given symbols.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the integrated expression.
        """
        integrated_expr = self.polyExpr.integrate(*symbols, **hints)
        return DGCVPolyClass(
            integrated_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def subs(self, substitutions, **hints):
        """
        Substitutes the given variables in the polynomial expression and returns a new DGCVPolyClass instance.

        Parameters
        ----------
        substitutions : dict or list
            A dictionary or list of substitutions, where keys are variables to replace and values are the new values.
        hints : dict, optional
            Additional hints to pass to the substitution process.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the substituted polynomial expression.
        """
        # Perform the substitution on the internal polyExpr
        substituted_expr = self.polyExpr.subs(substitutions, **hints)

        # Return a new DGCVPolyClass with the substituted expression, preserving varSpace and degreeUpperBound
        return DGCVPolyClass(
            substituted_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    def _apply_sympy_function(self, sympy_func, *args, **kwargs):
        """
        Applies the given SymPy function to the internal polyExpr and returns a new DGCVPolyClass instance.

        Parameters
        ----------
        sympy_func : callable
            The SymPy function to apply (e.g., factor, collect, cancel).
        args : tuple
            Positional arguments to pass to the SymPy function.
        kwargs : dict
            Keyword arguments to pass to the SymPy function.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the transformed polynomial expression.
        """
        transformed_expr = sympy_func(
            self.polyExpr, *args, **kwargs
        )  # Correct function call
        return DGCVPolyClass(
            transformed_expr,
            varSpace=self.varSpace,
            degreeUpperBound=self.degreeUpperBound,
        )

    # Implement _eval_* methods using _apply_sympy_function
    def _eval_factor(self, **hints):
        return self._apply_sympy_function(sp.factor, **hints)

    def _eval_cancel(self, **hints):
        return self._apply_sympy_function(self.polyExpr.cancel, **hints)

    def _eval_diff(self, *symbols, **hints):
        return self._apply_sympy_function(self.polyExpr.diff, *symbols, **hints)

    def _eval_integrate(self, *symbols, **hints):
        return self._apply_sympy_function(self.polyExpr.integrate, *symbols, **hints)

    def _eval_expand(self, **hints):
        return self._apply_sympy_function(self.polyExpr.expand, *sp.symbols, **hints)

    def __add__(self, other):
        """
        Adds the current polynomial expression to another polynomial or scalar.

        Parameters
        ----------
        other : DGCVPolyClass or sympy.Expr
            The object to add.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the added expression.
        """
        if isinstance(other, DGCVPolyClass):
            # Combine varSpaces, preserving order and removing duplicates
            new_varSpace = tuple(dict.fromkeys(self.varSpace + other.varSpace))
            return DGCVPolyClass(
                self.polyExpr + other.polyExpr,
                varSpace=new_varSpace,
                degreeUpperBound=self.degreeUpperBound,
            )

        elif isinstance(other, (int, float, sp.Expr)):
            # Add directly
            return DGCVPolyClass(
                self.polyExpr + other,
                varSpace=self.varSpace,
                degreeUpperBound=self.degreeUpperBound,
            )

        return NotImplemented

    def __sub__(self, other):
        """
        Subtracts another polynomial or scalar from the current polynomial expression.

        Parameters
        ----------
        other : DGCVPolyClass or sympy.Expr
            The object to subtract.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the subtracted expression.
        """
        if isinstance(other, DGCVPolyClass):
            # Combine varSpaces, preserving order and removing duplicates
            new_varSpace = tuple(dict.fromkeys(self.varSpace + other.varSpace))
            return DGCVPolyClass(
                self.polyExpr - other.polyExpr,
                varSpace=new_varSpace,
                degreeUpperBound=self.degreeUpperBound,
            )

        elif isinstance(other, (int, float, sp.Expr)):
            # Subtract directly
            return DGCVPolyClass(
                self.polyExpr - other,
                varSpace=self.varSpace,
                degreeUpperBound=self.degreeUpperBound,
            )

        return NotImplemented

    def __mul__(self, other):
        """
        Multiplies the current polynomial expression by another polynomial or scalar.

        Parameters
        ----------
        other : DGCVPolyClass or sympy.Expr
            The object to multiply with. This can be a DGCVPolyClass instance, a scalar, or a SymPy expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the multiplied polynomial expression.

        Raises
        ------
        ValueError
            If the multiplication involves a non-polynomial or unsupported expression.
        """
        # Case 1: Multiplying by another DGCVPolyClass instance
        if isinstance(other, DGCVPolyClass):
            # Combine varSpaces, preserving order and removing duplicates
            new_varSpace = tuple(dict.fromkeys(self.varSpace + other.varSpace))

            # Determine the new degreeUpperBound as the minimum of the non-None values
            if self.degreeUpperBound is None and other.degreeUpperBound is None:
                new_degreeUpperBound = None
            else:
                degree_values = [
                    deg
                    for deg in [self.degreeUpperBound, other.degreeUpperBound]
                    if deg is not None
                ]
                new_degreeUpperBound = min(degree_values) if degree_values else None

            return DGCVPolyClass(
                self.polyExpr * other.polyExpr,
                varSpace=new_varSpace,
                degreeUpperBound=new_degreeUpperBound,
            )

        # Case 2: Multiplying by a scalar (constant or polynomial)
        elif isinstance(other, (int, float, sp.Number, sp.Poly, sp.Symbol)):
            # Multiply directly
            return DGCVPolyClass(
                self.polyExpr * other,
                varSpace=self.varSpace,
                degreeUpperBound=self.degreeUpperBound,
            )

        # Case 3: Handle non-polynomial expressions
        elif isinstance(other, sp.Expr):
            # Check if the scalar is a polynomial
            if sp.Poly(other, self.varSpace).is_zero:
                raise ValueError(
                    "Multiplication with non-polynomial expressions is not supported."
                )

            # Proceed with multiplication (but warn it's non-polynomial)
            return DGCVPolyClass(
                self.polyExpr * other,
                varSpace=self.varSpace,
                degreeUpperBound=self.degreeUpperBound,
            )

        # Case 4: Unsupported types
        return NotImplemented

    def __rmul__(self, other):
        """
        Right-hand multiplication by a scalar (or polynomial).

        Parameters
        ----------
        other : int, float, sympy.Expr
            A scalar or polynomial expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the multiplied expression.
        """
        return self.__mul__(other)

    def leading_term(self):
        terms = sorted(self.polyMonomials, key=lambda x: sum(x[1]), reverse=True)
        return terms[0] if terms else None


############## variable creation
def createVariables(
    variable_label,
    real_label=None,
    imaginary_label=None,
    number_of_variables=None,
    initialIndex=1,
    withVF=None,
    complex=None,
    multiindex_shape=None,
    assumeReal=None,
    remove_guardrails=None,
    default_var_format=None,
):
    """
    This function serves as the default interface for creating variables within the DGCV package. It supports creating both standard variable systems and complex variable systems, with options for initializing coordinate vector fields and differential forms. Variables created through `createVariables` are automatically tracked within DGCV’s Variable Management Framework (VMF) and are assigned labels validated through a safeguards routine that prevents overwriting important labels.

    Parameters
    ----------
    variable_label : str
        The label for the primary variable or system of variables to be created. If creating a complex variable system,
        this will correspond to the holomorphic variable(s), whilst antiholomorphic variable(s) recieve this label pre-pended with "BAR".

    real_label : str, optional
        The label for the real part of the complex variable system. Required only when creating complex variable systems.

    imaginary_label : str, optional
        The label for the imaginary part of the complex variable system. Required only when creating complex variable systems.

    number_of_variables : int, optional
        The number of variables to be created, used to initialize a tuple of variables rather than a single variable (e.g., x=(x1, x2, x3) rather than just x).

    initialIndex : int, optional, default=1
        The starting index for tuple variables, allowing for flexible indexing when initializing variable systems
        as tuples (e.g., x=(x0, x1, x2) with `initialIndex=0`).

    withVF : bool, optional
        If set to True, creates associated coordinate vector fields and differential forms for the variable(s) in the system.

    complex : bool, optional
        Specifies whether to create a complex variable system. If not provided, the function will infer whether to create
        a complex system based on whether `real_label` or `imaginary_label` is provided. If provided a complex variable system will be created regardless of `real_label` and  `imaginary_label` settings, and string labels are automaticaly created for `real_label` and  `imaginary_label` if they are not provided.

    assumeReal : bool, optional
        If set to True, specifies that the variables being created are real-valued. This is only relevant for standard
        variable systems and is ignored for complex systems.

    remove_guardrails : bool, optional
        If set to True, bypasses DGCV's safeguard system for variable labeling, allowing one to overwrite certain
        reserved labels. Use with caution, as it may overwrite important variables in the global namespace.

    default_var_format : {'complex', 'real'}, optional
        Relevant only for complex variable systems. Specifies whether the system's vector fields and differential forms
        default to real coordinate expressions (`real`) or holomorphic coordinate expressions (`complex`). If not provided,
        the default is holomorphic coordinates.

    Returns
    -------
    None
        This function creates the specified variable system and registers it within DGCV’s Variable Management Framework.

    Functionality
    -------------
    - Creates standard or complex variable systems.
    - Automatically registers all created variables, vector fields, and differential forms in DGCV’s VMF.
    - Safeguards are applied to ensure that no critical Python or DGCV internal functions are overwritten.

    Notes
    -----
    - If `complex=True`, the function will automatically generate holomorphic, antiholomorphic, real, and imaginary parts
    for the variable system, along with corresponding coordinate vector fields and differential forms.
    - If `withVF=True`, the function will create both vector fields and differential forms for standard variable systems.

    Examples
    --------
    # Creating a single standard variable 'alpha'
    >>> createVariables('alpha')

    # Creating a tuple of 3 standard variables with associated vector fields and differential forms
    >>> createVariables('alpha', 3, withVF=True)

    # Creating a single complex variable system with real part 'beta' and imaginary part 'gamma'
    >>> createVariables('alpha', 'beta', 'gamma')

    # Creating a tuple of 3 complex variables with real parts 'beta1', 'beta2', 'beta3' and imaginary parts 'gamma1', 'gamma2', 'gamma3'
    >>> createVariables('alpha', 'beta', 'gamma', 3)

    # Creating a tuple of 3 standard variables with custom initial index
    >>> createVariables('alpha', 3, initialIndex=0)

    # Creating a complex variable system without specifying real and imaginary part labels. Note, this automatically creates intentionally obscure real and imaginary part labels.
    >>> createVariables('alpha', complex=True)

    Warnings
    --------
    If `complex=True` is provided along with incompatible keywords, such as `withVF=False`, or `complex=False` is provided while values are also given for `real_label` and `imaginary_label`, the function will resolve conflicts internally and issue a warning about the resolution.

    Use `DGCV_snapshot()` for a clear summary of the variables created and tracked within the DGCV VMF.

    """
    if multiindex_shape is not None:
        if (
            any(j is not None for j in [real_label, imaginary_label, withVF, complex])
            or default_var_format == "complex"
        ):
            warnings.warn(
                "A value for `multiindex_shape` was provided, so a standard variable system without vector fields or differential forms was created alligned with the multiindex_shape value. Multiindex variable labeling is not yet supported by DGCV's automated variable creation with vector fields or for complex variable systems. It will be in future DGCV updates. "
            )
            real_label = None
            imaginary_label = None
            withVF = None
            complex = None
            default_var_format = None
    if not isinstance(variable_label, str):
        raise TypeError(
            "`createVariables` requires its first argument to be a string, which will be used in lables for the created variables."
        )
    if (
        isinstance(real_label, int)
        and imaginary_label is None
        and number_of_variables is None
    ):
        number_of_variables = real_label
        real_label = None
    if real_label is not None and not isinstance(real_label, str):
        raise TypeError(
            "A non-string value cannot be assigned to the `real_label` keyword of `createVariables`"
        )
    if real_label is not None and not isinstance(imaginary_label, str):
        raise TypeError(
            "A non-string value cannot be assigned to the `imaginary_label` keyword of `createVariables`"
        )
    if complex and not withVF:
        warnings.warn(
            "`createVariables` was called with `complex=True` and `withVF=False`. The latter keyword was disregarded because DGCV automatically initializes associated differential objects whenever complex variable systems are created."
        )
    if complex and assumeReal:
        warnings.warn(
            "`createVariables` was called with `complex=True` and `assumeReal=True`. The latter keyword was disregarded because DGCV has fixed variable assumptions for elements in its complex variable systems."
        )
    if complex == False and any([real_label is not None, imaginary_label is not None]):  # noqa: E712
        warnings.warn(
            "`createVariables` recieved `complex=False` and recieved values for the `real_label` or `imaginary_label` keywords. Honoring `complex=False`, only a standard variable system was created, and latter labels were disregarded. Set `complex=True` if a complex variable system is needed instead."
        )
        real_label = None
        imaginary_label = None
    elif complex and all([real_label is None, imaginary_label is None]):
        key_string = retrieve_public_key()
        real_label = variable_label + "REAL" + key_string
        imaginary_label = variable_label + "IM" + key_string
        warnings.warn(
            "`createVariables` recieved `complex=True` did not recieve value assignements for `imaginary_label` or `real_label`, so intentionally obscure labels were created for both the real and imaginary variables in the created complex variable system. To have nicer labling while using `complex=True`, provide a preferred string label for the `real_label` and `imaginary_label` keywords."
        )
    elif any([real_label is not None, imaginary_label is not None]):
        if not complex and complex is not None:
            warnings.warn(
                "The keyword 'complex' was set to a non-Bool value. Since a string value was also assigned to either `real_label` or `imaginary_label`, `createVariables` proceeded under the assumption that it should create a complex variable system. If a standard variable system was prefered then set `complex=False` instead."
            )
        complex = True
        if real_label is None:
            real_label = variable_label + "REAL" + retrieve_public_key()
            warnings.warn(
                "`createVariables` recieved a value assigned to `imaginary_label` but not `real_label`, so an intentionally obscure label was created for the real variables in the created complex variable system. To have nice nicer labling, provide a preferred string label to `real_label` instead."
            )
        if imaginary_label is None:
            imaginary_label = variable_label + "IM" + retrieve_public_key()
            warnings.warn(
                "`createVariables` recieved a value assigned to `real_label` but not `imaginary_label`, so an intentionally obscure label was created for the imaginary variables in the created complex variable system. To have nice nicer labling, provide a preferred string label to `imaginary_label` instead."
            )

    def reformat_string(input_string: str):
        # Replace commas with spaces, then split on spaces
        substrings = input_string.replace(",", " ").split()
        # Return the list of non-empty substrings
        return [s for s in substrings if len(s) > 0]

    if isinstance(variable_label, str):
        variable_label = reformat_string(variable_label)
    if isinstance(real_label, str):
        real_label = reformat_string(real_label)
    if isinstance(imaginary_label, str):
        imaginary_label = reformat_string(imaginary_label)

    if complex:
        complexVarProc(
            variable_label,
            real_label,
            imaginary_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            default_var_format=default_var_format,
            remove_guardrails=remove_guardrails,
        )
        return
    elif withVF:
        varWithVF(
            variable_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            _doNotUpdateVar=False,
            assumeReal=assumeReal,
            _calledFromCVP=None,
            remove_guardrails=remove_guardrails,
        )
        return
    else:
        variableProcedure(
            variable_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            assumeReal=assumeReal,
            multiindex_shape=multiindex_shape,
            _tempVar=False,
            _doNotUpdateVar=None,
            _calledFromCVP=None,
            remove_guardrails=remove_guardrails,
        )


############## standard variables
def variableProcedure(
    variables_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    assumeReal=None,
    _tempVar=None,
    _doNotUpdateVar=None,
    _calledFromCVP=None,
    remove_guardrails=None,
    _obscure=None,
):
    """
    Initializes one or more standard variables (single or tuples) and integrates them into DGCV's Variable Management Framework.

    Parameters:
    ----------
    variables_label : str or list of str
        The label for the variable(s). If a string, it defines one variable set. If a list of strings, it initializes multiple variable sets.

    number_of_variables : positive int, optional
        If provided, specifies the number of variables to be initialized and labels them in an enumerated tuple creating labels based on variables_label. If not provided, a single variable is initialized.

    initialIndex : int, optional
        The starting index for enumerating variable names in a tuple. Defaults to 1.

    assumeReal : bool, optional
        If provided, specifies whether to treat the variable(s) as real. If True, the variable(s) are assumed to be real.

    _tempVar : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, marks the variable system as temporary.

    _obscure : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, marks the variable system as 'obscure'.

    _doNotUpdateVar : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, prevents the variable(s) from being cleared and reinitialized.

    _calledFromCVP : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, interfaces with the Variable Management Framework supposing this is called from complexVarProc.

    remove_guardrails : bool, optional
        If set to True, bypasses DGCV's safeguard system for variable labeling.

    Updates the internal variable_registry dict accordingly.
    """

    # Cache globals and passkey to avoid repeated lookups.
    globals_cache = _cached_caller_globals
    passkey = retrieve_passkey()

    variable_registry = get_variable_registry()
    # Check if the variable is part of the protected variables when not called from complexVarProc
    if not _calledFromCVP == passkey:
        for j in (tuple(variables_label) if isinstance(variables_label, (list, tuple)) else (variables_label,)):
            if j in variable_registry.get("protected_variables", set()):
                raise Exception(
                    f"{variables_label} is already assigned to the real or imaginary part of a complex variable system, so DGCV variable creation functions will not reassign it as a standard variable. Instead, use the clearVar function to remove the conflicting CV system first before implementing such reassignments."
                )

    # Process each variable label.
    for j in (tuple(variables_label) if isinstance(variables_label, (list, tuple)) else (variables_label,)):
        if not _calledFromCVP == passkey and not remove_guardrails:
            labelLoc = validate_label(j)
        else:
            labelLoc = j

        # Clear variable if necessary and if _doNotUpdateVar is not set.
        if _doNotUpdateVar != passkey:
            clearVar(j, report=False)

        # Mark as temporary/obscure if appropriate.
        if _tempVar == passkey:
            variable_registry["temporary_variables"].add(labelLoc)
            _tempVar = True
        else:
            _tempVar = None
        if _obscure == passkey:
            variable_registry["obscure_variables"].add(labelLoc)
            _obscure = True

        # Handle multi-index variable case.
        if isinstance(multiindex_shape, (list, tuple)) and all(isinstance(n, int) and n > 0 for n in multiindex_shape):
            # Generate multi-index variable names.
            indices = list(carProd(*[range(initialIndex, initialIndex + n) for n in multiindex_shape]))
            var_names = [f"{labelLoc}_{'_'.join(map(str, idx))}" for idx in indices]
            vars = [sp.symbols(f"{labelLoc}_{'_'.join(map(str, idx))}", real=assumeReal) for idx in indices]

            # Batch update globals.
            new_globals = dict(zip(var_names, vars))
            new_globals[labelLoc] = vars
            globals_cache.update(new_globals)

            # Create parent dictionary for the multi-indexed variables.
            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "multi_index",
                    "family_shape": multiindex_shape,
                    "family_names": tuple(var_names),
                    "family_values": globals_cache[labelLoc],
                    "differential_system": None,
                    "tempVar": _tempVar,
                    "obsVar": _obscure,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {"VFClass": None, "DFClass": None, "assumeReal": assumeReal}
                        for var_name in var_names
                    },
                }

        # Handle lone (single) variable.
        elif number_of_variables is None:
            symbol = sp.symbols(labelLoc, real=assumeReal)
            globals_cache[labelLoc] = symbol

            # Create parent dictionary for lone variable.
            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "single",
                    "family_names": (labelLoc,),
                    "family_values": (globals_cache[labelLoc],),
                    "differential_system": None,
                    "tempVar": _tempVar,
                    "obsVar": _obscure,
                    "initial_index": None,
                    "variable_relatives": {
                        labelLoc: {"VFClass": None, "DFClass": None, "assumeReal": assumeReal}
                    },
                }

        # Handle tuple of variables.
        elif isinstance(number_of_variables, int) and number_of_variables >= 0:
            lengthLoc = number_of_variables
            var_names = [f"{labelLoc}{i}" for i in range(initialIndex, lengthLoc + initialIndex)]
            vars = [sp.symbols(f"{labelLoc}{i}", real=assumeReal) for i in range(initialIndex, lengthLoc + initialIndex)]
            new_globals = dict(zip(var_names, vars))
            new_globals[labelLoc] = tuple(vars)
            globals_cache.update(new_globals)

            # Create individual vector fields.
            vf_instances = [VFClass(tuple(vars), [1 if k == j else 0 for k in range(len(vars))])
                            for j in range(len(vars))]
            new_globals = dict(zip([f"D_{var_name}" for var_name in var_names], vf_instances))
            globals_cache.update(new_globals)

            # Create individual differential 1-forms.
            df_instances = [DFClass(tuple(vars), {(j,): 1}, 1) for j in range(len(vars))]
            new_globals = dict(zip([f"d_{var_name}" for var_name in var_names], df_instances))
            globals_cache.update(new_globals)

            # Update parent dictionary for the tuple of variables.
            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "tuple",
                    "family_values": tuple(vars),
                    "family_names": tuple(var_names),
                    "differential_system": True,
                    "tempVar": _tempVar,
                    "initial_index": initialIndex,
                    "obsVar": _obscure,
                    "variable_relatives": {
                        var_name: {"VFClass": vf_instances[i], "DFClass": df_instances[i], "assumeReal": assumeReal}
                        for i, var_name in enumerate(var_names)
                    },
                }
        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided."
            )

def varProcMultiIndex(arg1, arg2, arg3):
    """
    Initializes a variable with label arg1 equal to a 2d array of Symbol objects labeled arg1=(arg1_1_1,arg1_1_2,...,arg1_arg2_arg3).

    Args:
        arg1: string
        arg2: positive integer
        arg3: positive integer

    Returns:
        Nothing

    Raises:
        NA
    """
    warnings.warn(
        "`varProcMultiIndex` is depricated and will be removed from DGCV in future updates. Use `createVariables` instead."
    )
    _cached_caller_globals.update(
        zip(
            [arg1],
            [
                sp.Array(
                    [
                        [
                            sp.symbols(arg1 + "_{}_{}".format(k, j))
                            for j in range(1, arg3 + 1)
                        ]
                        for k in range(1, arg2 + 1)
                    ]
                )
            ],
        )
    )
    _cached_caller_globals.update(
        zip(
            [
                arg1 + "_{}_{}".format(j, k)
                for j in range(1, arg3 + 1)
                for k in range(1, arg2 + 1)
            ],
            [
                sp.symbols(arg1 + "_{}_{}".format(j, k))
                for j in range(1, arg3 + 1)
                for k in range(1, arg2 + 1)
            ],
        )
    )

def varWithVF(
    variables_label,
    number_of_variables=None,
    initialIndex=1,
    _doNotUpdateVar=False,
    assumeReal=None,
    _calledFromCVP=None,
    remove_guardrails=None,
):
    """
    Initializes one or more standard variables with accompanying vector fields and differential 1-forms.
    Updates the internal variable_registry dict accordingly.

    Parameters:
    ----------
    variables_label : str or list
        The label for the variable(s). If a string, it defines a single variable. If a list, it initializes multiple variables.

    number_of_variables : positive int, optional
        If provided, specifies the number of variables in a tuple. The first argument in `args` defines the length of the tuple.

    initialIndex : int, optional
        The starting index for enumerating variable names in a tuple. Defaults to 1.

    _doNotUpdateVar : bool, optional
        If True, prevents the variable(s) from being cleared and reinitialized. Used when avoiding overwriting existing variables.

    assumeReal : bool, optional
        If provided, specifies whether to treat the variable(s) as real. If True, the variable(s) are assumed to be real.

    _calledFromCVP : None
        Reserved for internal DGCV use when called from complexVarProc. Does not affect the behavior in this function.

    Updates the internal variable_registry dict:
    --------------------------
    - For a single variable:
        - Adds a new entry in 'standard_variable_systems' with:
            - 'family_type': 'single'
            - 'family_names': (str,)
            - 'family_values': (variables,)
            - 'differential_system': bool (default None)
            - 'tempVar': bool (default None)
            - 'initial_index': None
            - 'variable_relatives': A dictionary containing the variable's label and its associated properties (VFClass, DFClass, assumeReal).

    - For a tuple of variables:
        - Adds a new entry in 'standard_variable_systems' with:
            - 'family_type': 'tuple'
            - 'family_names': tuple of strings
            - 'family_values': tuple of variables
            - 'differential_system': bool (default None)
            - 'tempVar': bool (default None)
            - 'initial_index': The starting index of the tuple
            - 'variable_relatives': A dictionary mapping each variable in the tuple to its associated properties (VFClass, DFClass, assumeReal).
        - Also adds the tuple of variables (e.g., y = (y1, y2, y3)) to _cached_caller_globals.

    Raises:
    -------
    Exception:
        If the variable label is part of the 'protected_variables' set in the internal variable_registry dict and this function was not called from complexVarProc.
    """
    variable_registry = get_variable_registry()

    # Convert arg1 to a list if it's a single string
    if isinstance(variables_label, str):
        variables_label = [variables_label]

    for labelLoc in variables_label:
        if not _calledFromCVP == retrieve_passkey() and not remove_guardrails:
            labelLoc = validate_label(labelLoc)

        if not _doNotUpdateVar == retrieve_passkey():
            clearVar(labelLoc, report=False)

        # Handle lone variable case
        if number_of_variables is None:
            symbol = sp.symbols(labelLoc, real=assumeReal)
            _cached_caller_globals[labelLoc] = symbol

            # Create vector field and differential form for the lone variable
            vf_instance = VFClass((symbol,), [1])
            df_instance = DFClass((symbol,), {(0,): 1}, 1)
            _cached_caller_globals[f"D_{labelLoc}"] = vf_instance
            _cached_caller_globals[f"d_{labelLoc}"] = df_instance

            # Update standard_variable_systems for the lone variable (only if not called from CVP)
            if not _calledFromCVP == retrieve_passkey():
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "single",
                    "family_values": (symbol,),
                    "family_names": (labelLoc,),
                    "differential_system": True,
                    "tempVar": None,
                    "initial_index": None,
                    "variable_relatives": {
                        labelLoc: {
                            "VFClass": vf_instance,
                            "DFClass": df_instance,
                            "assumeReal": assumeReal,
                        }
                    },
                }

        # Handle the tuple case
        elif isinstance(number_of_variables, int) and number_of_variables >= 0:
            lengthLoc = number_of_variables
            var_names = [
                f"{labelLoc}{i}" for i in range(initialIndex, lengthLoc + initialIndex)
            ]
            vars = [
                sp.symbols(f"{labelLoc}{i}", real=assumeReal)
                for i in range(initialIndex, lengthLoc + initialIndex)
            ]

            # Add each variable to _cached_caller_globals
            _cached_caller_globals.update(zip(var_names, vars))

            # Create the tuple of variables and add it to _cached_caller_globals
            _cached_caller_globals[labelLoc] = tuple(vars)

            # Create individual vector fields
            vf_instances = [
                VFClass(tuple(vars), [1 if k == j else 0 for k in range(len(vars))])
                for j in range(len(vars))
            ]
            _cached_caller_globals.update(
                zip([f"D_{var_name}" for var_name in var_names], vf_instances)
            )

            # Create individual differential 1-forms
            df_instances = [
                DFClass(tuple(vars), {(j,): 1}, 1) for j in range(len(vars))
            ]
            _cached_caller_globals.update(
                zip([f"d_{var_name}" for var_name in var_names], df_instances)
            )

            # Update standard_variable_systems for the parent variable (only if not called from CVP)
            if not _calledFromCVP == retrieve_passkey():
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "tuple",
                    "family_values": tuple(vars),
                    "family_names": tuple(var_names),
                    "differential_system": True,
                    "tempVar": None,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": vf_instances[i],
                            "DFClass": df_instances[i],
                            "assumeReal": assumeReal,
                        }
                        for i, var_name in enumerate(var_names)
                    },
                }
        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided."
            )


############## complex variables

def complexVarProc(
    holom_label,
    real_label,
    im_label,
    number_of_variables=None,
    initialIndex=1,
    default_var_format="complex",
    remove_guardrails=None,
):
    """
    Initializes a complex variable system, linking a holomorphic variable with its real and imaginary parts and
    a symbolic representative of its complex conjugate.

    (Docstring truncated for brevity.)
    """
    # Validate default_var_format.
    if default_var_format not in ("real", "complex"):
        if default_var_format is not None:
            warnings.warn(
                "`default_var_format` was set to an unsupported value, so it was reset to the default 'complex'."
            )
        default_var_format = "complex"

    # Cache the registry sub-dictionaries for faster access.
    variable_registry = get_variable_registry()
    conv = variable_registry["conversion_dictionaries"]
    find_parents = conv["find_parents"]
    protected_vars = variable_registry["protected_variables"]

    # Set up temporary accumulators for batched conversion updates.
    conj_updates = {}
    holToReal_updates = {}
    realToSym_updates = {}
    symToHol_updates = {}
    symToReal_updates = {}
    realToHol_updates = {}
    real_part_updates = {}
    im_part_updates = {}
    complex_system_updates = {}

    # For tuple systems, store data for deferred differential object creation.
    # Each entry: (labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc)
    tuple_system_data = []

    def validate_variable_labels(*labels):
        """
        Checks if provided labels start with 'BAR', reformats them,
        prevents duplicates, and checks for protected globals.
        """
        reformatted_labels = []
        seen_labels = set()
        protectedGlobals = protected_caller_globals()
        for label in labels:
            if label in protectedGlobals:
                raise ValueError(
                    f"DGCV recognizes label '{label}' as a protected global name and recommends not using it as a variable name. "
                    "Set remove_guardrails=True in the variable creation functions to force it."
                )
            if label.startswith("BAR"):
                reformatted_label = "anti_" + label[3:]
                warnings.warn(
                    f"Label '{label}' starts with 'BAR', which has special meaning in DGCV. It has been automatically reformatted to '{reformatted_label}'."
                )
            else:
                reformatted_label = label
            if reformatted_label in seen_labels:
                raise ValueError(
                    f"Duplicate label found: '{reformatted_label}'. Each label must be unique."
                )
            seen_labels.add(reformatted_label)
            reformatted_labels.append(reformatted_label)
        return tuple(reformatted_labels)

    # Convert single string inputs into lists.
    if isinstance(holom_label, str):
        holom_label = [holom_label]
        real_label = [real_label]
        im_label = [im_label]

    # Main loop over each set of labels.
    for j in range(len(holom_label)):
        if remove_guardrails:
            labelLoc1 = holom_label[j]
            labelLoc2 = real_label[j]
            labelLoc3 = im_label[j]
        else:
            labelLoc1, labelLoc2, labelLoc3 = validate_variable_labels(
                holom_label[j], real_label[j], im_label[j]
            )
        labelLocBAR = f"BAR{labelLoc1}"

        # Clear any existing variables.
        clearVar(labelLoc1, report=False)
        clearVar(labelLoc2, report=False)
        clearVar(labelLoc3, report=False)
        clearVar(labelLocBAR, report=False)

        # Add real and imaginary parts to the protected set.
        protected_vars.update({labelLoc2, labelLoc3})

        # -------------------------------------------------
        # Single Variable System
        # -------------------------------------------------
        if number_of_variables is None:
            variableProcedure(
                labelLoc1,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLocBAR,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc2,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc3,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )

            # Retrieve created variables from the caller's globals.
            var_hol = _cached_caller_globals[labelLoc1]
            var_bar = _cached_caller_globals[labelLocBAR]
            var_real = _cached_caller_globals[labelLoc2]
            var_im = _cached_caller_globals[labelLoc3]

            # Accumulate conversion updates.
            conj_updates[var_hol] = var_bar
            conj_updates[var_bar] = var_hol
            holToReal_updates[var_hol] = var_real + I * var_im
            realToSym_updates[var_real] = sp.Rational(1, 2) * (var_hol + var_bar)
            realToSym_updates[var_im] = -I * sp.Rational(1, 2) * (var_hol - var_bar)
            symToHol_updates[var_bar] = sp.conjugate(var_hol)
            symToReal_updates[var_hol] = var_real + I * var_im
            symToReal_updates[var_bar] = var_real - I * var_im
            realToHol_updates[var_real] = sp.Rational(1, 2) * (var_hol + sp.conjugate(var_hol))
            realToHol_updates[var_im] = I * sp.Rational(1, 2) * (sp.conjugate(var_hol) - var_hol)
            real_part_updates[var_hol] = var_real
            real_part_updates[var_bar] = var_real
            im_part_updates[var_hol] = var_im
            im_part_updates[var_bar] = -var_im

            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)


            # Register this single variable system in the registry.
            variable_registry['complex_variable_systems'][labelLoc1] = {
                    "family_type": "single",
                    "family_names": ((labelLoc1,), (labelLocBAR,), (labelLoc2,), (labelLoc3,)),
                    "family_values": (var_hol, var_bar, var_real, var_im),
                    "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                    "differential_system": True,
                    "initial_index": None,
                    "variable_relatives": {
                    labelLoc1: {
                        "complex_positioning": "holomorphic",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_hol,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": None,
                    },
                    labelLocBAR: {
                        "complex_positioning": "antiholomorphic",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_bar,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": None,
                    },
                    labelLoc2: {
                        "complex_positioning": "real",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_real,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": True,
                    },
                    labelLoc3: {
                        "complex_positioning": "imaginary",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_im,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": True,
                    }
                }
            }

            def create_differential_objects_single(var_hol, var_bar, var_real, var_im, default_var_format):
                if default_var_format == "real":
                    # Differential objects using the real/imaginary parts.
                    vf_instance_hol = VFClass((var_real, var_im), [sp.Rational(1, 2), -I/2], DGCVType="complex")
                    vf_instance_aHol = VFClass((var_real, var_im), [sp.Rational(1, 2), I/2], DGCVType="complex")
                    vf_instance_real = VFClass((var_real, var_im), [1, 0], DGCVType="complex")
                    vf_instance_im = VFClass((var_real, var_im), [0, 1], DGCVType="complex")
                    df_instance_hol = DFClass((var_real, var_im), {(0,): 1, (1,): I}, 1, DGCVType="complex")
                    df_instance_aHol = DFClass((var_real, var_im), {(0,): 1, (1,): -I}, 1, DGCVType="complex")
                    df_instance_real = DFClass((var_real, var_im), {(0,): 1}, 1, DGCVType="complex")
                    df_instance_im = DFClass((var_real, var_im), {(1,): 1}, 1, DGCVType="complex")
                else:  # default_var_format == "complex"
                    vf_instance_hol = VFClass((var_hol, var_bar), [1, 0], DGCVType="complex")
                    vf_instance_aHol = VFClass((var_hol, var_bar), [0, 1], DGCVType="complex")
                    vf_instance_real = VFClass((var_hol, var_bar), [1, 1], DGCVType="complex")
                    vf_instance_im = VFClass((var_hol, var_bar), [I, -I], DGCVType="complex")
                    df_instance_hol = DFClass((var_hol, var_bar), {(0,): 1}, 1, DGCVType="complex")
                    df_instance_aHol = DFClass((var_hol, var_bar), {(1,): 1}, 1, DGCVType="complex")
                    df_instance_real = DFClass((var_hol, var_bar), {(0,): sp.Rational(1,2),(1,): sp.Rational(1,2)}, 1, DGCVType="complex")
                    df_instance_im = DFClass((var_hol, var_bar), {(0,): -I/2, (1,): I/2}, 1, DGCVType="complex")
                return vf_instance_hol, vf_instance_aHol, vf_instance_real, vf_instance_im, df_instance_hol, df_instance_aHol, df_instance_real, df_instance_im
            # Create differential objects for single systems.
            vf_instance_hol, vf_instance_aHol, vf_instance_real, vf_instance_im, df_instance_hol, df_instance_aHol, df_instance_real, df_instance_im = create_differential_objects_single(
                var_hol, var_bar, var_real, var_im, default_var_format
            )

            # Register the differential objects in the caller's globals.
            _cached_caller_globals.update({
                f"D_{labelLoc1}": vf_instance_hol,
                f"D_{labelLocBAR}": vf_instance_aHol,
                f"d_{labelLoc1}": df_instance_hol,
                f"d_{labelLocBAR}": df_instance_aHol,
                f"D_{labelLoc2}": vf_instance_real,
                f"D_{labelLoc3}": vf_instance_im,
                f"d_{labelLoc2}": df_instance_real,
                f"d_{labelLoc3}": df_instance_im,
            })

            # Update find_parents.
            find_parents[var_real] = (var_hol, var_bar)
            find_parents[var_im] = (var_hol, var_bar)

            # Register VFClass and DFClass
            address = variable_registry['complex_variable_systems'][labelLoc1]["variable_relatives"]
            address[labelLoc1] |= {
                    "VFClass": _cached_caller_globals[f"D_{labelLoc1}"],
                    "DFClass": _cached_caller_globals[f"d_{labelLoc1}"]
            }
            address[labelLocBAR] |= {
                    "VFClass": _cached_caller_globals[f"D_{labelLocBAR}"],
                    "DFClass": _cached_caller_globals[f"d_{labelLocBAR}"]
            }
            address[labelLoc2] |= {
                    "VFClass": _cached_caller_globals[f"D_{labelLoc2}"],
                    "DFClass": _cached_caller_globals[f"d_{labelLoc2}"]
            }
            address[labelLoc3] |= {
                    "VFClass": _cached_caller_globals[f"D_{labelLoc3}"],
                    "DFClass": _cached_caller_globals[f"d_{labelLoc3}"]
            }

        # -------------------------------------------------
        # Tuple System Case (Phase 1: Build variables and update conversion info)
        # -------------------------------------------------
        elif isinstance(number_of_variables, int) and number_of_variables > 0:
            lengthLoc = number_of_variables
            variableProcedure(
                labelLoc1,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLocBAR,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc2,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc3,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )

            # Retrieve lists of created variables from the caller's globals.
            var_names1 = _cached_caller_globals[labelLoc1]
            var_namesBAR = _cached_caller_globals[labelLocBAR]
            var_names2 = _cached_caller_globals[labelLoc2]
            var_names3 = _cached_caller_globals[labelLoc3]

            # Build string labels for registry.
            var_str1 = tuple(f"{labelLoc1}{i}" for i in range(initialIndex, lengthLoc + initialIndex))
            var_strBAR = tuple(f"{labelLocBAR}{i}" for i in range(initialIndex, lengthLoc + initialIndex))
            var_str2 = tuple(f"{labelLoc2}{i}" for i in range(initialIndex, lengthLoc + initialIndex))
            var_str3 = tuple(f"{labelLoc3}{i}" for i in range(initialIndex, lengthLoc + initialIndex))

            # Register the tuple system (the differential objects will be added in Phase 2).
            complex_system_updates[labelLoc1] = {
                "family_type": "tuple",
                "family_names": (var_str1, var_strBAR, var_str2, var_str3),
                "family_values": (var_names1, var_namesBAR, var_names2, var_names3),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": initialIndex,
                "variable_relatives": {},
            }

            # Accumulate conversion updates for each tuple element.
            totalVarListLoc = list(zip(var_names1, var_namesBAR, var_names2, var_names3))
            for idx, (comp_var, bar_comp_var, real_var, imag_var) in enumerate(totalVarListLoc):
                find_parents[real_var] = (comp_var, bar_comp_var)
                find_parents[imag_var] = (comp_var, bar_comp_var)

                conj_updates[comp_var] = bar_comp_var
                conj_updates[bar_comp_var] = comp_var
                holToReal_updates[comp_var] = real_var + I * imag_var
                realToSym_updates[real_var] = sp.Rational(1, 2) * (comp_var + bar_comp_var)
                realToSym_updates[imag_var] = -I * sp.Rational(1, 2) * (comp_var - bar_comp_var)
                symToHol_updates[bar_comp_var] = sp.conjugate(comp_var)
                symToReal_updates[comp_var] = real_var + I * imag_var
                symToReal_updates[bar_comp_var] = real_var - I * imag_var
                realToHol_updates[real_var] = sp.Rational(1, 2) * (comp_var + sp.conjugate(comp_var))
                realToHol_updates[imag_var] = I * sp.Rational(1, 2) * (sp.conjugate(comp_var) - comp_var)
                real_part_updates[comp_var] = real_var
                real_part_updates[bar_comp_var] = real_var
                im_part_updates[comp_var] = imag_var
                im_part_updates[bar_comp_var] = -imag_var

            # Save tuple system info for Phase 2.
            tuple_system_data.append((labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc))

            # Batch update the conversion dictionaries.
            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)            
        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided."
            )

    # End main loop.
    # Update the registry with the constructed complex variable systems.
    variable_registry["complex_variable_systems"].update(complex_system_updates)

    # --------------------------------------------------
    # Phase 2: Create VFClass and DFClass objects for tuple systems.
    # --------------------------------------------------
    for (labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc) in tuple_system_data:
        relatives = variable_registry["complex_variable_systems"][labelLoc1]["variable_relatives"]
        # Build a list of tuples for each element in the tuple system.
        # Each tuple: (comp_var, bar_comp_var, real_var, imag_var)
        totalVarListLoc = list(zip(var_names1, var_namesBAR, var_names2, var_names3))

        # --- Batch conversion dictionary updates ---
        # These updates are built over the entire tuple system.
        conj_updates_batch = {comp: anti for comp, anti, _, _ in totalVarListLoc}
        conj_updates_batch.update({anti: comp for comp, anti, _, _ in totalVarListLoc})

        holToReal_updates_batch = {comp: real + I * imag for comp, _, real, imag in totalVarListLoc}

        realToSym_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            realToSym_updates_batch[real] = sp.Rational(1, 2) * (comp + anti)
            realToSym_updates_batch[imag] = -I * sp.Rational(1, 2) * (comp - anti)

        symToHol_updates_batch = {anti: sp.conjugate(comp) for comp, anti, _, _ in totalVarListLoc}

        symToReal_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            symToReal_updates_batch[comp] = real + I * imag
            symToReal_updates_batch[anti] = real - I * imag

        realToHol_updates_batch = {}
        for comp, _, real, _ in totalVarListLoc:
            realToHol_updates_batch[real] = sp.Rational(1, 2) * (comp + sp.conjugate(comp))
        for comp, _, _, imag in totalVarListLoc:
            realToHol_updates_batch[imag] = I * sp.Rational(1, 2) * (sp.conjugate(comp) - comp)

        real_part_updates_batch = {}
        im_part_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            real_part_updates_batch[comp] = real
            real_part_updates_batch[anti] = real
            im_part_updates_batch[comp] = imag
            im_part_updates_batch[anti] = -imag

        # --- End Batch Updates ---

        # Now, for each tuple element, create the differential objects.
        # We loop over each element (indexed by idx) in totalVarListLoc.
        for idx, (comp_var, bar_comp_var, real_var, imag_var) in enumerate(totalVarListLoc):
            if default_var_format == "real":
                # Use the original formulas based on real/imaginary parts.
                N = len(var_names2)
                D_comp = VFClass(
                    var_names2 + var_names3,
                    [sp.Rational(1, 2) if i == idx else 0 for i in range(lengthLoc)]
                    + [-I / 2 if i == idx else 0 for i in range(lengthLoc)],
                    "complex",
                )
                D_bar_comp = VFClass(
                    var_names2 + var_names3,
                    [sp.Rational(1, 2) if i == idx else 0 for i in range(lengthLoc)]
                    + [I / 2 if i == idx else 0 for i in range(lengthLoc)],
                    "complex",
                )
                D_real = VFClass(
                    var_names2 + var_names3,
                    [1 if i == idx else 0 for i in range(lengthLoc)]
                    + [0] * N,
                    "complex",
                )
                D_im = VFClass(
                    var_names2 + var_names3,
                    [0] * N
                    + [1 if i == idx else 0 for i in range(lengthLoc)],
                    "complex",
                )
                d_comp = DFClass(
                    var_names2 + var_names3, {(idx,): 1, (idx + N,): I}, 1, "complex"
                )
                d_bar_comp = DFClass(
                    var_names2 + var_names3,
                    {(idx,): 1, (idx + N,): -I},
                    1,
                    "complex",
                )
                d_real = DFClass(
                    var_names2 + var_names3, {(idx,): 1}, 1, "complex"
                )
                d_im = DFClass(
                    var_names2 + var_names3,
                    {(idx + N,):1},
                    1,
                    "complex",
                )
            else:  # default_var_format == "complex"
                # Here we use the holomorphic/antiholomorphic formulas.
                # Note: len(var_names1) equals the number of tuple elements.
                N = len(var_names1)
                D_comp = VFClass(
                    var_names1 + var_namesBAR,
                    [1 if i == idx else 0 for i in range(N)] + [0] * N,
                    "complex",
                )
                D_bar_comp = VFClass(
                    var_names1 + var_namesBAR,
                    [0] * N + [1 if i == idx else 0 for i in range(N)],
                    "complex",
                )
                D_real = VFClass(
                    var_names1 + var_namesBAR,
                    [1 if i == idx else 0 for i in range(N)]
                    + [1 if i == idx else 0 for i in range(N)],
                    "complex",
                )
                D_im = VFClass(
                    var_names1 + var_namesBAR,
                    [I if i == idx else 0 for i in range(N)]
                    + [-I if i == idx else 0 for i in range(N)],
                    "complex",
                )
                d_comp = DFClass(
                    var_names1 + var_namesBAR, {(idx,): 1}, 1, "complex"
                )
                d_bar_comp = DFClass(
                    var_names1 + var_namesBAR, {(idx + N,): 1}, 1, "complex"
                )
                d_real = DFClass(
                    var_names1 + var_namesBAR, {(idx,): sp.Rational(1,2), (idx + N,): sp.Rational(1,2)}, 1, "complex"
                )
                d_im = DFClass(
                    var_names1 + var_namesBAR,
                    {(idx,): -I/2, (idx + N,): I/2},
                    1,
                    "complex",
                )

            # Register the differential objects in the caller's globals.
            _cached_caller_globals[f"D_{comp_var}"] = D_comp
            _cached_caller_globals[f"D_{bar_comp_var}"] = D_bar_comp
            _cached_caller_globals[f"d_{comp_var}"] = d_comp
            _cached_caller_globals[f"d_{bar_comp_var}"] = d_bar_comp
            _cached_caller_globals[f"D_{real_var}"] = D_real
            _cached_caller_globals[f"D_{imag_var}"] = D_im
            _cached_caller_globals[f"d_{real_var}"] = d_real
            _cached_caller_globals[f"d_{imag_var}"] = d_im

            # Update variable_relatives for this tuple element.
            relatives[str(comp_var)] = {
                "complex_positioning": "holomorphic",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": comp_var,
                "VFClass": D_comp,
                "DFClass": d_comp,
                "assumeReal": None,
            }
            relatives[str(bar_comp_var)] = {
                "complex_positioning": "antiholomorphic",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": bar_comp_var,
                "VFClass": D_bar_comp,
                "DFClass": d_bar_comp,
                "assumeReal": None,
            }
            relatives[str(real_var)] = {
                "complex_positioning": "real",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": real_var,
                "VFClass": _cached_caller_globals[f"D_{real_var}"],
                "DFClass": _cached_caller_globals[f"d_{real_var}"],
                "assumeReal": True,
            }
            relatives[str(imag_var)] = {
                "complex_positioning": "imaginary",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": imag_var,
                "VFClass": _cached_caller_globals[f"D_{imag_var}"],
                "DFClass": _cached_caller_globals[f"d_{imag_var}"],
                "assumeReal": True,
            }
        # End of Phase 2 loop for this tuple system.
    if number_of_variables is not None:
        # After processing all tuple systems, update the conversion dictionaries in one bulk call:
        conv["conjugation"].update(conj_updates_batch)
        conv["holToReal"].update(holToReal_updates_batch)
        conv["realToSym"].update(realToSym_updates_batch)
        conv["symToHol"].update(symToHol_updates_batch)
        conv["symToReal"].update(symToReal_updates_batch)
        conv["realToHol"].update(realToHol_updates_batch)
        conv["real_part"].update(real_part_updates_batch)
        conv["im_part"].update(im_part_updates_batch)

def _format_complex_coordinates(
    coordinate_tuple, default_var_format="complex", pass_error_report=None
):
    """
    Format var. lists consisting of variables within DGCV complex variable systems, formatting as real or holomorphic and completeing the basis as needed (i.e., adding BARz if only z is present, adding y if only x, etc.)
    """
    vr = get_variable_registry()
    exaustList = list(coordinate_tuple)
    newList1 = []
    newList2 = []
    try:
        for var in coordinate_tuple:
            if var in exaustList:
                varStr = str(var)
                for parent in vr["complex_variable_systems"]:
                    if (
                        varStr
                        in vr["complex_variable_systems"][parent]["variable_relatives"]
                    ):
                        foundVars = vr["complex_variable_systems"][parent][
                            "variable_relatives"
                        ][varStr]["complex_family"]
                        if default_var_format == "complex":
                            newList1 = newList1 + [foundVars[0]]
                            newList2 = newList2 + [foundVars[1]]
                        else:
                            newList1 = newList1 + [foundVars[2]]
                            newList2 = newList2 + [foundVars[3]]
                        for j in foundVars:
                            if j in exaustList:
                                exaustList.remove(j)
    except KeyError:
        if pass_error_report == retrieve_passkey():
            return "At least one element in the given variable list is not registered as part of a complex variable system in the DGCV variable management framework."
    return tuple(newList1 + newList2)


############## variable format conversion


def _VFDF_conversion(obj, default_var_format=None, _converter=None, coeffsOnly = None):
    def converter(expr, _conv):     # typically invoking some recursion. E.g., symToReal may call _VFDF_conversion and have converter() apply symToReal to object components.
        if _conv is None:
            return expr
        return _conv(expr)
    if coeffsOnly is not False:
        if isinstance(obj,(VFClass,DFClass)):
            return obj.subs(coeffsOnly)
    if default_var_format == "complex":
        if isinstance(obj, VFClass):
            if obj.DGCVType == "standard":
                return VFClass(
                    obj.varSpace,
                    [converter(j, _converter) for j in obj.coeffs],
                    DGCVType=obj.DGCVType,
                    _simplifyKW=obj._simplifyKW,
                )
            varSpace = obj.cd_formats["compCoeffDataDict"][0]
            coeffsDict = obj.cd_formats["compCoeffDataDict"][1]
            coeffs = [converter(coeffsDict[(j,)], _converter) for j in range(len(varSpace))]
            return VFClass(
                varSpace, coeffs, DGCVType=obj.DGCVType, _simplifyKW=obj._simplifyKW
            )
        elif isinstance(obj, DFClass):
            specialCase = obj.DGCVType == "complex" and obj._varSpace_type == "complex"
            if obj.DGCVType == "standard" or specialCase:
                return DFClass(
                    obj.varSpace,
                    {
                        a: converter(b, _converter)
                        for a, b in obj.DFClassDataDict.items()
                    },
                    obj.degree,
                    DGCVType=obj.DGCVType,
                    _simplifyKW=obj._simplifyKW,
                )
            varSpace = obj.cd_formats["compCoeffDataDict"][0]
            dataDict = obj.cd_formats["compCoeffDataDict"][1]
            dataDict = {a: converter(b, _converter) for a, b in dataDict.items()}
            return DFClass(
                varSpace,
                dataDict,
                obj.degree,
                DGCVType=obj.DGCVType,
                _simplifyKW=obj._simplifyKW,
            )
    elif default_var_format == "real":
        if isinstance(obj, VFClass):
            if obj.DGCVType == "standard":
                return VFClass(
                    obj.varSpace,
                    [converter(a, _converter) for a in obj.coeffs],
                    DGCVType=obj.DGCVType,
                    _simplifyKW=obj._simplifyKW,
                )
            varSpace = obj.cd_formats["realCoeffDataDict"][0]
            coeffsDict = obj.cd_formats["realCoeffDataDict"][1]
            coeffs = [converter(coeffsDict[(j,)], _converter) for j in range(len(varSpace))]
            return VFClass(
                varSpace, coeffs, DGCVType=obj.DGCVType, _simplifyKW=obj._simplifyKW
            )
        elif isinstance(obj, DFClass):
            specialCase = obj.DGCVType == "complex" and obj._varSpace_type == "real"
            if obj.DGCVType == "standard" or specialCase:
                return DFClass(
                    obj.varSpace,
                    {
                        a: converter(b, _converter)
                        for a, b in obj.DFClassDataDict.items()
                    },
                    obj.degree,
                    DGCVType=obj.DGCVType,
                    _simplifyKW=obj._simplifyKW,
                )
            varSpace = obj.cd_formats["realCoeffDataDict"][0]
            dataDict = obj.cd_formats["realCoeffDataDict"][1]
            dataDict = {a: converter(b, _converter) for a, b in dataDict.items()}
            return DFClass(
                varSpace,
                dataDict,
                obj.degree,
                DGCVType=obj.DGCVType,
                _simplifyKW=obj._simplifyKW,
            )

def holToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts holomorphic variables in the expression to real variables.

    Parameters:
        expr : sympy.Expr or VFClass/DFClass
            The expression to convert.
        skipVar : list of str, optional
            A list of holomorphic variable system labels to skip during conversion.
            For any variable in skipVar, the associated holomorphic variables will
            not be substituted.
        simplify_everything : bool, optional
            If True, simplifies the resulting expression.

    Returns:
        input type
            The expression with holomorphic variables replaced by real variables,
            except for those specified in skipVar.
    """
    vr = get_variable_registry()
    # Work on a copy so as not to modify the original conversion dict.
    conv_holToReal = vr.get("conversion_dictionaries", {}).get("holToReal", {}).copy()

    # If skipVar is provided, remove the holomorphic variables from conv_holToReal
    if skipVar:
        for var in skipVar:
            complex_system = vr.get("complex_variable_systems", {}).get(var, {})
            family_values = complex_system.get("family_values", ((), (), (), ()))
            # For a single variable system, ensure family_values is iterable as tuple of iterables.
            if complex_system.get("family_type") == "single":
                family_values = tuple([j] for j in family_values)
            # Assume the first tuple contains the holomorphic variables.
            holomorphic_vars = family_values[0] if len(family_values) > 0 else ()
            for hol_var in holomorphic_vars:
                if hol_var in conv_holToReal:
                    conv_holToReal.pop(hol_var)

    if isinstance(expr, (VFClass, DFClass)) and simplify_everything:
        co = False if simplify_everything is True else conv_holToReal
        return _VFDF_conversion(expr, default_var_format="real", _converter=holToReal, coeffsOnly = co)

    if isinstance(expr, (int, float)):
        return expr

    if isinstance(expr, sp.Symbol):
        if expr in conv_holToReal:
            return conv_holToReal[expr]
        else:
            return expr

    if isinstance(expr, sp.Expr):
        filtered_subs = {sym: conv_holToReal[sym] for sym in expr.free_symbols if sym in conv_holToReal}
        result = sp.sympify(expr).subs(filtered_subs)
        if simplify_everything:
            result = sp.simplify(result)
        return result

    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(lambda e: holToReal(e, skipVar=skipVar, simplify_everything=simplify_everything))
    return expr

def realToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts real variables in the expression to symbolic conjugates.

    Parameters:
        expr : sympy.Expr or VFClass/DFClass
            The expression to convert.
        skipVar : list of str, optional
            A list of real variable system labels to skip during conversion.
            For any variable in skipVar, the associated real and imaginary variables
            will not be substituted.
        simplify_everything : bool, optional
            If True, simplifies the resulting expression.

    Returns:
        input type
            The expression with real variables replaced by symbolic conjugates,
            except for those specified in skipVar.
    """
    vr = get_variable_registry()
    conv_realToSym = vr.get("conversion_dictionaries", {}).get("realToSym", {}).copy()

    # If skipVar is provided, remove the corresponding real and imaginary variables from conv_realToSym.
    if skipVar:
        for var in skipVar:
            complex_system = vr.get("complex_variable_systems", {}).get(var, {})
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                # Ensure family_values is a tuple of iterables
                family_values = tuple([(j,) for j in family_values])
            # Assume the third and fourth tuples are real and imaginary variables, respectively.
            real_vars = family_values[2] if len(family_values) > 2 else ()
            imag_vars = family_values[3] if len(family_values) > 3 else ()
            for real_var in real_vars + imag_vars:
                if real_var in conv_realToSym:
                    conv_realToSym.pop(real_var)

    if isinstance(expr, (VFClass, DFClass)) and simplify_everything:
        co = False if simplify_everything is True else conv_realToSym
        return _VFDF_conversion(expr, default_var_format="complex", _converter=realToSym, coeffsOnly = co)

    if isinstance(expr, (int, float)):
        return expr

    if isinstance(expr, sp.Symbol):
        if expr in conv_realToSym:
            return conv_realToSym[expr]
        else:
            return expr

    if isinstance(expr, sp.Expr):
        filtered_subs = {
            sym: conv_realToSym[sym]
            for sym in expr.free_symbols
            if sym in conv_realToSym
        }
        result = sp.sympify(expr).subs(filtered_subs)
        if simplify_everything:
            result = sp.simplify(result)
        return result

    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(lambda e: realToSym(e, skipVar=skipVar, simplify_everything=simplify_everything))
    return expr

def symToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts symbolic conjugated variables in the expression to holomorphic variables.

    Parameters:
        expr : sympy.Expr or VFClass/DFClass
            The expression to convert.
        skipVar : list of str, optional
            A list of holomorphic variable system labels to skip during conversion.
            For any variable in skipVar, the associated antiholomorphic variables
            will not be substituted.
        simplify_everything : bool, optional
            If True, simplifies the resulting expression.

    Returns:
        Same type as input:
            The expression with symbolic conjugates replaced by holomorphic variables,
            except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    # Work on a copy so as not to modify the original conversion dict.
    conversion_dict = variable_registry.get("conversion_dictionaries", {}).get("symToHol", {}).copy()

    # If skipVar is provided, remove the antiholomorphic keys for those variable systems.
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get("complex_variable_systems", {}).get(var, {})
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                family_values = tuple([(j,) for j in family_values])
            # Assume the second tuple contains the antiholomorphic variables.
            antiholomorphic_vars = family_values[1] if len(family_values) > 1 else ()
            for anti_var in antiholomorphic_vars:
                if anti_var in conversion_dict:
                    del conversion_dict[anti_var]

    if isinstance(expr, (VFClass, DFClass)) and simplify_everything:
        co = False if simplify_everything is True else conversion_dict
        return _VFDF_conversion(expr, default_var_format="complex", _converter=symToHol, coeffsOnly = co)
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, sp.Symbol):
        if expr in conversion_dict:
            return conversion_dict[expr]
        else:
            return expr
    if isinstance(expr, sp.Expr):
        filtered_subs = {sym: conversion_dict[sym] for sym in expr.free_symbols if sym in conversion_dict}
        result = expr.subs(filtered_subs)
        if simplify_everything:
            result = sp.simplify(result)
        return result
    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(lambda e: symToHol(e, skipVar=skipVar, simplify_everything=simplify_everything))
    return expr


def holToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts holomorphic variables in the expression to symbolic conjugates.
    This is done by first converting holomorphic variables to real variables,
    and then converting real variables to symbolic conjugates.

    Note: This process will also convert any present real variables
    (both real and imaginary parts) to their symbolic conjugate format.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of holomorphic variable system labels to skip during conversion.
        For any variable in skipVar, the associated holomorphic variables and
        their real counterparts will not be substituted.

    Returns:
    sympy.Expr
        The expression with holomorphic variables (and any present real variables)
        replaced by symbolic conjugates, except for those specified in skipVar.
    """
    # First apply holToReal()
    expr = holToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    # Then apply realToSym()
    expr = realToSym(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    return expr


def cleanUpConjugation(arg1, skipVar=None):
    """
    While working with symbolic conjugate expressions it can easily happen that sympy generates expressions having actual conjugation applied to symbolic conjugate variables. Apply this command to simplify such redundancies.

    This is often necessary for *solve* functions to succeed.

    Args:
        arg1: sympy expression

    Returns:
        sympy expression

    Raises:
        NA
    """
    return realToSym(
        symToReal(arg1, skipVar=skipVar, simplify_everything=False),
        skipVar=skipVar,
        simplify_everything=False,
    )


def realToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts real variables in the expression to holomorphic variables.

    Parameters:
        expr : sympy.Expr
            The expression to convert.
        skipVar : list of str, optional
            A list of real variable system labels to skip during conversion.
            For any variable in skipVar, the associated real and imaginary variables
            will not be substituted.
        simplify_everything : bool, optional
            If True, simplifies the resulting expression.

    Returns:
        sympy.Expr
            The expression with real variables replaced by holomorphic variables,
            except for those specified in skipVar.
    """
    import sympy as sp
    # Get variable registry and conversion dictionary
    vr = get_variable_registry()
    # We'll work on a copy so as not to modify the original dict
    conv_realToHol = vr.get("conversion_dictionaries", {}).get("realToHol", {}).copy()

    # If skipVar provided, remove the corresponding real and imaginary variables from conv_realToHol.
    if skipVar:
        for var in skipVar:
            complex_system = vr.get("complex_variable_systems", {}).get(var, {})
            family_values = complex_system.get("family_values", ((), (), (), ()))
            # For a single system, ensure family_values is treated as tuple of iterables
            if complex_system.get("family_type") == "single":
                family_values = tuple([j] for j in family_values)
            # Assume the third and fourth tuples are real and imaginary parts.
            real_vars = family_values[2] if len(family_values) > 2 else ()
            imag_vars = family_values[3] if len(family_values) > 3 else ()
            for real_var in real_vars + imag_vars:
                if real_var in conv_realToHol:
                    conv_realToHol.pop(real_var)


    if isinstance(expr, (VFClass, DFClass)) and simplify_everything:
        co = False if simplify_everything is True else conv_realToHol
        return _VFDF_conversion(
            expr, default_var_format="complex", _converter=realToHol, coeffsOnly = co
        )
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, sp.Symbol):
        if expr in conv_realToHol:
            return conv_realToHol[expr]
        else:
            return expr
    if isinstance(expr, sp.Expr):
        # Build a reformatted substitution dictionary only for symbols that occur in expr.free_symbols.
        reformatted_subs_dict = {
            sym: conv_realToHol[sym] for sym in expr.free_symbols if sym in conv_realToHol
        }
        result = sp.sympify(expr).subs(reformatted_subs_dict)
        if simplify_everything:
            result = sp.simplify(result)
        return result
    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(lambda e: realToHol(e, skipVar=skipVar, simplify_everything=simplify_everything))
    return expr


def symToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts symbolic conjugates in the expression to real variables.

    Parameters:
        expr : sympy.Expr or VFClass/DFClass
            The expression to convert.
        skipVar : list of str, optional
            A list of symbolic conjugate system labels to skip during conversion.
            For any variable in skipVar, the associated symbolic conjugates will not be substituted.
        simplify_everything : bool, optional
            If True, simplifies the resulting expression.

    Returns:
        Same type as input
            The expression with symbolic conjugates replaced by real variables,
            except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    conversion_dict = variable_registry.get("conversion_dictionaries", {}).get("symToReal", {}).copy()

    # If skipVar is provided, remove the symbolic conjugate keys for the given systems.
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get("complex_variable_systems", {}).get(var, {})
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                # Ensure family_values is a tuple of iterables
                family_values = tuple([(j,) for j in family_values])
            # Assume the second tuple contains the symbolic conjugate (antiholomorphic) variables.
            anti_vars = family_values[1] if len(family_values) > 1 else ()
            for anti_var in anti_vars:
                if anti_var in conversion_dict:
                    del conversion_dict[anti_var]

    if isinstance(expr, (VFClass, DFClass)) and simplify_everything:
        co = False if simplify_everything is True else conversion_dict
        return _VFDF_conversion(expr, default_var_format="real", _converter=symToReal, coeffsOnly = co)

    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, sp.Symbol):
        if expr in conversion_dict:
            return conversion_dict[expr]
        else:
            return expr
    if isinstance(expr, sp.Expr):
        filtered_subs = {sym: conversion_dict[sym]
                         for sym in expr.free_symbols if sym in conversion_dict}
        result = expr.subs(filtered_subs)
        if simplify_everything:
            result = sp.simplify(result)
        return result
    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(lambda e: symToReal(e, skipVar=skipVar, simplify_everything=simplify_everything))
    return expr

def allToReal(expr, skipVar=None, simplify_everything=True):
    """
    ⚠️ ... not to be confused with "all too real"! 😊

    Converts all variables (holomorphic, symbolic conjugates, and real) to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
        The expression with all variables converted to real variables,
        except for those specified in skipVar.
    """
    # First convert symbolic conjugates to real variables
    expr = symToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    # Then convert holomorphic variables to real variables
    expr = holToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    return expr


def allToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables (real, symbolic conjugates, and holomorphic) to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
        The expression with all variables converted to holomorphic variables,
        except for those specified in skipVar.
    """

    if isinstance(expr, list):
        return [allToHol(j) for j in expr]
    if isinstance(expr, tuple):
        return [allToHol(j) for j in expr]

    # First convert symbolic conjugates to real variables
    expr = symToHol(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    # Then convert real variables to holomorphic variables
    expr = realToHol(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    return expr


def allToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables (holomorphic, real, and symbolic conjugates) to symbolic conjugates.

    Note: This process will also convert any present real variables
    (both real and imaginary parts) to their symbolic conjugate format.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
        The expression with all variables converted to symbolic conjugates,
        except for those specified in skipVar.
    """
    return holToSym(expr, skipVar=skipVar, simplify_everything=simplify_everything)


def _remove_complex_handling(arg):
    if isinstance(arg, VFClass):
        return VFClass(arg.varSpace, arg.coeffs)
    if isinstance(arg, DFClass):
        return DFClass(arg.varSpace, arg.DFClassDataDict, arg.degree)


def complex_struct_op(vf):
    if isinstance(vf, VFClass):
        if vf.DGCVType == "standard":
            return vf
        elif vf._varSpace_type == "real":
            newVarSpace = vf.cd_formats["realCoeffDataDict"][0]
            cd_data = vf.cd_formats["realCoeffDataDict"][1]
            CDim = len(vf.holVarSpace)
            coeffs1 = []
            coeffs2 = []
            for j in range(CDim):
                yIndex = (CDim + j,)
                xIndex = (j,)
                coeffs1 = coeffs1 + [-cd_data[yIndex]]
                coeffs2 = coeffs2 + [cd_data[xIndex]]
            return VFClass(
                newVarSpace,
                coeffs1 + coeffs2,
                DGCVType="complex",
                _simplifyKW=vf._simplifyKW,
            )
        elif vf._varSpace_type == "complex":
            newVarSpace = vf.cd_formats["compCoeffDataDict"][0]
            cd_data = vf.cd_formats["compCoeffDataDict"][1]
            CDim = len(vf.holVarSpace)
            coeffs1 = []
            coeffs2 = []
            for j in range(CDim):
                zIndex = (j,)
                BARzIndex = (CDim + j,)
                coeffs1 = coeffs1 + [I * cd_data[zIndex]]
                coeffs2 = coeffs2 + [-I * cd_data[BARzIndex]]
            return VFClass(
                newVarSpace,
                coeffs1 + coeffs2,
                DGCVType="complex",
                _simplifyKW=vf._simplifyKW,
            )


def conjugate_DGCV(expr):
    if isinstance(expr, (VFClass, DFClass)):
        return _conjComplexVFDF(expr)
    else:
        return sp.conjugate(expr)


def conj_with_real_coor(expr):
    return allToReal(expr).subs({I: -I})


def re_with_real_coor(expr):
    expr = allToReal(expr)
    s = sp.simplify(sp.Rational(1, 2) * (expr + conj_with_real_coor(expr)))
    return s


def im_with_real_coor(expr):
    expr = allToReal(expr)
    s = sp.simplify(-I * sp.Rational(1, 2) * (expr - conj_with_real_coor(expr)))
    return s


def conj_with_hol_coor(expr):
    vr = get_variable_registry()
    subsDictA = vr["conversion_dictionaries"]["conjugation"]
    subsDict = subsDictA | {I: -I}
    return allToSym(expr).subs(subsDict, simultaneous=True)


def re_with_hol_coor(expr):
    expr = allToSym(expr)
    s = sp.simplify(sp.Rational(1, 2) * (expr + conj_with_hol_coor(expr)))
    return s


def im_with_hol_coor(expr):
    expr = allToSym(expr)
    s = sp.simplify(-I * sp.Rational(1, 2) * (expr - conj_with_hol_coor(expr)))
    return s


############## basic vector field and differential forms operations


def VF_coeffs_direct(vf, var_space, sparse=False):
    """
    Depricated: Use `VF_coeffs` instead.
    """
    if not isinstance(vf, VFClass):
        raise TypeError("Expected first argument to be an instance of VFClass")

    if not isinstance(var_space, (list, tuple)):
        raise TypeError("Expected second argument to be a list or tuple of variables")

    # Evaluate the vector field on each element in var_space
    coeffs = [vf(var) for var in var_space]

    # Return sparse or full result
    if sparse:
        return [((i,), coeffs[i]) for i in range(len(coeffs)) if coeffs[i] != 0] or [
            ((0,), 0)
        ]
    return coeffs


def minimalVFDataDict(vf):
    if isinstance(vf, VFClass):
        return {
            vf.varSpace[j]: vf.coeffs[j]
            for j in range(len(vf.varSpace))
            if vf.coeffs[j] != 0
        }


def minimalDFDataDict(df):
    varTuple = df.varSpace
    list_collection = df.DFClassDataMinimal
    # Find all integers missing from the lists in list_collection
    full_range = set(range(len(varTuple)))
    found_indices = set()

    for lst, _ in list_collection:
        found_indices.update(lst)

    missing_indices = full_range - found_indices

    # Create new_varTuple by removing elements at missing indices
    new_varTuple = tuple(
        varTuple[i] for i in range(len(varTuple)) if i not in missing_indices
    )

    # Create a mapping of old indices to new indices
    index_mapping = {}
    new_index = 0
    for old_index in range(len(varTuple)):
        if old_index not in missing_indices:
            index_mapping[old_index] = new_index
            new_index += 1

    # Reformat the lists in list_collection to use new indices
    new_DFDataDict = dict()
    for lst, val in list_collection:
        new_inex = tuple([index_mapping[old_index] for old_index in lst])
        new_DFDataDict[new_inex] = val

    return new_varTuple, new_DFDataDict


def compressDGCVClass(obj):
    """Removes superfluous variables from the variable space that a VFClass or DFClass object is defined w.r.t."""
    if isinstance(obj, DFClass):
        newVarSpace, newDFData = minimalDFDataDict(obj)
        return DFClass(
            newVarSpace,
            newDFData,
            obj.degree,
            DGCVType=obj.DGCVType,
            _simplifyKW=obj._simplifyKW,
        )
    elif isinstance(obj, VFClass):
        VFData = minimalVFDataDict(obj)
        newVarSpace = list(VFData.keys())
        newCoeffs = [VFData[j] for j in newVarSpace]
        return VFClass(
            newVarSpace, newCoeffs, DGCVType=obj.DGCVType, _simplifyKW=obj._simplifyKW
        )


def VF_coeffs(vf, var_list, sparse=False):
    """
    Expresses coefficients of the vector field with respect to the list of variables in var_list,
    assuming var_list contains at least all of the basis variables found in the varSpace attribute of vf.
    It also works if var_list merely contains the minimum set of such variables needed to define the vector field,
    and returns an error if not.

    Additionally, if vf has DGCVType="complex" then var_list can have variables in real or holomorphic coordinates
    but these types cannot be mixed.

    Args:
        vf: a vector field (an instance of VFClass)
        var_list: a list or tuple containing Sympy Symbol objects

    Returns:
        List of Sympy expressions corresponding to the coefficients of vf with respect to each symbol in var_list.

    Raises:
        TypeError if vf is not a VFClass or if the symbols in var_list are not all from one coordinate format.
    """
    # Get the DGCV variable registry
    variable_registry = get_variable_registry()

    # Ensure vf is a VFClass instance
    if not isinstance(vf, VFClass):
        raise TypeError(f"VF_coeffs expects the first argument to be an instance of VFClass, not type: {type(vf)}")

    # For complex vector fields, check that the variable list is uniformly in one coordinate format.
    if vf.DGCVType == "complex":
        conv_realToSym = variable_registry["conversion_dictionaries"]["realToSym"]
        conv_symToReal = variable_registry["conversion_dictionaries"]["symToReal"]
        if all(var in conv_realToSym for var in var_list):
            # All variables are in real coordinates.
            if vf._varSpace_type == "complex":
                vf = allToReal(vf)
        elif all(var in conv_symToReal for var in var_list):
            # All variables are in holomorphic coordinates.
            if vf._varSpace_type == "real":
                vf = allToSym(vf)
        else:
            raise TypeError(
                "The VF_coeffs function was given a VFClass with DGCVType='complex' while the variable list contains "
                "a mixture of real and holomorphic coordinates. Variables should be provided in just one coordinate format."
            )

    # Get a minimal vector field data dictionary mapping basis variables to their coefficients.
    MVFD = minimalVFDataDict(vf)


    coeffs = [MVFD.get(var, 0) for var in var_list]

    # Return sparse or full result
    if sparse:
        return [((i,), coeffs[i]) for i in range(len(coeffs)) if coeffs[i] != 0] or [
            ((0,), 0)
        ]
    return coeffs

def changeVFBasis(arg1, arg2):
    return VFClass(arg2, VF_coeffs(arg1, arg2), arg1.DGCVType)


def addVF(*args):
    """
    Adds the given vector fields (i.e., VFClass instances).

    Args:
        *args: any number of VFClass class instances

    Returns:
        A VFClass instance

    Raises:
        Exception: If any argument is not an instance of VFClass
    """
    if len(args) == 0:
        return args
    if all([isinstance(j, VFClass) for j in args]):
        typeList = list(set([j.DGCVType for j in args]))
        if len(typeList) == 1 and typeList[0] == "complex":
            typeLoc = "complex"
        else:
            typeLoc = "standard"

        # Initialize an empty list for varSpace and extend it with each j.varSpace
        varSpaceLoc = []
        for j in args:
            varSpaceLoc.extend(j.varSpace)

        # Convert to a tuple with unique entries using dict.fromkeys
        varSpaceLoc = tuple(dict.fromkeys(varSpaceLoc))

        coeffListLoc = [VF_coeffs(j, varSpaceLoc) for j in args]
        coeffsLoc = [sum(j) for j in zip(*coeffListLoc)]

        return VFClass(varSpaceLoc, coeffsLoc, typeLoc)
    else:
        raise Exception("Expected all arguments to be instances of the VFClass class.")


def scaleVF(scalar, vector_field):
    """
    Scales the given vector field (i.e., VFClass instance) *vector_field* by the scalar function (i.e., SymPy expression) *scalar*.

    Args:
        scalar: A SymPy expression representing the scalar multiplier.
        vector_field: A VFClass instance representing the vector field to scale.

    Returns:
        A new VFClass instance representing the scaled vector field.

    Raises:
        TypeError: If the second argument is not an instance of the VFClass class.
    """
    if isinstance(vector_field, VFClass):
        new_coeffs = [scalar * coeff for coeff in vector_field.coeffs]
        return VFClass(vector_field.varSpace, new_coeffs, vector_field.DGCVType)
    else:
        raise TypeError(
            "Expected second argument to be an instance of the VFClass class."
        )


def VF_bracket(arg1, arg2, doNotSimplify=False, fast_algorithm=True):
    """
    Computes the Lie bracket of two vector fields (i.e., VFClass instances) *arg1* and *arg2* that are
    defined in the same or related variable space(s). The function supports both standard and complex
    vector fields and offers an optimized fast algorithm for standard cases.

    Parameters:
    -----------
    arg1 : VFClass
        The first vector field to compute the Lie bracket.
    arg2 : VFClass
        The second vector field to compute the Lie bracket.
    doNotSimplify : bool, optional
        If True, the result is returned without simplification (default is False).
    fast_algorithm : bool, optional
        If True, the function uses a faster algorithm that converts all coefficients to real variables
        for vector fields, reducing overhead but sacrificing variable format fidelity (default is True).

    Returns:
    --------
    VFClass
        A new vector field instance representing the Lie bracket of *arg1* and *arg2*.

    Raises:
    -------
    Exception
        If either *arg1* or *arg2* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import VF_bracket, complexVarProc
    >>> complexVarProc('z','x','y',3)
    >>> print(VF_bracket(x2*D_x1 - x1*D_x2, x3*D_x1 - x1*D_x3))
    x3*D_x2 - x2*D_x3

    >>> print(VF_bracket(z2*D_z1 - z1*D_z2, z3*D_z1 - z1*D_z3))
    z3/2*D_x2 - z2/2*D_x3 - I*z3/2*D_y2 + I*z2/2*D_y3
    """
    if {arg1.__class__.__name__, arg2.__class__.__name__} == {"VFClass"}:
        # Determine the type (standard/complex) based on DGCVType attribute
        if arg1.DGCVType == "complex" or arg2.DGCVType == "complex":
            typeLoc = (
                "standard"
                if (arg1.DGCVType != "complex" or arg2.DGCVType != "complex")
                else "complex"
            )
            fast_handling = False
        else:
            typeLoc = "standard"
            fast_handling = True

        # align variable spaces
        varSpaceLoc = (
            arg1.varSpace
            if arg1.varSpace == arg2.varSpace
            else tuple(dict.fromkeys(arg1.varSpace + arg2.varSpace))
        )

        # Fast-handling: Avoid conversions if both are standard vector fields
        if fast_handling:
            coefLoc1 = VF_coeffs(arg1, varSpaceLoc)
            coefLoc2 = VF_coeffs(arg2, varSpaceLoc)
        elif fast_algorithm:
            # Convert both vector fields to real variables so that complex handling can be ignored later
            coefLoc1 = [symToReal(j) for j in VF_coeffs(arg1, varSpaceLoc)]
            coefLoc2 = [symToReal(j) for j in VF_coeffs(arg2, varSpaceLoc)]
        else:
            coefLoc1 = VF_coeffs(allToReal(arg1), varSpaceLoc)
            coefLoc2 = VF_coeffs(allToReal(arg2), varSpaceLoc)

        # Combine the coefficients and optionally simplify
        if doNotSimplify:
            result_coefs = [
                arg1(
                    coefLoc2[j], ignore_complex_handling=fast_handling or fast_algorithm
                )
                - arg2(
                    coefLoc1[j], ignore_complex_handling=fast_handling or fast_algorithm
                )
                for j in range(len(varSpaceLoc))
            ]
        else:
            result_coefs = [
                sp.simplify(
                    arg1(
                        coefLoc2[j],
                        ignore_complex_handling=fast_handling or fast_algorithm,
                    )
                    - arg2(
                        coefLoc1[j],
                        ignore_complex_handling=fast_handling or fast_algorithm,
                    )
                )
                for j in range(len(varSpaceLoc))
            ]

        return VFClass(varSpaceLoc, result_coefs, typeLoc)

    else:
        raise Exception("Expected arguments to be VFClass instances")


def contravariantVFTensorCoeffs(varSpace, *vector_fields):
    """
    Computes the contravariant tensor product of k vector fields with respect to the given variable space.

    This function evaluates the coefficients of the given vector fields in `vector_fields` on the specified
    `varSpace` and returns a list representing the tensor product of these coefficients. It is primarily
    used when evaluating a k-form on k vector fields.

    Parameters
    ----------
    varSpace : list or tuple
        A list or tuple of SymPy Symbol objects representing the variable space on which the vector fields are defined.

    *vector_fields : VFClass instances
        Any number of vector fields (instances of VFClass) that are evaluated in terms of the `varSpace`.

    Returns
    -------
    list
        A list of tuples, where each tuple contains:
        - A tuple of coordinate indices corresponding to the variables involved in each product term.
        - A SymPy expression representing the coefficient of the corresponding product term.

    Raises
    ------
    TypeError
        If any of the provided `vector_fields` is not an instance of `VFClass`.

    Examples
    --------
    >>> from DGCV import VFClass, contravariantVFTensorCoeffs, creatVariables
    >>> creatVariables('x1','x2')
    >>> vf1 = VFClass([x, y], [1, -y])
    >>> vf2 = VFClass([x, y], [y, 1])

    # Compute the contravariant tensor product
    >>> contravariantVFTensorCoeffs([x, y], vf1, vf2)
    [((0, 0), 1), ((0, 1), -y), ((1, 0), y), ((1, 1), -y**2)]
    """

    # Evaluate the coefficients of each vector field with respect to varSpace
    coeff_values = [
        VF_coeffs(vf, varSpace, sparse=True) for vf in vector_fields
    ]

    # Get the number of terms in each vector field's coefficient list
    term_counts = [len(coeff_list) for coeff_list in coeff_values]

    # Initialize product of coefficients with the first vector field
    product_coeffs = coeff_values[0]

    # Compute the tensor product for the remaining vector fields
    for jj in range(1, len(term_counts)):
        product_coeffs = [
            (k[0] + ll[0], k[1] * ll[1]) for k in product_coeffs for ll in coeff_values[jj]
        ]

    return product_coeffs


def changeDFBasis(arg1, arg2):
    newDFDataLoc = sparseKFormDataNewBasis(arg1.DFClassDataMinimal, arg1.varSpace, arg2)
    return DFClass(
        arg2,
        newDFDataLoc,
        arg1.degree,
        DGCVType=arg1.DGCVType,
        _simplifyKW=arg1._simplifyKW,
    )


def changeTFBasis(tensor_field, new_varSpace):
    """
    Change the basis of a tensorField to align with a new variable space.

    Parameters
    ----------
    tensor_field : tensorField
        The tensorField instance whose basis is to be changed.
    new_varSpace : list or tuple
        The new variable space to align the tensorField with.

    Returns
    -------
    tensorField
        A new tensorField instance with coefficients aligned to the new basis.
    """
    # Transform coeff_dict
    new_coeff_dict = _TFDictToNewBasis(tensor_field.coeff_dict, tensor_field.varSpace, new_varSpace)

    # Return a new tensorField instance with updated basis
    return tensorField(
        new_varSpace,
        new_coeff_dict,
        tensor_field.valence,
        data_shape=tensor_field.data_shape,DGCVType=tensor_field.DGCVType,_simplifyKW=tensor_field._simplifyKW
    )

def changeTensorFieldBasis(tensor_field, new_varSpace):
    """
    Converts a tensorField to a new basis while handling complex variable transformations.

    If `tensor_field` has `DGCVType='complex'`, `new_varSpace` is validated so that:
    - All variables belong to a complex variable system.
    - Variables are converted to match the tensor's target type (real/imaginary or hol/antihol).

    Parameters
    ----------
    tensor_field : tensorField
        The tensor field instance to be converted.
    new_varSpace : list or tuple
        The new variable space.

    Returns
    -------
    tensorField
        A new tensorField instance with coefficients aligned to the new basis.

    Warnings
    --------
    A warning is raised if `new_varSpace` contains variables that are not part of
    a recognized DGCV complex coordinate system.
    """

    new_type = tensor_field.DGCVType
    if new_type == 'complex':
        cd = get_variable_registry()['conversion_dictionaries']  # Retrieve conversion dictionaries

        # Determine the target type based on the first variable in tensor_field.varSpace
        first_var = tensor_field.varSpace[0]
        if first_var in cd['real_part']:
            target_type = "hol"
        elif first_var in cd['find_parents']:
            target_type = "real"
        else:
            raise KeyError(f"First variable {first_var} in tensor_field.varSpace is not part of complex variable system in DGCV's variable management framework.")

        # Validate that all variables in new_varSpace belong to DGCV's complex variable system
        if not all(var in cd['real_part'] or var in cd['find_parents'] for var in new_varSpace):
            warnings.warn(
                "`changeTensorFieldBasis` was given a `complex` type tensor field and `new_varSpace` "
                "containing variables that do not belong to complex coordinate systems, so a non-complex "
                "type tensor field was returned."
            )
            new_type = 'standard'
        else:
            # Build formatted_varSpace
            formatted_varSpace = list(tensor_field.varSpace)

            for var in new_varSpace:
                if target_type == "hol":
                    if var in cd['real_part']:  # var is holomorphic/antiholomorphic
                        formatted_varSpace.append(var)
                    elif var in cd['find_parents']:  # Convert real/imag to hol/antihol
                        hol_var, antihol_var = cd['find_parents'][var]  # Retrieve hol/antihol variables
                        if hol_var not in formatted_varSpace:
                            formatted_varSpace.append(hol_var)
                        if antihol_var not in formatted_varSpace:
                            formatted_varSpace.append(antihol_var)
                elif target_type == "real":
                    if var in cd['find_parents']:  # var is real/imag 
                        formatted_varSpace.append(var)
                    elif var in cd['real_part']:  # Convert hol/antihol to real/imag
                        real_var = cd['real_part'][var]  # Retrieve real part
                        imag_var = next(iter((cd['im_part'][var]).free_symbols))  # Retrieve imaginary part
                        if real_var not in formatted_varSpace:
                            formatted_varSpace.append(real_var)
                        if imag_var not in formatted_varSpace:
                            formatted_varSpace.append(imag_var)

            # Replace new_varSpace with the validated version
            new_varSpace = formatted_varSpace

    new_coeff_dict = _TFDictToNewBasis(tensor_field.expanded_coeff_dict, tensor_field.varSpace, new_varSpace)

    # Return a new tensorField instance with updated basis
    return tensorField(
        varSpace=tuple(new_varSpace),
        coeff_dict=new_coeff_dict,
        valence=tensor_field.valence,
        data_shape=tensor_field.data_shape,
        DGCVType=new_type,
        _simplifyKW=tensor_field._simplifyKW
    )

def _TFDictToNewBasis(data_dict, oldBasis, newBasis):
    """
    Transforms a tensorField's coefficient dictionary to a new variable space.

    Parameters
    ----------
    data_dict : dict
        The coefficient dictionary of a tensorField in its original basis.
    oldBasis : tuple
        The original variable space.
    newBasis : tuple
        The new variable space to which we want to align.

    Returns
    -------
    dict
        A transformed coefficient dictionary aligned to the new variable space.
    """
    data_list = list(data_dict.items())
    degree = len(data_list[0][0]) if data_list else 0

    try:
        new_data_dict = {
            tuple(newBasis.index(oldBasis[k]) for k in j[0]): j[1]
            for j in data_list if j[1] != 0
        }
    except ValueError as e:
        raise ValueError(
            f"`_TFDictToNewBasis` received bases where an element in oldBasis {oldBasis} does not exist in newBasis {newBasis}. "
            f"This issue arises because the sparse tensor data structure indicates this element is crucial in the tensor's definition: {e}"
        )

    if not new_data_dict:
        new_data_dict = {(0,) * degree: 0}

    return new_data_dict


def changeSTFBasis(arg1, arg2):
    newSTFDataLoc = sparseKFormDataNewBasis(
        arg1.STFClassDataMinimal, arg1.varSpace, arg2
    )
    return STFClass(
        arg2,
        newSTFDataLoc,
        arg1.degree,
        DGCVType=arg1.DGCVType,
        _simplifyKW=arg1._simplifyKW,
    )

def addTensorFields(*args, doNotSimplify=False):
    """
    Adds tensorField instances of the same type.
    """
    if len(args) == 0:
        return None  # No arguments
    if len(args) == 1:
        return args[0]  # Single tensorField, nothing to add

    # Check all arguments are tensorField instances
    if not all(isinstance(arg, tensorField) for arg in args):
        raise TypeError("addTensorFields expected all arguments to be tensorField instances.")

    # Check that all tensorFields have the same valence
    degrees = {tuple(arg.valence) for arg in args}
    if len(degrees) != 1:
        raise ValueError("Adding tensorFields with different valences is not supported.")

    # Determine `data_shape` for the resulting tensorField
    data_shapes = {arg.data_shape for arg in args}
    if len(data_shapes) == 1:
        result_data_shape = data_shapes.pop()  # Use the shared data_shape
        use_expanded_coeffs = False
    else:
        result_data_shape = "general"  # Default to general if data_shapes differ
        use_expanded_coeffs = True

    # Align variable spaces
    if len({tuple(arg.varSpace) for arg in args}) != 1:
        # Combine varSpace while preserving order
        varSpaceLoc = tuple(dict.fromkeys(sum([arg.varSpace for arg in args], ())))
        args = [changeTensorFieldBasis(arg, varSpaceLoc) for arg in args]
    else:
        varSpaceLoc = args[0].varSpace

    # Combine coefficients
    combined_coeffs = {}
    for tensor in args:
        coeffs_to_add = tensor.expanded_coeff_dict if use_expanded_coeffs else tensor.coeff_dict
        for key, value in coeffs_to_add.items():
            if key in combined_coeffs:
                if doNotSimplify:
                    combined_coeffs[key] += value
                else:
                    combined_coeffs[key] = sp.simplify(combined_coeffs[key] + value)
            else:
                combined_coeffs[key] = value

    # Create the resulting tensorField
    return tensorField(
        varSpaceLoc,
        combined_coeffs,
        args[0].valence,
        data_shape=result_data_shape,
        DGCVType=args[0].DGCVType,
        _simplifyKW=args[0]._simplifyKW,
    )

def scaleTensorField(arg1, arg2):
    """
    Multiplies the given tensor field (tensorField instance) by a scalar or function.
    """
    if isinstance(arg2,tensorField):
        return tensorField(
            arg2.varSpace,
            {a: arg1 * b for a, b in arg2.coeff_dict.items()},
            arg2.valence,
            arg2.DGCVType,
            arg2._simplifyKW
        )
    else:
        raise Exception(
            "scaleTensorField expected second positional argument to be a tensorField instance."
        )


def addDF(*args, doNotSimplify=False):
    """
    Adds the given differential k-forms (DFClass instances). All forms must have the same degree.

    This function adds any number of differential forms, returning a single differential form representing their sum.
    If the forms are defined on different variable spaces, the function will handle the change of basis and align the
    forms on a common variable space. The resulting form is returned as a DFClass instance.

    Parameters
    ----------
    *args : DFClass
        Any number of differential forms to be added. All forms must have the same degree.

    doNotSimplify : bool, optional
        If True, the result is returned without simplifying the coefficients (default is False).

    Returns
    -------
    DFClass
        A differential form (DFClass instance) representing the sum of the input forms.

    Raises
    ------
    Exception
        If the input forms have different degrees, or if the input arguments are not instances of DFClass.

    Examples
    --------
    >>> from sympy import symbols
    >>> from DGCV import DFClass, addDF
    >>> x, y, z = symbols('x y z')
    >>> df1 = DFClass([x, y], {(0,): 1, (1,): 2}, 1)
    >>> df2 = DFClass([x, y], {(0,): 3, (1,): 4}, 1)
    >>> df_sum = addDF(df1, df2)
    >>> df_sum
    DFClass([x, y], {(0,): 4, (1,): 6}, 1)

    # Adding forms with change of basis
    >>> df3 = DFClass([y, z], {(0,): 1, (1,): 2}, 1)
    >>> df_sum_with_basis_change = addDF(df1, df3)
    >>> df_sum_with_basis_change
    DFClass([x, y, z], {...}, 1)  # Aligned on common variable space
    """
    if len(args) == 0:
        return args
    if len(args) == 1:
        return args[0]
    if all([isinstance(j, DFClass) for j in args]):
        if len(set([j.degree for j in args])) == 1:
            typeList = list(set([j.DGCVType for j in args]))
            if len(typeList) == 1 and typeList[0] == "complex":
                typeLoc = "complex"
                if args[0]._varSpace_type == "real":
                    args = [
                        allToReal(df) if df._varSpace_type != "real" else df
                        for df in args
                    ]
                else:
                    args = [
                        allToSym(df) if df._varSpace_type != "complex" else df
                        for df in args
                    ]
            else:
                typeLoc = "standard"
                if len(typeList) != 1:
                    warnings.warn(
                        "Addition was performed between differential forms of `DGCVType` both `complex` and `standard`, so the resulting DFClass object has `DGCVType='standard'`, which disables complex variable handling. To preserve `DGCVType='complex'` make sure all DFClass objects in sum have `DGCVType='complex'`."
                    )

            # Fix here: Convert varSpace to tuples
            if len(set([tuple(j.varSpace) for j in args])) != 1:
                varSpaceLoc = list(dict.fromkeys(sum([j.varSpace for j in args], ())))
                args = [changeDFBasis(j, varSpaceLoc) for j in args]
            else:
                varSpaceLoc = args[0].varSpace

            coeffsLoc = dict()
            if doNotSimplify:
                for j in args:
                    dictLoc = j.DFClassDataDict
                    for a, b in dictLoc.items():
                        permS, orderedList = permSign(a, returnSorted=True)
                        orderedTuple = tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = (
                                coeffsLoc[orderedTuple] + permS * b
                            )
                        else:
                            coeffsLoc[orderedTuple] = permS * b
            else:
                for j in args:
                    dictLoc = j.DFClassDataDict
                    for a, b in dictLoc.items():
                        permS, orderedList = permSign(a, returnSorted=True)
                        orderedTuple = tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = sp.simplify(
                                coeffsLoc[orderedTuple] + permS * b
                            )
                        else:
                            coeffsLoc[orderedTuple] = permS * b

            return DFClass(tuple(varSpaceLoc), coeffsLoc, args[0].degree, typeLoc)
        else:
            raise Exception(
                "cannot add differential forms of different degrees. Suggestion: organize components of different degrees in a list that represents their sum and compute the addition with list comprehension."
            )
    else:
        raise Exception("addDF expected all arguments to be DFClass instances.")


def scaleDF(arg1, arg2):
    """
    Multiplies the given differential form (DFClass instance) by a scalar or function.

    This function scales a differential form `arg2` by a scalar or a symbolic expression `arg1`. The scaling is applied
    to each coefficient in the differential form, resulting in a new scaled differential form.

    Parameters
    ----------
    arg1 : Expr
        A SymPy expression representing the scalar or function to scale the differential form by.

    arg2 : DFClass
        A differential form (DFClass instance) that will be scaled by `arg1`.

    Returns
    -------
    DFClass
        A new differential form (DFClass instance) where each coefficient has been multiplied by `arg1`.

    Raises
    ------
    Exception
        If `arg2` is not an instance of DFClass.

    Examples
    --------
    >>> from DGCV import DFClass, scaleDF, creatVariables
    >>> creatVariables('x1','x2')
    >>> df = DFClass([x, y], {(0,): 1, (1,): 2}, 1)
    >>> scale = 3
    >>> scaled_df = scaleDF(scale, df)
    >>> scaled_df
    DFClass([x, y], {(0,): 3, (1,): 6}, 1)

    # Scaling by a symbolic function
    >>> f = x + y
    >>> scaled_df_func = scaleDF(f, df)
    >>> scaled_df_func
    DFClass([x, y], {(0,): x + y, (1,): 2*(x + y)}, 1)
    """
    if arg2.__class__.__name__ == "DFClass":
        return DFClass(
            arg2.varSpace,
            {a: arg1 * b for a, b in arg2.DFClassDataDict.items()},
            arg2.degree,
            arg2.DGCVType,
        )
    else:
        raise Exception(
            "scaleDF expected second positional argument to be a DFClass instance."
        )


def exteriorProduct(*args, doNotSimplify=False):
    """
    Computes the exterior (wedge) product of differential forms (DFClass instances).

    This function takes any number of DFClass instances and computes their exterior product. It works by first
    aligning the variable spaces of the forms, multiplying terms, and enforcing antisymmetry by using a permutation-based
    approach. The result is a new DFClass instance representing the product.

    Parameters
    ----------
    args : DFClass
        One or more differential forms (DFClass instances).

    doNotSimplify : bool, optional
        If True, the resulting form's coefficients will not be simplified (default is False).

    Returns
    -------
    DFClass
        The exterior product of the input forms, as a new DFClass instance.

    Raises
    ------
    Exception
        If any of the arguments are not instances of DFClass.

    Examples
    --------
    >>> from DGCV import DFClass, exteriorProduct, creatVariables
    >>> creatVariables('x1','x2')
    >>> df1 = DFClass([x], {0: 1}, 1)
    >>> df2 = DFClass([y], {0: 1}, 1)
    >>> exteriorProduct(df1, df2)
    DFClass([x, y], {(0, 1): 1}, 2)
    """
    # Check if all arguments are DFClass instances
    if not all(isinstance(j, DFClass) for j in args):
        raise Exception("Expected all arguments to be instances of DFClass")

    # Determine the type of the exterior product (standard or complex)
    typeList = {j.DGCVType for j in args}
    typeLoc = "complex" if len(typeList) == 1 and "complex" in typeList else "standard"
    if typeLoc == "complex":
        if args[0]._varSpace_type == "real":
            if any(j._varSpace_type != "real" for j in args[1:]):
                args = [allToReal(j) for j in args]
        else:
            if any(j._varSpace_type != "complex" for j in args[1:]):
                args = [allToSym(j) for j in args]

    # Efficient term combination for k-forms
    def kFormTermMultiplier(arg1, arg2, degr):
        result = dict()
        for j in arg1:
            for k in arg2:
                combined_indices = j[0] + k[0]
                if len(combined_indices) == len(
                    set(combined_indices)
                ):  # antisymmetry
                    permS, orderedList = permSign(combined_indices, returnSorted=True)
                    orderedTuple = tuple(orderedList)
                    if orderedTuple in result:
                        result[orderedTuple] = (
                            result[orderedTuple] + permS * j[1] * k[1]
                        )
                    else:
                        result[orderedTuple] = permS * j[1] * k[1]
        return result if result else {(0,) * degr: 0}

    # Compute the exterior product of two forms
    def exteriorProductOf2(arg1, arg2):
        varSpaceLoc = tuple(
            dict.fromkeys(arg1.varSpace + arg2.varSpace)
        )  # Combined variable space
        prodDegree = arg1.degree + arg2.degree

        # Align the variable spaces and get terms in the new basis
        newForm1Loc = changeDFBasis(arg1, varSpaceLoc).DFClassDataMinimal
        newForm2Loc = changeDFBasis(arg2, varSpaceLoc).DFClassDataMinimal

        # Multiply terms and enforce antisymmetry
        productTermsData = kFormTermMultiplier(newForm1Loc, newForm2Loc, prodDegree)

        # Create new DFClass instances for the product terms
        return DFClass(varSpaceLoc, productTermsData, prodDegree, typeLoc)

    # Handle multiple differential forms
    if len(args) > 1:
        resultLoc = args[0]
        for j in range(1, len(args)):
            resultLoc = exteriorProductOf2(resultLoc, args[j])
        return resultLoc
    elif len(args) == 1:
        return args[0]


def _TFDictToNewBasis(data_dict, oldBasis, newBasis):
    data_list = list(data_dict.items())
    degree = len(data_list[0][0])
    try:
        dataDict = dict(
            [
                (tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1])
                for j in data_list
                if j[1] != 0
            ]
        )
    except ValueError as e:
        raise ValueError(
            f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form's definition: {e}"
        )
    if not dataDict:
        dataDict = {(0,) * degree: 0}

    return dataDict


def sparseKFormDataNewBasis(sparseKFormData, oldBasis, newBasis):
    """
    Converts the indices of a k-form's sparse data representation from an old basis to a new basis.

    This function is primarily a helper for DGCV's exterior product calculus, allowing sparse data representing
    differential forms to be transformed into a new variable basis.

    Parameters
    ----------
    sparseKFormData : list of tuples
        A list where each tuple consists of the indices of variables in the old basis and the corresponding coefficient.

    oldBasis : list
        A list of variables representing the current basis.

    newBasis : list
        A list of variables representing the new basis, into which the k-form will be transformed.

    Returns
    -------
    list of tuples
        A transformed list where the indices are mapped from the old basis to the new basis, preserving the
        corresponding coefficients.

    Raises
    ------
    ValueError
        If a variable in the old basis is not found in the new basis.

    Examples
    --------
    >>> oldBasis = ['x', 'y', 'z']
    >>> newBasis = ['y', 'x', 'z']
    >>> sparseKFormData = [((0, 1), 1), ((1, 2), -1)]
    >>> sparseKFormDataNewBasis(sparseKFormData, oldBasis, newBasis)
    {(0, 2):−1, (1, 0):1}
    """
    if (
        not sparseKFormData
    ):  # Maybe safe to remove following October DFClassDataDict reformat!!!
        return {tuple(): 0}
    degree = len(sparseKFormData[0][0])
    try:
        dataDict = dict(
            [
                (tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1])
                for j in sparseKFormData
                if j[1] != 0
            ]
        )
    except ValueError as e:
        raise ValueError(
            f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form's definition: {e}"
        )
    if not dataDict:
        dataDict = {(0,) * degree: 0}
    return dataDict


############## tensor fields


def addSTF(*args, doNotSimplify=False):
    """
    Adds symmetric tensor fields (STFClass objects) of the same degree.
    """
    if len(args) == 0:
        return args
    if len(args) == 1:
        return args[0]
    if all([isinstance(j, STFClass) for j in args]):
        if len(set([j.degree for j in args])) == 1:
            typeList = list(set([j.DGCVType for j in args]))
            if len(typeList) == 1 and typeList[0] == "complex":
                typeLoc = "complex"
                if args[0]._varSpace_type == "real":
                    args = [
                        allToReal(stf) if stf._varSpace_type != "real" else stf
                        for stf in args
                    ]
                else:
                    args = [
                        allToSym(stf) if stf._varSpace_type != "complex" else stf
                        for stf in args
                    ]
            else:
                typeLoc = "standard"
                if len(typeList) != 1:
                    warnings.warn(
                        "Addition was performed between symmetric tensor fields of `DGCVType` both `complex` and `standard`, so the resulting STFClass object has `DGCVType='standard'`, which disables complex variable handling. To preserve `DGCVType='complex'`, use only STFClass objects in sum with `DGCVType='complex'`."
                    )

            # Fix here: Convert varSpace to tuples
            if len(set([tuple(j.varSpace) for j in args])) != 1:
                varSpaceLoc = list(dict.fromkeys(sum([j.varSpace for j in args], ())))
                args = [changeSTFBasis(j, varSpaceLoc) for j in args]
            else:
                varSpaceLoc = args[0].varSpace

            coeffsLoc = dict()
            if doNotSimplify:
                for j in args:
                    dictLoc = j.STFClassDataDict
                    for a, b in dictLoc.items():
                        _, orderedList = permSign(a, returnSorted=True)
                        orderedTuple = tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = coeffsLoc[orderedTuple] + b
                        else:
                            coeffsLoc[orderedTuple] = b
            else:
                for j in args:
                    dictLoc = j.STFClassDataDict
                    for a, b in dictLoc.items():
                        _, orderedList = permSign(a, returnSorted=True)
                        orderedTuple = tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = sp.simplify(
                                coeffsLoc[orderedTuple] + b
                            )
                        else:
                            coeffsLoc[orderedTuple] = b

            return STFClass(tuple(varSpaceLoc), coeffsLoc, args[0].degree, typeLoc)
        else:
            raise Exception(
                "Adding symmetric tensor fields of different degrees is not supported at this time. Suggestion: organize components of different degrees in a list that represents their sum and compute the addition with list comprehension."
            )
    else:
        raise Exception("addSTF expected all arguments to be STFClass instances.")


def scaleTF(arg1, arg2):
    """
    Scales tensor fields TFClass and STFClass alike.
    """
    if isinstance(arg2, STFClass):
        return STFClass(
            arg2.varSpace,
            {a: arg1 * b for a, b in arg2.STFClassDataDict.items()},
            arg2.degree,
            DGCVType=arg2.DGCVType,
            _simplifyKW=arg2._simplifyKW,
        )
    else:
        raise Exception(
            "scaleTF expected second positional argument to be a TFClass or STFClass instance."
        )

def tensor_product(*args, doNotSimplify=False):
    """
    Computes the tensor product of tensorField instances, after aligning the coordinate systems they
    are defined w.r.t.

    Parameters
    ----------
    args : tensorField
        One or more tensorField instances.

    doNotSimplify : bool, optional
        If True, the resulting form's coefficients will not be simplified (default is False).

    Returns
    -------
    tensorField
        The tensor product of the input tensor fields, as a new tensorField instance.

    Raises
    ------
    Exception
        If any of the arguments are not instances of tensorField.
    """
    # Check if all arguments are tensorField instances
    if not all(isinstance(arg, (tensorField,float,int,sp.Expr)) for arg in args):
        bad_types = []
        for arg in args:
            if not isinstance(arg, (tensorField,float,int,sp.Expr)):
                bad_types += [type(arg)]
        bad_types = list(set(bad_types))
        bt_str = ', '.join(bad_types)
        raise Exception(f"Expected all arguments to be instances of tensorField or scalar-like objects, not type: {bt_str}")
    non_scalars = [arg for arg in args if isinstance(arg,tensorField)]

    if len(set([tf.DGCVType for tf in non_scalars]))==1:
        target_type = non_scalars[0].DGCVType
    else:
        if len(non_scalars)>0:
            warnings.warn('`tensor_product` was applied to tensorField instances with different DGCVType attributes, so a `DGCVType = \'standard\'` tensorField was returned and variable formatting specific to any particular DGCVType was ignored.')
        target_type = 'standard'
    if target_type == 'complex':
        cd = get_variable_registry()['conversion_dictionaries']  # Retrieve conversion dictionaries
        # Determine the sub-target type based on the first variable in tensor_field.varSpace
        first_var = args[0].varSpace[0]
        if first_var in cd['real_part']:  # Means it is in holomorphic/antiholomorphic set
            args = [allToSym(arg) for arg in args]
        elif first_var in cd['find_parents']:  # Means it is in real/imaginary set
            args = [allToReal(arg) for arg in args]
        else:
            raise KeyError(f"First variable {first_var} in tensor_field.varSpace is not part of complex variable system in DGCV's variable management framework.")

    # Helper function to multiply terms
    def TermMultiplier(coeff_dict1, coeff_dict2):
        result = dict()
        for (key1, value1) in coeff_dict1.items():
            for (key2, value2) in coeff_dict2.items():
                combined_key = key1 + key2
                combined_value = value1 * value2
                result[combined_key] = combined_value
        return result if result else {(0,) * (len(coeff_dict1) + len(coeff_dict2)): 0}

    # Helper function to compute the tensor product of two tensorFields
    def tensor_productOf2(arg1, arg2):
        new_varSpace = tuple(dict.fromkeys(arg1.varSpace + arg2.varSpace))

        # Align the variable spaces and transform coefficients
        aligned_arg1 = changeTensorFieldBasis(arg1, new_varSpace)
        aligned_arg2 = changeTensorFieldBasis(arg2, new_varSpace)

        # Combine valences
        new_valence = aligned_arg1.valence + aligned_arg2.valence

        # Multiply coefficient dictionaries
        new_coeff_dict = TermMultiplier(aligned_arg1.expanded_coeff_dict, aligned_arg2.expanded_coeff_dict)

        # Return the new tensorField
        return tensorField(new_varSpace, new_coeff_dict, new_valence, data_shape='general',DGCVType=arg1,_simplifyKW=arg1._simplifyKW)

    # Handle multiple tensors
    if len(args) > 1:
        result = args[0]
        for next_tensor in args[1:]:
            if isinstance(next_tensor,(float,int,sp.Expr)):
                result = next_tensor * result
            else:
                result = tensor_productOf2(result, next_tensor)
        return result
    elif len(args) == 1:
        return args[0]
    else:
        raise ValueError("At least one tensorField is required.")


############## complex vector fields


def holVF_coeffs(arg1, arg2, doNotSimplify=False):
    """
    Evaluates the vector field (i.e., VFClass instance) *arg1* on each holomorphic variable in *arg2*,
    and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate
    vector fields.

    Parameters:
    -----------
    arg1 : VFClass
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    list
        A list of sympy expressions representing the coefficients in holomorphic coordinates.

    Raises:
    -------
    Exception
        If any variables in *arg2* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import complexVarProc, holVF_coeffs
    >>> complexVarProc('z','x','y',3)
    >>> holVF_coeffs(D_z1 + D_BARz2, [z1, z2, z3])
    [1, 0, 0]
    """
    if doNotSimplify:
        return [realToHol(arg1(j)) for j in arg2]
    else:
        return [sp.simplify(allToHol(arg1(j))) for j in arg2]


def antiholVF_coeffs(arg1, arg2, doNotSimplify=False):
    """
    Evaluates the vector field (i.e., VFClass instance) *arg1* on the conjugate of each holomorphic variable
    in *arg2*, and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate
    vector fields.

    Parameters:
    -----------
    arg1 : VFClass
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    list
        A list of sympy expressions representing the coefficients in antiholomorphic coordinates.

    Raises:
    -------
    Exception
        If any variables in *arg2* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import complexVarProc, antiholVF_coeffs
    >>> complexVarProc('z','x','y',3)
    >>> antiholVF_coeffs(D_z1 + D_BARz2, [z1, z2, z3])
    [0, 1, 0]
    """
    if doNotSimplify:
        return [realToHol(arg1(sp.conjugate(j))) for j in arg2]
    else:
        return [sp.simplify(realToHol(arg1(sp.conjugate(j)))) for j in arg2]


def complexVFC(arg1, arg2, doNotSimplify=False):
    """
    Evaluates the vector field (i.e., VFClass instance) *arg1* on the holomorphic variables in *arg2*
    and their complex conjugates, returning the result as two lists of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients for both the holomorphic and antiholomorphic parts of the vector field when expressed in
    terms of the respective coordinate vector fields.

    Parameters:
    -----------
    arg1 : VFClass
        A vector field instance to evaluate on the holomorphic and antiholomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    tuple of two lists
        The first list contains the coefficients of the holomorphic part, and the second list contains
        the coefficients of the antiholomorphic part.

    Raises:
    -------
    Exception
        If *arg1* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import complexVarProc, complexVFC
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> complexVFC(D_z1 + D_BARz2, [z1, z2, z3])
    ([1, 0, 0], [0, 1, 0])
    """
    if arg1.__class__.__name__ == "VFClass":
        # Use holVF_coeffs and antiholVF_coeffs, with optional simplification
        hol_coeffs = holVF_coeffs(arg1, arg2, doNotSimplify=doNotSimplify)
        antihol_coeffs = antiholVF_coeffs(arg1, arg2, doNotSimplify=doNotSimplify)
        return hol_coeffs, antihol_coeffs
    else:
        raise Exception("Expected first positional argument to be of type VFClass")


def conjComplex(arg):
    """
    Computes the complex conjugate of a complex vector field (i.e., VFClass instance).

    Parameters:
    -----------
    arg : VFClass
        A complex vector field instance whose complex conjugate is to be computed.

    Returns:
    --------
    VFClass
        A new vector field instance representing the complex conjugate of the input vector field.

    Raises:
    -------
    Exception
        If *arg* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import complexVarProc, conjComplexVF
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> vf = assembleFromCompVFC((z1, z2, 0), (BARz1, BARz2, 0), [z1, z2, z3])
    >>> conjComplexVF(vf)
    VFClass instance representing the complex conjugate of *vf*
    """
    warnings.warn("`conjComplex` has been depricated. Use conjugate_DGCV instead")
    return _conjComplexVFDF(arg)


def _conjComplexVFDF(arg):
    """
    Computes the complex conjugate of a complex vector field or differential form (i.e., VFClass/DFClass instance).

    Parameters:
    -----------
    arg : VFClass/DFClass
        A complex vector field instance whose complex conjugate is to be computed.

    Returns:
    --------
    VFClass/DFClass
        A new vector field instance representing the complex conjugate of the input vector field.

    Raises:
    -------
    Exception
        If *arg* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import complexVarProc, conjComplexVF
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> vf = assembleFromCompVFC((z1, z2, 0), (BARz1, BARz2, 0), [z1, z2, z3])
    >>> conjComplexVF(vf)
    VFClass instance representing the complex conjugate of *vf*
    """

    if isinstance(arg, VFClass):
        # Return the complex conjugate of the vector field
        if arg.DGCVType == "complex":
            if arg._varSpace_type == "complex":
                arg = allToReal(arg)
                return allToSym(
                    VFClass(
                        arg.varSpace,
                        [sp.conjugate(j) for j in arg.coeffs],
                        DGCVType="complex",
                        _simplifyKW=arg._simplifyKW,
                    )
                )
        return VFClass(
            arg.varSpace,
            [sp.conjugate(j) for j in arg.coeffs],
            DGCVType=arg.DGCVType,
            _simplifyKW=arg._simplifyKW,
        )
    elif isinstance(arg, DFClass):
        # Return the complex conjugate of the vector field
        if arg.DGCVType == "complex":
            if arg._varSpace_type == "complex":
                arg = allToReal(arg)
                return allToSym(
                    DFClass(
                        arg.varSpace,
                        {a: sp.conjugate(b) for a, b in arg.DFClassDataDict.items()},
                        arg.degree,
                        DGCVType="complex",
                        _simplifyKW=arg._simplifyKW,
                    )
                )
        return DFClass(
            arg.varSpace,
            {a: sp.conjugate(b) for a, b in arg.DFClassDataDict.items()},
            arg.degree,
            DGCVType=arg.DGCVType,
            _simplifyKW=arg._simplifyKW,
        )
    else:
        raise Exception("Expected the input to be of type VFClass or DFClass.")


def realPartOfVF(arg1, *args):
    """
    Computes the real part of a complex vector field (i.e., VFClass instance) *arg1* expressed in terms
    of complex variables in *arg2*.

    The real part is obtained by adding the vector field to its complex conjugate.

    Parameters:
    -----------
    arg1 : VFClass
        A complex vector field instance whose real part is to be computed.
    arg2 : list or tuple
        A list or tuple containing Symbol objects that were initialized as complex variables via complexVarProc.

    Returns:
    --------
    VFClass
        A new vector field instance representing the real part of the input complex vector field.

    Raises:
    -------
    Exception
        If *arg1* is not an instance of VFClass, or if variables in *arg2* were not initialized via complexVarProc.

    Example:
    --------
    >>> from DGCV import complexVarProc, realPartOfVF, assembleFromCompVFC
    >>> from sympy import simplify
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> vf = assembleFromCompVFC((z1, z2, 0), (BARz1, BARz2, 0), [z1, z2, z3])
    >>> print(simplify(realPartOfVF(vf, [z1, z2, z3]).simplify_format('real'))) # using simplify, it is easy to see that it is real!
    2*x1*D_x1 + 2*x2*D_x2 + 2*y1*D_y1 + 2*y2*D_y2
    """
    if isinstance(arg1, VFClass):
        # Return the real part of the vector field by adding it to its complex conjugate
        return addVF(arg1, _conjComplexVFDF(arg1))
    else:
        raise Exception("Expected the input to be of type VFClass.")


############## variable management
def listVar(
    standard_only=False,
    complex_only=False,
    algebras_only=False,
    zeroForms_only=False,
    coframes_only=False,
    temporary_only=False,
    obscure_only=False,
    protected_only=False,
):
    """
    This function lists all parent labels for objects tracked within the DGCV Variable Management Framework (VMF). In particular strings that are keys in DGCV's internal `standard_variable_systems`, `complex_variable_systems`, 'finite_algebra_systems', 'eps' dictionaries, etc. It also accepts optional keywords to filter the results, showing only temporary, protected, or "obscure" object system labels.

    Parameters
    ----------
    standard_only : bool, optional
        If True, only standard variable system labels will be listed.
    complex_only : bool, optional
        If True, only complex variable system labels will be listed.
    algebras_only : bool, optional
        If True, only finite algebra system labels will be listed.
    zeroForms_only : bool, optional
        If True, only zeroFormAtom system labels will be listed.
    coframes_only : bool, optional
        If True, only coframe system labels will be listed.
    temporary_only : bool, optional
        If True, only variable system labels marked as temporary will be listed.
    protected_only : bool, optional
        If True, only variable system labels marked as protected will be listed.

    Returns
    -------
    list
        A list of object system labels matching the provided filters.

    Notes
    -----
    - If no filters are specified, the function returns all labels tracked in the VMF.
    - If multiple filters are specified, the function combines them, displaying
      labels that meet any of the selected criteria.
    """
    variable_registry = get_variable_registry()

    # Collect all labels
    standard_labels = set(variable_registry["standard_variable_systems"].keys())
    complex_labels = set(variable_registry["complex_variable_systems"].keys())
    algebra_labels = set(variable_registry["finite_algebra_systems"].keys())
    zeroForm_labels = set(variable_registry["eds"]["atoms"].keys())  # New zeroFormAtom labels
    coframe_labels = set(variable_registry["eds"]["coframes"].keys())

    selected_labels = set()
    if standard_only:
        selected_labels |= standard_labels
    if complex_only:
        selected_labels |= complex_labels
    if algebras_only:
        selected_labels |= algebra_labels
    if zeroForms_only:
        selected_labels |= zeroForm_labels
    if coframes_only:
        selected_labels |= coframe_labels

    all_labels = selected_labels if selected_labels else standard_labels | complex_labels | algebra_labels | zeroForm_labels | coframe_labels

    # Apply additional property filters
    if temporary_only:
        all_labels = all_labels & variable_registry["temporary_variables"]
    if obscure_only:
        all_labels = all_labels & variable_registry["obscure_variables"]
    if protected_only:
        all_labels = all_labels & variable_registry["protected_variables"]

    # Return the filtered labels list
    return list(all_labels)

def _clearVar_single(label):
    """
    Helper function that clears a single variable system (standard, complex, or finite algebra)
    from the DGCV variable management framework. Instead of printing a report, it returns
    a tuple (system_type, label) indicating what was cleared.
    """
    registry = get_variable_registry()
    global_vars = _cached_caller_globals
    cleared_info = None

    # Handle standard variable systems
    if label in registry["standard_variable_systems"]:
        system_dict = registry["standard_variable_systems"][label]
        family_names = system_dict["family_names"]
        if isinstance(family_names, str):
            family_names = (family_names,)
        # Remove all family members from globals
        for var in family_names:
            if var in global_vars:
                del global_vars[var]
        if label in global_vars:
            del global_vars[label]
        # Remove differential objects if present
        if system_dict.get("differential_system", None):
            for var in family_names:
                if f"D_{var}" in global_vars:
                    del global_vars[f"D_{var}"]
                if f"d_{var}" in global_vars:
                    del global_vars[f"d_{var}"]
            if f"D_{label}" in global_vars:
                del global_vars[f"D_{label}"]
            if f"d_{label}" in global_vars:
                del global_vars[f"d_{label}"]
        # Remove temporary and obscure markers
        if system_dict.get("tempVar", None):
            registry["temporary_variables"].discard(label)
        if system_dict.get("obsVar", None):
            registry["obscure_variables"].discard(label)
        # Remove the system from the registry
        del registry["standard_variable_systems"][label]
        cleared_info = ("standard", label)

    # Handle complex variable systems
    elif label in registry["complex_variable_systems"]:
        system_dict = registry["complex_variable_systems"][label]
        family_houses = system_dict["family_houses"]

        # Remove real and imaginary parent labels from protected_variables
        real_parent = family_houses[-2]
        imag_parent = family_houses[-1]
        registry["protected_variables"].discard(real_parent)
        registry["protected_variables"].discard(imag_parent)

        # For tuple or multi_index systems, remove all parent labels from globals
        if system_dict["family_type"] in ("tuple", "multi_index"):
            for house in family_houses:
                if house in global_vars:
                    del global_vars[house]

        # Remove variables and associated DFClass/VFClass objects from globals
        variable_relatives = system_dict["variable_relatives"]
        for var_label, var_data in variable_relatives.items():
            if var_label in global_vars:
                del global_vars[var_label]
            if var_data.get("DFClass") and f"D_{var_label}" in global_vars:
                del global_vars[f"D_{var_label}"]
            if var_data.get("VFClass") and f"d_{var_label}" in global_vars:
                del global_vars[f"d_{var_label}"]

        # Clean up conversion dictionaries
        conversion_dictionaries = registry["conversion_dictionaries"]
        for var_label, var_data in variable_relatives.items():
            position = var_data.get("complex_positioning")
            value_pointer = var_data.get("variable_value")
            if position == "holomorphic":
                conversion_dictionaries["conjugation"].pop(value_pointer, None)
                conversion_dictionaries["holToReal"].pop(value_pointer, None)
                conversion_dictionaries["symToReal"].pop(value_pointer, None)
            elif position == "antiholomorphic":
                conversion_dictionaries["symToHol"].pop(value_pointer, None)
                conversion_dictionaries["symToReal"].pop(value_pointer, None)
            elif position in ["real", "imaginary"]:
                conversion_dictionaries["realToHol"].pop(value_pointer, None)
                conversion_dictionaries["realToSym"].pop(value_pointer, None)
                conversion_dictionaries["find_parents"].pop(value_pointer, None)

        # Remove temporary marker
        registry["temporary_variables"].discard(label)
        # Remove the system from the registry
        del registry["complex_variable_systems"][label]
        cleared_info = ("complex", label)

    # Handle finite algebra systems
    elif label in registry["finite_algebra_systems"]:
        family_names = registry["finite_algebra_systems"][label].get("family_names", {})
        for family_member in family_names:
            if family_member in global_vars:
                del global_vars[family_member]
        if label in global_vars:
            del global_vars[label]
        del registry["finite_algebra_systems"][label]
        cleared_info = ("algebra", label)

    # Handle zeroFormAtom systems
    elif label in registry["eds"]["atoms"]:
        system_dict = registry["eds"]["atoms"][label]

        # Remove family names from globals()
        family_names = system_dict["family_names"]
        if isinstance(family_names, str):
            family_names = (family_names,)
        for var in family_names:
            if var in global_vars:
                del global_vars[var]

        # Remove conjugates if they exist
        family_relatives = system_dict.get("family_relatives", {})
        for var in family_relatives:
            if var in global_vars:
                del global_vars[var]

        # Remove parent label from globals()
        if label in global_vars:
            del global_vars[label]

        # Remove from registry
        del registry["eds"]["atoms"][label]

        # for reporting
        cleared_info = ("DFAtom", label)

    # Handle coframes
    elif label in registry["eds"]["coframes"]:
        coframe_info = registry["eds"]["coframes"][label]
        cousins_parent = coframe_info.get("cousins_parent", None)

        # Remove coframe label from globals()
        if label in global_vars:
            del global_vars[label]

        # Remove from registry
        del registry["eds"]["coframes"][label]

        # Report coframe and cousins_parent separately
        cleared_info = ("coframe", (label, cousins_parent))

    return cleared_info

def clearVar(*labels, report=True):
    """
    Clears variables from the registry and global namespace. Because sometimes, we all need a fresh start.

    This function takes one or more variable system labels (strings) and clears all
    associated variables, vector fields, differential forms, and metadata from the
    DGCV system. Variable system refers to object systems created by the DGCV
    variable creation functions `variableProcedure`, `varWithVF`, and
    `complexVarProc`. Use `listVar()` to retriev a list of existed variable system
    labels. The function handles both standard and complex variable systems,
    ensuring that all related objects are removed from the caller's globals()
    namespace, `variable_registry`, and the conversion dictionaries.

    Parameters
    ----------
    *labels : str
        One or more string labels representing variable systems (either
        standard or complex). These labels will be removed along with all
        associated components.
    report : bool (optional)
        Set True to report about any variable systems cleared from the VMF

    Functionality
    -------------
    - For standard variable systems:
        1. All family members associated with the variable label will be
           removed from the caller's globals() namespace.
        2. If the variable system has associated differential forms (DFClass)
           or vector fields (VFClass), these objects will also be removed.
        3. The label will be removed from `temporary_variables`, if applicable.
        4. Finally, the label will be deleted from `standard_variable_systems`
           in `variable_registry`.

    - For complex variable systems:
        1. For each complex variable system:
            - Labels for the real and imaginary parts will be removed
              from `variable_registry['protected_variables']`.
            - If the system is a tuple, the parent labels for holomorphic,
              antiholomorphic, real, and imaginary variable tuples will be
              removed from the caller's globals() namespace.
            - The `variable_relatives` dictionary will be traversed to remove
              all associated variables, vector fields, and differential forms
              from the caller's globals() namespace.
            - The function will also clean up the corresponding entries in
              `conversion_dictionaries`, depending on the `complex_positioning`
              (holomorphic, antiholomorphic, real, or imaginary).
        2. The complex variable label will be removed from `temporary_variables`,
           if applicable.
        3. Finally, the label will be deleted from `complex_variable_systems`
           in `variable_registry`.

    Notes
    -----
    - Comprehensively clears variables and their associated metadata from the DGCV
      system.
    - Use with `listVar` to expediantly clear everything, e.g., `clearVar(*listVar())`.

    Examples
    --------
    >>> clearVar('x') # removes any DGCV variable system labeled as x, such as
                      # (x, D_x, d_x), (x=(x1, x2), x1, x2, D_x1, d_x1,...), etc.
    >>> clearVar('z', 'y', 'w')

    This will remove all variables, vector fields, and differential forms
    associated with the labels 'z', 'y', and 'w'.

    """
    cleared_standard = []
    cleared_complex = []
    cleared_algebras = []
    cleared_diffFormAtoms = []
    cleared_coframes = []

    for label in labels:
        info = _clearVar_single(label)
        if info:
            system_type, cleared_label = info
            if system_type == "standard":
                cleared_standard.append(cleared_label)
            elif system_type == "complex":
                cleared_complex.append(cleared_label)
            elif system_type == "algebra":
                cleared_algebras.append(cleared_label)
            elif system_type == "DFAtom":
                cleared_diffFormAtoms.append(cleared_label)
            elif system_type == "coframe":
                coframe_label, cousins_parent_label = cleared_label
                cleared_coframes.append((coframe_label, cousins_parent_label))
                clearVar(cousins_parent_label, report=False)

    if report:
        if cleared_standard:
            print(
                f"Cleared standard variable systems from the DGCV variable management framework: {', '.join(cleared_standard)}"
            )
        if cleared_complex:
            print(
                f"Cleared complex variable systems from the DGCV variable management framework: {', '.join(cleared_complex)}"
            )
        if cleared_algebras:
            print(
                f"Cleared finite algebra systems from the DGCV variable management framework: {', '.join(cleared_algebras)}"
            )
        if cleared_diffFormAtoms:
            print(
                f"Cleared differential form systems from the DGCV variable management framework: {', '.join(cleared_diffFormAtoms)}"
            )
        if cleared_coframes:
            for cf_label, cp_label in cleared_coframes:
                print(f"Cleared coframe '{cf_label}' along with associated zero form atom system '{cp_label}'")

def DGCV_snapshot(style="default", use_latex=False, complete_report = None):
    """
    Generate a comprehensive snapshot of DGCV's variable management framework (VMF), including variables,
    algebras, coordinate systems, vector fields, and differential forms.

    Parameters
    ----------
    style : str, optional
        The style options to apply to the summary table. Default theme is 'chalkboard_green'. Use the DGCV function `get_DGCV_themes()` to display a list of other available themes.
    use_latex : bool, optional
        If True, the table will format text using LaTeX for better
        mathematical display. Default is False.

    Returns
    -------
    DataFrame
        A formatted snapshot displaying the initialized variables, algebras, coordinate systems,
        vector fields, and differential forms.
    """
    if complete_report is True:
        force_report = True
    else:
        force_report = False
    if complete_report is None:
        complete_report = True

    vr = get_variable_registry()
    outputs = []

    if vr['standard_variable_systems'] or vr['complex_variable_systems'] or force_report:
        outputs.append(_snapshot_coor_(style=style, use_latex=use_latex))
    if vr['finite_algebra_systems'] or force_report:
        outputs.append(_snapshot_algebras_(style=style, use_latex=use_latex))
    if vr['eds']['atoms'] or force_report:
        outputs.append(_snapshot_eds_atoms_(style=style, use_latex=use_latex))
    if vr['eds']['coframes'] or force_report:
        outputs.append(_snapshot_coframes_(style=style, use_latex=use_latex))

    if not outputs and not force_report:
        print("There are no objects currently registered in the DGCV VMF.")
    else:
        for table in outputs:
            display(table)



def variableSummary(*args, **kwargs):
    warnings.warn(
        "variableSummary() is deprecated and will be removed in a future version. "
        "Please use DGCV_snapshot() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DGCV_snapshot(*args, **kwargs)

def _snapshot_coor_(style="default", use_latex=False):
    """
    Returns a Pandas DataFrame summarizing the coordinate systems in DGCV.

    This snapshot includes both complex and standard variable systems.
    For each system, the summary includes:
      - The number of variables (tuple length)
      - The initial index
      - For complex systems, the labels used for the real and imaginary parts,
        as well as formatted representations of the vector fields and differential forms.
      - For standard systems, the corresponding formatted vector fields and differential forms.

    Parameters:
        use_latex : bool, optional
            If True, formatting is applied using LaTeX.
        style : str, optional
            A style theme to apply to the summary table.

    Returns:
        A styled Pandas DataFrame summarizing the coordinate systems and finite algebras.
    """
    def wrap_in_dollars(content):
        return f"${content}$"

    def convert_to_greek(var_name):
        # Assume greek_letters is defined module-wide; its values are LaTeX-valid strings.
        for name, greek in greek_letters.items():
            if var_name.lower().startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def format_variable_name(var_name, system_type, use_latex=False):
        # Look up system info from the registry.
        if system_type == "standard":
            info = variable_registry["standard_variable_systems"].get(var_name, {})
            family_names = info.get("family_names", var_name)
            initial_index = info.get("initial_index", 1)
        elif system_type == "complex":
            info = variable_registry["complex_variable_systems"].get(var_name, {})
            # For complex systems, assume family_names is a 4-tuple; we use its first element (holomorphic names)
            family_names = info.get("family_names", ())
            if family_names and isinstance(family_names, (list, tuple)) and len(family_names) > 0:
                family_names = family_names[0]
            else:
                family_names = var_name
            initial_index = info.get("initial_index", 1)
        elif system_type == "algebra":
            info = variable_registry["finite_algebra_systems"].get(var_name, {})
            family_names = info.get("family_names", var_name)
            initial_index = None
        else:
            family_names, initial_index = var_name, None

        # If family_names is a sequence with more than one element, format with an ellipsis.
        if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
            if use_latex:
                if initial_index:
                    content = (f"{convert_to_greek(var_name)} = \\left( {convert_to_greek(var_name)}_{{{initial_index}}}, "
                            f"\\ldots, {convert_to_greek(var_name)}_{{{initial_index + len(family_names) - 1}}} \\right)")
                else:
                    content = (f"{convert_to_greek(var_name)} = {convert_to_greek(var_name)}")
            else:
                content = f"{var_name} = ({family_names[0]}, ..., {family_names[-1]})"
        else:
            content = f"{var_name}"
        if use_latex:
            content = wrap_in_dollars(content)
        return content

    def build_object_string(obj_type, var_name, start_index, tuple_len, system_type, use_latex=False):
        # For standard systems, if there's only one variable, return a simple label;
        # otherwise, show the range of indices.
        if tuple_len == 1:
            content = f"{obj_type}_{var_name}"
        else:
            content = f"{obj_type}_{var_name}{start_index},...,{obj_type}_{var_name}{start_index + tuple_len - 1}"
        if use_latex:
            content = wrap_in_dollars(content)
        return content

    def build_object_string_for_complex(obj_type, family_houses, family_names, start_index, use_latex=False):
        """
        For a complex variable system, family_houses is expected to be a 4-tuple of labels
        (e.g. (hol, anti, real, imag)) and family_names is a 4-tuple of sequences, one for each part.
        This function formats each part (only if the corresponding list has more than one element)
        using an ellipsis; otherwise, it shows the single variable.
        """
        parts = []
        # Expect family_names to be a 4-tuple: (hol_names, anti_names, real_names, imag_names)
        if isinstance(family_names, (list, tuple)) and len(family_names) == 4:
            for i, part in enumerate(family_houses):
                part_names = family_names[i]  # list of names for this part
                if isinstance(part_names, (list, tuple)) and len(part_names) > 1:
                    if use_latex:
                        part_str = f"\\frac{{\\partial}}{{\\partial {convert_to_greek(part)}_{{{start_index}}}}}, \\ldots, \\frac{{\\partial}}{{\\partial {convert_to_greek(part)}_{{{start_index + len(part_names) - 1}}}}}" if obj_type=="D" else f"d {convert_to_greek(part)}_{{{start_index}}}, \\ldots, d {convert_to_greek(part)}_{{{start_index + len(part_names) - 1}}}"
                        part_str = wrap_in_dollars(part_str)
                    else:
                        part_str = f"{obj_type}_{part}{start_index},...,{obj_type}_{part}{start_index + len(part_names) - 1}"
                else:
                    if use_latex:
                        part_str = wrap_in_dollars(f"{obj_type}_{convert_to_greek(part)}")
                    else:
                        part_str = f"{obj_type}_{part}"
                parts.append(part_str)
        else:
            # Fallback: if family_names is not the expected 4-tuple, treat as a single list.
            if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
                if use_latex:
                    part_str = wrap_in_dollars(f"{obj_type}_{convert_to_greek(family_houses[0])}{start_index},...,{obj_type}_{convert_to_greek(family_houses[0])}{start_index + len(family_names) - 1}")
                else:
                    part_str = f"{obj_type}_{family_houses[0]}{start_index},...,{obj_type}_{family_houses[0]}{start_index + len(family_names) - 1}"
                parts.append(part_str)
            else:
                if use_latex:
                    parts.append(wrap_in_dollars(f"{obj_type}_{convert_to_greek(family_houses[0])}"))
                else:
                    parts.append(f"{obj_type}_{family_houses[0]}")
        return ", ".join(parts)

    variable_registry = get_variable_registry()
    data = []          # Each row is: [# Variables, Initial Index, Real Part, Imaginary Part, Vector Fields, Differential Forms]
    var_system_labels = []  # Regular column for variable system labels

    # Process complex variable systems
    for var_name in sorted(variable_registry.get("complex_variable_systems", {}).keys()):
        system = variable_registry["complex_variable_systems"][var_name]
        # For complex systems, family_names should be a 4-tuple.
        fn = system.get("family_names", ())
        if fn and isinstance(fn, (list, tuple)) and len(fn) == 4:
            hol_names = fn[0]
        else:
            hol_names = []
        tuple_len = len(hol_names) if isinstance(hol_names, (list, tuple)) else 1
        start_index = system.get("initial_index", 1)
        formatted_label = format_variable_name(var_name, "complex", use_latex=use_latex)
        var_system_labels.append(formatted_label)
        family_houses = system.get("family_houses", ("N/A", "N/A", "N/A", "N/A"))
        # For complex systems, display real and imaginary parts from the 3rd and 4th entries.
        if isinstance(fn, (list, tuple)) and len(fn)==4:
            real_names = fn[2]
            imag_names = fn[3]
        else:
            real_names, imag_names = "N/A", "N/A"
        if use_latex:
            real_part = wrap_in_dollars(f"{convert_to_greek(family_houses[2])} = \\left( {convert_to_greek(family_houses[2])}_{{{start_index}}}, \\ldots, {convert_to_greek(family_houses[2])}_{{{start_index + len(real_names) - 1}}} \\right)") if isinstance(real_names, (list, tuple)) and len(real_names)>1 else wrap_in_dollars(f"{convert_to_greek(family_houses[2])}")
            imag_part = wrap_in_dollars(f"{convert_to_greek(family_houses[3])} = \\left( {convert_to_greek(family_houses[3])}_{{{start_index}}}, \\ldots, {convert_to_greek(family_houses[3])}_{{{start_index + len(imag_names) - 1}}} \\right)") if isinstance(imag_names, (list, tuple)) and len(imag_names)>1 else wrap_in_dollars(f"{convert_to_greek(family_houses[3])}")
        else:
            real_part = f"{family_houses[2]} = ({real_names[0]}, ..., {real_names[-1]})" if isinstance(real_names, (list, tuple)) and len(real_names)>1 else f"{family_houses[2]}"
            imag_part = f"{family_houses[3]} = ({imag_names[0]}, ..., {imag_names[-1]})" if isinstance(imag_names, (list, tuple)) and len(imag_names)>1 else f"{family_houses[3]}"
        vf_str = build_object_string_for_complex("D", family_houses, fn, start_index, use_latex)
        df_str = build_object_string_for_complex("d", family_houses, fn, start_index, use_latex)
        data.append([tuple_len, real_part, imag_part, vf_str, df_str])

    # Process standard variable systems
    for var_name in sorted(variable_registry.get("standard_variable_systems", {}).keys()):
        system = variable_registry["standard_variable_systems"][var_name]
        family_names = system.get("family_names", ())
        tuple_len = len(family_names) if isinstance(family_names, (list, tuple)) else 1
        start_index = system.get("initial_index", 1)
        formatted_label = format_variable_name(var_name, "standard", use_latex=use_latex)
        var_system_labels.append(formatted_label)
        vf_str = build_object_string("D", var_name, start_index, tuple_len, "standard", use_latex)
        df_str = build_object_string("d", var_name, start_index, tuple_len, "standard", use_latex)
        data.append([tuple_len, "----", "----", vf_str, df_str])

    # Process finite algebra systems
    finite_algebra_vars = sorted(variable_registry.get("finite_algebra_systems", {}).keys())
    algebra_labels = []
    for var_name in finite_algebra_vars:
        system = variable_registry["finite_algebra_systems"][var_name]
        family_names = system.get("family_names", ())
        if isinstance(family_names, (list, tuple)):
            num_basis = len(family_names)
            basis_str = f"{family_names[0]}, ..., {family_names[-1]}" if num_basis > 5 else ", ".join(str(x) for x in family_names)
        else:
            num_basis = 1
            basis_str = str(family_names)
        formatted_label = format_variable_name(var_name, "algebra", use_latex=use_latex)
        algebra_labels.append(formatted_label)
        # For finite algebras, we place placeholders for other columns.
        data.append([num_basis, "----", "----", "----", "----"])

    # Combine labels from coordinate systems and algebras.
    all_labels = var_system_labels + algebra_labels
    # Prepend the label to each data row.
    combined_data = []
    for label, row in zip(all_labels, data):
        combined_data.append([label] + row)

    columns = ["Coordinate System", "# of Variables", "Real Part", "Imaginary Part", "Vector Fields", "Differential Forms"]
    table = DataFrame(combined_data, columns=columns)
    table_header = "Initialized Coordinate Systems"
    table_styles = get_style(style)+[{'selector': 'td', 'props': [('text-align', 'left')]}]+[{'selector': 'th', 'props': [('text-align', 'left')]}]
    styled_table = (
        table.style
        .set_table_styles(table_styles)
        .set_caption(f"{table_header}")
        .hide(axis="index")
    )
    return styled_table

def _snapshot_algebras_(style="default", use_latex=False):
    """
    Returns a Pandas DataFrame summarizing the finite algebra systems in DGCV's VMF.

    For each finite algebra system, the snapshot includes:
      - The formatted algebra label.
      - The family type.
      - The basis (family_names), formatted as a comma‑separated string (with ellipsis if there are too many elements).
      - The number of basis elements.

    Args:
        use_latex (bool, optional): If True, formats labels using LaTeX.
        style (str, optional): A style theme name to retrieve table styles.

    Returns:
        A styled Pandas DataFrame.
    """

    # Helper: wrap content in dollars for LaTeX
    def wrap_in_dollars(content):
        return f"${content}$"

    # Helper: convert variable name to Greek if applicable.
    def convert_to_greek(var_name):
        # Assume greek_letters is defined module-wide as a dict mapping strings to LaTeX-valid Greek strings.
        # Example: {'alpha': '\\alpha', 'beta': '\\beta', ...}
        for name, greek in greek_letters.items():
            if var_name.lower().startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    # Helper: process a basis label into a LaTeX string.
    def process_basis_label(label):
        # Use regex to separate alphabetic and numeric parts.
        match = re.match(r"(.*?)(\d+)?$", label)
        if match:
            basis_elem_name = match.group(1).rstrip("_")
            basis_elem_number = match.group(2)
            # Convert the alphabetic part to Greek if applicable.
            basis_elem_name = convert_to_greek(basis_elem_name)
            if basis_elem_number:
                return f"{basis_elem_name}_{{{basis_elem_number}}}"
            else:
                return f"{basis_elem_name}"
        else:
            return label

    # Retrieve the variable registry.
    registry = get_variable_registry()
    data = []

    finite_algebras = registry.get("finite_algebra_systems", {})
    for label in sorted(finite_algebras.keys()):
        system = finite_algebras[label]
        family_names = system.get("family_names", ())
        if isinstance(family_names, (list, tuple)):
            num_basis = len(family_names)
            if num_basis > 5:
                basis_str = f"{process_basis_label(family_names[0])}, ..., {process_basis_label(family_names[-1])}"
            else:
                basis_str = ", ".join(process_basis_label(x) for x in family_names)
        else:
            num_basis = 1
            basis_str = process_basis_label(family_names)
        # For the algebra label, use the module-wide format_variable_name with system_type "algebra".
        # Here, we simply assume the formatted label is the label itself wrapped in LaTeX if needed.
        if use_latex:
            formatted_label = wrap_in_dollars(convert_to_greek(label))
        else:
            formatted_label = label

        data.append([formatted_label, basis_str, num_basis])

    columns = ["Algebra Label", "Basis", "Dimension"]
    df = pd.DataFrame(data, columns=columns)

    # Apply styling using get_style().
    table_styles = get_style(style)+[{'selector': 'td', 'props': [('text-align', 'left')]}]+[{'selector': 'th', 'props': [('text-align', 'left')]}]
    styled_df = df.style.set_table_styles(table_styles)
    styled_df = (
        df.style
        .set_table_styles(table_styles)
        .set_caption("Initialized Finite Algebra Systems")
        .hide(axis="index")
    )
    return styled_df

def _snapshot_eds_atoms_(style="default", use_latex=False):
    """
    Returns a summary table listing abstract differential form atoms in the VMF scope

    Parameters:
    ----------
    style : str, optional
        A style theme to apply to the summary table.
    use_latex : bool, optional
        If True, formats text using LaTeX

    Returns:
    -------
    pandas.DataFrame
        A summary table listing abstract differential form atoms in the VMF scope
    """
    variable_registry = get_variable_registry()
    eds_atoms_registry = variable_registry["eds"]["atoms"]

    if not eds_atoms_registry:
        return pd.DataFrame(columns=["DF System", "Degree", "# Elements", "Differential Forms", "Conjugate Forms", "Primary Coframe"])

    data = []
    for label, system in sorted(eds_atoms_registry.items()):
        df_system = label
        degree = system.get("degree", "----")
        family_values = system.get("family_values", ())
        num_elements = len(family_values) if isinstance(family_values, tuple) else 1

        # Format Differential Forms
        if isinstance(family_values, tuple):
            if len(family_values) > 3:
                diff_forms = f"{family_values[0]}, ..., {family_values[-1]}"
            else:
                diff_forms = ", ".join(str(x) for x in family_values)
        else:
            diff_forms = str(family_values)

        # Apply LaTeX formatting for Differential Forms if required
        if use_latex and family_values:
            if isinstance(family_values, tuple) and len(family_values) > 3:
                diff_forms = f"$ {family_values[0]._latex()}, ..., {family_values[-1]._latex()} $"
            elif isinstance(family_values, tuple):
                diff_forms = ", ".join(f"{x._latex()}" if hasattr(x, "_latex") else str(x) for x in family_values)
                diff_forms = f"$ {diff_forms} $"

        # Format Conjugate Forms
        real_status = system.get("real", False)
        if real_status:
            conjugate_forms = "----"
        else:
            conjugates = system.get("conjugates", {})
            if conjugates:
                conjugate_list = list(conjugates.values())
                if len(conjugate_list) > 3:
                    conjugate_forms = f"{conjugate_list[0]}, ..., {conjugate_list[-1]}"
                else:
                    conjugate_forms = ", ".join(str(x) for x in conjugate_list)
                if use_latex and conjugate_list and hasattr(conjugate_list[0], "_latex"):
                    if len(conjugate_list) > 3:
                        conjugate_forms = f"$ {conjugate_list[0]._latex()}, ..., {conjugate_list[-1]._latex()} $"
                    else:
                        conjugate_forms = ", ".join(x._latex() for x in conjugate_list)
                        conjugate_forms = f"$ {conjugate_forms} $"
            else:
                conjugate_forms = "----"

        # Format Primary Coframe
        primary_coframe = system.get("primary_coframe", None)
        if primary_coframe is None:
            primary_coframe_str = "----"
        else:
            primary_coframe_str = primary_coframe._latex() if use_latex and hasattr(primary_coframe, "_latex") else repr(primary_coframe)

        data.append([df_system, degree, num_elements, diff_forms, conjugate_forms, primary_coframe_str])

    columns = ["DF System", "Degree", "# Elements", "Differential Forms", "Conjugate Forms", "Primary Coframe"]
    df = pd.DataFrame(data, columns=columns)

    table_styles = get_style(style) + [
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ]
    styled_df = df.style.set_table_styles(table_styles).set_caption("Initialized abstract differential forms in the VMF scope").hide(axis="index")

    return styled_df



def _snapshot_coframes_(style="default", use_latex=False):
    """
    Returns a summary table listing coframes in the VMF scope

    Parameters:
    ----------
    style : str, optional
        A style theme to apply to the summary table.
    use_latex : bool, optional
        If True, formats text using LaTeX

    Returns:
    -------
    pandas.DataFrame
    Returns a summary table listing coframes in the VMF scope
    """

    variable_registry = get_variable_registry()
    coframes_registry = variable_registry["eds"].get("coframes", {})

    if not coframes_registry:
        return pd.DataFrame(columns=["Coframe Label", "Coframe 1-Forms", "Structure Coefficients"])

    data = []
    for label, system in sorted(coframes_registry.items()):
        # Retrieve coframe object
        coframe_obj = _cached_caller_globals.get(label, label)
        coframe_label = label

        # Process Coframe 1-Forms
        if isinstance(coframe_obj,str):
            children = system.get("children", [])
            if len(children)>0:
                if any(child not in _cached_caller_globals for child in children):
                    children = []
                else:
                    children = [_cached_caller_globals[child] for child in children]
        else:
            children = coframe_obj.forms
        if len(children) > 3:
            coframe_1_forms = f"{children[0]._latex()}, ..., {children[-1]._latex()}"
        else:
            coframe_1_forms = ", ".join(
                child._latex() if use_latex and hasattr(child, "_latex") else repr(child)
                for child in children
            )
        coframe_1_forms = f"$ {coframe_1_forms} $" if use_latex else coframe_1_forms

        # Process Structure Coefficients
        cousins = system.get("cousins_vals", [])
        if len(cousins) > 3:
            structure_coeffs = f"{cousins[0]._latex()}, ..., {cousins[-1]._latex()}"
        else:
            structure_coeffs = ", ".join(
                cousin._latex() if use_latex and hasattr(cousin, "_latex") else repr(cousin)
                for cousin in cousins
            )
        structure_coeffs = f"$ {structure_coeffs} $" if use_latex else structure_coeffs

        data.append([coframe_label, coframe_1_forms, structure_coeffs])

    columns = ["Coframe Label", "Coframe 1-Forms", "Structure Coefficients"]
    df = pd.DataFrame(data, columns=columns)

    # Apply styling
    table_styles = get_style(style) + [
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ]
    styled_df = df.style.set_table_styles(table_styles).set_caption("Initialized Abstract Coframes").hide(axis="index")

    return styled_df
