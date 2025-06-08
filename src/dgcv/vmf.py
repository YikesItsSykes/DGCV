"""
dgcv: Differential Geometry with Complex Variables

This module defines defines functions for interacting with the Variable Management Framework (VMF), 
which is dgcv's system for managing object creation and labeling. It additionaly contains functions
for internal use by the VMF. The intended public functions include the following

Functions for listing and clearing objects in the VMF:
    - listVar(): Lists the "parent names" of objects currently tracked by the dgcv VMF.
    - clearVar(): Clears the variables from the dgcv registry and deletes them from caller's globals().

Functions for summarizing the state of the VMF:
    - vmf_summary(): Takes a snapshot of the current dgcv VMF and reports a summary in a
    Pandas table.

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy
    - pandas

License:
    MIT License
"""

import re
import warnings

import pandas as pd
import sympy as sp
from IPython.display import display
from sympy import I

from ._config import (
    _cached_caller_globals,
    get_dgcv_settings_registry,
    get_variable_registry,
    greek_letters,
)
from .combinatorics import carProd_with_weights_without_R, permSign
from .conversions import _allToSym
from .styles import get_style


def _coeff_dict_formatter(
    varSpace,coeff_dict,valence,total_degree,_varSpace_type,data_shape
):
    """
    Helper function to populate conversion dicts for tensor field classes
    """
    variable_registry = get_variable_registry()
    CVS = variable_registry["complex_variable_systems"]

    exhaust1 = list(varSpace)
    populate = {
        "compCoeffDataDict": dict(),
        "realCoeffDataDict": dict(),
        "holVarDict": dict(),
        "antiholVarDict": dict(),
        "realVarDict": dict(),
        "imVarDict": dict(),
        "preProcessMinDataToHol": dict(),
        "preProcessMinDataToReal": dict(),
    }
    if _varSpace_type == "real":
        for var in varSpace:
            varStr = str(var)
            if var in exhaust1:
                for parent in CVS.values():
                    if varStr in parent["variable_relatives"]:
                        cousin = (
                            set(
                                parent["variable_relatives"][varStr][
                                    "complex_family"
                                ][2:]
                            )
                            - {var}
                        ).pop()
                        if cousin in exhaust1:
                            exhaust1.remove(cousin)
                        if (
                            parent["variable_relatives"][varStr][
                                "complex_positioning"
                            ]
                            == "real"
                        ):
                            realVar = var
                            exhaust1.remove(var)
                            imVar = cousin
                        else:
                            realVar = cousin
                            exhaust1.remove(var)
                            imVar = var
                        holVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][0]
                        antiholVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][1]
                        populate["holVarDict"][holVar] = [realVar, imVar]
                        populate["antiholVarDict"][antiholVar] = [
                            realVar,
                            imVar,
                        ]
                        populate["realVarDict"][realVar] = [holVar, antiholVar]
                        populate["imVarDict"][imVar] = [holVar, antiholVar]
    else:  # _varSpace_type == 'complex'
        for var in varSpace:
            varStr = str(var)
            if var in exhaust1:
                for parent in CVS.values():
                    if varStr in parent["variable_relatives"]:
                        cousin = (
                            set(
                                parent["variable_relatives"][varStr][
                                    "complex_family"
                                ][:2]
                            )
                            - {var}
                        ).pop()
                        if cousin in exhaust1:
                            exhaust1.remove(cousin)
                        if (
                            parent["variable_relatives"][varStr][
                                "complex_positioning"
                            ]
                            == "holomorphic"
                        ):
                            holVar = var
                            exhaust1.remove(var)
                            antiholVar = cousin
                        else:
                            holVar = cousin
                            exhaust1.remove(var)
                            antiholVar = var
                        realVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][2]
                        imVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][3]
                        populate["holVarDict"][holVar] = [realVar, imVar]
                        populate["antiholVarDict"][antiholVar] = [
                            realVar,
                            imVar,
                        ]
                        populate["realVarDict"][realVar] = [holVar, antiholVar]
                        populate["imVarDict"][imVar] = [holVar, antiholVar]
    new_realVarSpace = tuple(populate["realVarDict"].keys())
    new_holVarSpace = tuple(populate["holVarDict"].keys())
    new_antiholVarSpace = tuple(populate["antiholVarDict"].keys())
    new_imVarSpace = tuple(populate["imVarDict"].keys())

    if len(valence) == 0:
        if _varSpace_type == "real":
            populate["realCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["compCoeffDataDict"] = [
                new_holVarSpace + new_antiholVarSpace,
                {(0,) * total_degree: coeff_dict[(0,) * total_degree]},
            ]
        else:
            populate["compCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["realCoeffDataDict"] = [
                new_realVarSpace + new_imVarSpace,
                {(0,) * total_degree: coeff_dict[(0,) * total_degree]},
            ]
    else:

        def _retrieve_indices(term, typeSet=None):
            if typeSet == "symb":
                dictLoc = populate["realVarDict"] | populate["imVarDict"]
                refTuple = new_holVarSpace + new_antiholVarSpace
                termList = dictLoc[term]
            elif typeSet == "real":
                dictLoc = populate["holVarDict"] | populate["antiholVarDict"]
                refTuple = new_realVarSpace + new_imVarSpace
                termList = dictLoc[term]
            index_a = refTuple.index(termList[0])
            index_b = refTuple.index(termList[1], index_a + 1)
            return [index_a, index_b]

        # set up the conversion dicts for index conversion
        if _varSpace_type == "real":
            populate["preProcessMinDataToHol"] = {
                j: _retrieve_indices(varSpace[j], "symb")
                for j in range(len(varSpace))
            }

        else:  # if _varSpace_type == 'complex'
            populate["preProcessMinDataToReal"] = {
                j: _retrieve_indices(varSpace[j], "real")
                for j in range(len(varSpace))
            }

        # coordinate VF and DF conversion
        def decorateWithWeights(index, variance_rule, target="symb"):
            if variance_rule == 0:  # covariant case
                covariance = True
            else:                   # contravariant case
                covariance = False

            if target == "symb":
                if varSpace[index] in variable_registry['conversion_dictionaries']['real_part'].values():
                    holScale = sp.Rational(1, 2) if covariance else 1 # D_z (d_z) coeff of D_x (d_x)
                    antiholScale = sp.Rational(1, 2) if covariance else 1 # D_BARz (d_BARz) coeff of D_x (d_x)
                else:
                    holScale = -I / 2 if covariance else I  # D_z (d_z) coeff of D_y (d_y)
                    antiholScale = I / 2 if covariance else -I  # d_BARz (D_BARz) coeff of d_y (D_y)
                return [
                    [populate["preProcessMinDataToHol"][index][0], holScale],
                    [
                        populate["preProcessMinDataToHol"][index][1],
                        antiholScale,
                    ],
                ]
            else:  # converting from hol to real
                if varSpace[index] in variable_registry['conversion_dictionaries']['holToReal']:
                    realScale = 1 if covariance else sp.Rational(1,2)   # D_x (d_x) coeff in D_z (d_z)
                    imScale = I if covariance else -I*sp.Rational(1,2)  # D_y (d_y) coeff in D_z (d_z)
                else:
                    realScale = 1 if covariance else sp.Rational(1,2)   # D_x (d_x) coeff of D_BARz (d_BARz)
                    imScale = -I if covariance else I*sp.Rational(1,2) # D_y (d_y) coeff of D_BARz (d_BARz)
                return [
                    [populate["preProcessMinDataToReal"][index][0], realScale],
                    [populate["preProcessMinDataToReal"][index][1], imScale],
                ]

        otherDict = dict()
        for term_index, term_coeff in coeff_dict.items():
            if _varSpace_type == "real":
                reformatTarget = "symb"
            else:
                reformatTarget = "real"
            termIndices = [
                decorateWithWeights(k, valence[j], target=reformatTarget) for j,k in enumerate(term_index)
            ]
            prodWithWeights = carProd_with_weights_without_R(*termIndices)
            prodWWRescaled = [[tuple(k[0]), term_coeff * k[1]] for k in prodWithWeights]
            minimal_term_set = _shape_basis(prodWWRescaled,data_shape)
            for term in minimal_term_set:
                if term[0] in otherDict:
                    oldVal = otherDict[term[0]]
                    otherDict[term[0]] = _allToSym(oldVal + term[1])
                else:
                    otherDict[term[0]] = _allToSym(term[1])

        if _varSpace_type == "real":
            populate["realCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["compCoeffDataDict"] = [
                new_holVarSpace + new_antiholVarSpace,
                otherDict,
            ]
        else:
            populate["compCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["realCoeffDataDict"] = [
                new_realVarSpace + new_imVarSpace,
                otherDict,
            ]

    return populate,new_realVarSpace,new_holVarSpace,new_antiholVarSpace,new_imVarSpace

def _shape_basis(basis,shape):
    if shape == 'symmetric':
        old_basis = dict(basis)
        new_basis = dict()
        for index, value in old_basis.items():
            new_index = tuple(sorted(index))
            if new_index in new_basis:
                new_basis[new_index] += value
            else:
                new_basis[new_index] = value
        return list(new_basis.items())
    if shape == 'skew':
        old_basis = dict(basis)
        new_basis = dict()
        for index, value in old_basis.items():
            permS, new_index = permSign(index,returnSorted=True)
            new_index = tuple(new_index)
            if new_index in new_basis:
                new_basis[new_index] += permS*value
            else:
                new_basis[new_index] = permS*value
        return list(new_basis.items())
    return basis


############## clearing and listing
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
    This function lists all parent labels for objects tracked within the dgcv Variable Management Framework (VMF). In particular strings that are keys in dgcv's internal `standard_variable_systems`, `complex_variable_systems`, 'finite_algebra_systems', 'eps' dictionaries, etc. It also accepts optional keywords to filter the results, showing only temporary, protected, or "obscure" object system labels.

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
    from the dgcv variable management framework. Instead of printing a report, it returns
    a tuple (system_type, label) indicating what was cleared.
    """
    registry = get_variable_registry()
    global_vars = _cached_caller_globals
    cleared_info = None

    # If not tracked, nothing to clear
    if label not in registry["_labels"]:
        return None

    path = registry["_labels"][label]["path"]
    branch = path[0]

    # Handle standard variable systems
    if branch == "standard_variable_systems":
        system_dict = registry[branch][label]
        family_names = system_dict["family_names"]
        if isinstance(family_names, str):
            family_names = (family_names,)
        for var in family_names:
            global_vars.pop(var, None)
        global_vars.pop(label, None)
        if system_dict.get("differential_system"):
            for var in family_names:
                global_vars.pop(f"D_{var}", None)
                global_vars.pop(f"d_{var}", None)
            global_vars.pop(f"D_{label}", None)
            global_vars.pop(f"d_{label}", None)
        if system_dict.get("tempVar"):
            registry["temporary_variables"].discard(label)
        if system_dict.get("obsVar"):
            registry["obscure_variables"].discard(label)
        del registry[branch][label]
        cleared_info = ("standard", label)

    # Handle complex variable systems
    elif branch == "complex_variable_systems":
        system_dict = registry[branch][label]
        family_houses = system_dict["family_houses"]
        real_parent, imag_parent = family_houses[-2], family_houses[-1]
        registry["protected_variables"].discard(real_parent)
        registry["protected_variables"].discard(imag_parent)
        if system_dict["family_type"] in ("tuple", "multi_index"):
            for house in family_houses:
                global_vars.pop(house, None)
        variable_relatives = system_dict["variable_relatives"]
        for var_label, var_data in variable_relatives.items():
            global_vars.pop(var_label, None)
            if var_data.get("DFClass"):
                global_vars.pop(f"D_{var_label}", None)
            if var_data.get("VFClass"):
                global_vars.pop(f"d_{var_label}", None)
        conv = registry["conversion_dictionaries"]
        for var_label, var_data in variable_relatives.items():
            pos = var_data.get("complex_positioning")
            val = var_data.get("variable_value")
            if pos == "holomorphic":
                conv["conjugation"].pop(val, None)
                conv["holToReal"].pop(val, None)
                conv["symToReal"].pop(val, None)
            elif pos == "antiholomorphic":
                conv["symToHol"].pop(val, None)
                conv["symToReal"].pop(val, None)
            elif pos in ("real", "imaginary"):
                conv["realToHol"].pop(val, None)
                conv["realToSym"].pop(val, None)
                conv["find_parents"].pop(val, None)
        registry["temporary_variables"].discard(label)
        del registry[branch][label]
        cleared_info = ("complex", label)

    # Handle finite algebra systems
    elif branch == "finite_algebra_systems":
        system_dict = registry[branch][label]
        family_names = system_dict.get("family_names", ())
        for member in family_names:
            global_vars.pop(member, None)
        global_vars.pop(label, None)
        del registry[branch][label]
        cleared_info = ("algebra", label)

    # Handle EDS atoms
    elif branch == "eds" and path[1] == "atoms":
        system_dict = registry["eds"]["atoms"][label]
        family_names = system_dict["family_names"]
        if isinstance(family_names, str):
            family_names = (family_names,)
        for var in family_names:
            global_vars.pop(var, None)
        for var in system_dict.get("family_relatives", {}):
            global_vars.pop(var, None)
        global_vars.pop(label, None)
        del registry["eds"]["atoms"][label]
        cleared_info = ("DFAtom", label)

    # Handle EDS coframes
    elif branch == "eds" and path[1] == "coframes":
        coframe_info = registry["eds"]["coframes"][label]
        cousins_parent = coframe_info.get("cousins_parent")
        global_vars.pop(label, None)
        del registry["eds"]["coframes"][label]
        cleared_info = ("coframe", (label, cousins_parent))

    # Remove from label index
    registry["_labels"].pop(label, None)

    return cleared_info

def clearVar(*labels, report=True):
    """
    Clears variables from the registry and global namespace. Because sometimes, we all need a fresh start.

    This function takes one or more variable system labels (strings) and clears all
    associated variables, vector fields, differential forms, and metadata from the
    dgcv system. Variable system refers to object systems created by the dgcv
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
    - Comprehensively clears variables and their associated metadata from the dgcv
      system.
    - Use with `listVar` to expediantly clear everything, e.g., `clearVar(*listVar())`.

    Examples
    --------
    >>> clearVar('x') # removes any dgcv variable system labeled as x, such as
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
                f"Cleared standard variable systems from the dgcv variable management framework: {', '.join(cleared_standard)}"
            )
        if cleared_complex:
            print(
                f"Cleared complex variable systems from the dgcv variable management framework: {', '.join(cleared_complex)}"
            )
        if cleared_algebras:
            print(
                f"Cleared finite algebra systems from the dgcv variable management framework: {', '.join(cleared_algebras)}"
            )
        if cleared_diffFormAtoms:
            print(
                f"Cleared differential form systems from the dgcv variable management framework: {', '.join(cleared_diffFormAtoms)}"
            )
        if cleared_coframes:
            for cf_label, cp_label in cleared_coframes:
                print(f"Cleared coframe '{cf_label}' along with associated zero form atom system '{cp_label}'")



############## displaying summaries
def DGCV_snapshot(style=None, use_latex=None, complete_report = None):
    warnings.warn(
        "`DGCV_snapshot` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library. "
        "It will be removed in 2026. Please use `vmf_summary` instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return vmf_summary(style=style, use_latex=use_latex, complete_report = complete_report)

def vmf_summary(style=None, use_latex=None, complete_report = None):
    """
    Generate a comprehensive snapshot of dgcv's variable management framework (VMF), such as coordinate systems, algebras, coframes, and more.

    Parameters
    ----------
    style : str, optional
        The style options to apply to the summary table. Default theme is 'chalkboard_green'. Use the dgcv function `get_DGCV_themes()` to display a list of other available themes.
    use_latex : bool, optional
        If True, the table will format text using LaTeX for better
        mathematical display. Default is False.

    Returns
    -------
    DataFrame
        A formatted summary table displaying the initialized objects, such as coordinate systems, algebras, coframes, etc.
    """
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

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
        print("There are no objects currently registered in the dgcv VMF.")
    else:
        for table in outputs:
            display(table)

def variableSummary(*args, **kwargs):
    warnings.warn(
        "variableSummary() is deprecated and will be removed in a future version. "
        "Please use vmf_summary() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return vmf_summary(*args, **kwargs)

def _snapshot_coor_(style=None, use_latex=None):
    """
    Returns a Pandas DataFrame summarizing the coordinate systems in dgcv.

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
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

    def convert_to_greek(var_name):
        for name, greek in greek_letters.items():
            if var_name.lower().startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def format_variable_name(var_name, system_type, use_latex=False):
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
                if initial_index is not None and isinstance(family_names, (list, tuple)) and len(family_names) > 1:
                    content = (f"{format_latex_subscripts(var_name)} = \\left( {format_latex_subscripts(var_name,nest_braces=True)}_{{{initial_index}}}, "
                               f"\\ldots, {format_latex_subscripts(var_name,nest_braces=True)}_{{{initial_index + len(family_names) - 1}}} \\right)")
                else:
                    content = f"{convert_to_greek(var_name)} = {convert_to_greek(var_name)}"
            else:
                content = f"{var_name} = ({family_names[0]}, ..., {family_names[-1]})"
        else:
            content = f"{var_name}"
        if use_latex:
            content = f"${content}$"
        return content

    def format_latex_subscripts(var_name, nest_braces=False):
        """for use_latex branches"""
        if var_name[-1]=='_':
            var_name=var_name[:-1]
        parts = var_name.split("_")
        if len(parts) == 1:
            return convert_to_greek(var_name)
        base = convert_to_greek(parts[0])
        subscript = ", ".join(parts[1:])
        if nest_braces is True:
            return f"{{{base}_{{{subscript}}}}}"
        else:
            return f"{base}_{{{subscript}}}"

    def build_object_string(obj_type, var_name, start_index, tuple_len, system_type, use_latex=False):
        # For standard systems, if there's only one variable, return a simple label;
        # otherwise, show the range of indices.
        if tuple_len == 1:
            if use_latex:
                if use_latex:
                    var_name=format_latex_subscripts(var_name)
                if obj_type=='D':
                    content = f"$\\frac{{\\partial}}{{\\partial {var_name}}}$"
                else:
                    content = f"$\\operatorname{{d}} {var_name}$"
            else:
                content = f"{obj_type}_{var_name}"
        else:
            if use_latex:
                if use_latex:
                    var_name=format_latex_subscripts(var_name,nest_braces=True)
                if obj_type=='D':
                    content = f"$\\frac{{\\partial}}{{\\partial {var_name}_{{{start_index}}}}}$, $\\ldots$, $\\frac{{\\partial}}{{\\partial {var_name}_{{{start_index + tuple_len - 1}}}}}$"
                else:
                    content = f"$\\operatorname{{d}} {var_name}_{{{start_index}}}$, $\\ldots$, $\\operatorname{{d}} {var_name}_{{{start_index + tuple_len - 1}}}$"
            else:
                content = f"{obj_type}_{var_name}{start_index},...,{obj_type}_{var_name}{start_index + tuple_len - 1}"
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
                        part_str = f"\\frac{{\\partial}}{{\\partial {format_latex_subscripts(part,nest_braces=True)}_{{{start_index}}}}}, \\ldots, \\frac{{\\partial}}{{\\partial {format_latex_subscripts(part,nest_braces=True)}_{{{start_index + len(part_names) - 1}}}}}" if obj_type=="D" else f"d {format_latex_subscripts(part,nest_braces=True)}_{{{start_index}}}, \\ldots, d {format_latex_subscripts(part,nest_braces=True)}_{{{start_index + len(part_names) - 1}}}"
                        part_str = f"${part_str}$"
                    else:
                        part_str = f"{obj_type}_{part}{start_index},...,{obj_type}_{part}{start_index + len(part_names) - 1}"
                else:
                    if use_latex:
                        part_str = f"${obj_type}_{format_latex_subscripts(part)}$"
                    else:
                        part_str = f"{obj_type}_{part}"
                parts.append(part_str)
        else:
            # Fallback: if family_names is not the expected 4-tuple, treat as a single list.
            if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
                if use_latex:
                    part_str = f"${obj_type}_{format_latex_subscripts(family_houses[0])}{start_index}$, $\\ldots$, ${obj_type}_{format_latex_subscripts(family_houses[0])}{start_index + len(family_names) - 1}$"
                else:
                    part_str = f"{obj_type}_{family_houses[0]}{start_index},...,{obj_type}_{family_houses[0]}{start_index + len(family_names) - 1}"
                parts.append(part_str)
            else:
                if use_latex:
                    parts.append(f"${obj_type}_{format_latex_subscripts(family_houses[0])}$")
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
            real_part = f"${format_latex_subscripts(family_houses[2])} = \\left( {format_latex_subscripts(family_houses[2],nest_braces=True)}_{{{start_index}}}, \\ldots, {format_latex_subscripts(family_houses[2],nest_braces=True)}_{{{start_index + len(real_names) - 1}}} \\right)$" if isinstance(real_names, (list, tuple)) and len(real_names)>1 else f"${format_latex_subscripts(family_houses[2])}$"
            imag_part = f"${format_latex_subscripts(family_houses[3])} = \\left( {format_latex_subscripts(family_houses[3],nest_braces=True)}_{{{start_index}}}, \\ldots, {format_latex_subscripts(family_houses[3],nest_braces=True)}_{{{start_index + len(imag_names) - 1}}} \\right)$" if isinstance(imag_names, (list, tuple)) and len(imag_names)>1 else f"${format_latex_subscripts(family_houses[3])}$"
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


    # Combine labels from coordinate systems and algebras.
    all_labels = var_system_labels
    # Prepend the label to each data row.
    combined_data = []
    for label, row in zip(all_labels, data):
        combined_data.append([label] + row)

    columns = ["Coordinate System", "# of Variables", "Real Part", "Imaginary Part", "Vector Fields", "Differential Forms"]
    table = pd.DataFrame(combined_data, columns=columns)
    table_header = "Initialized Coordinate Systems"
    table_styles = get_style(style)+[{'selector': 'td', 'props': [('text-align', 'left')]}]+[{'selector': 'th', 'props': [('text-align', 'left')]}]
    styled_table = (
        table.style
        .set_table_styles(table_styles)
        .set_caption(f"{table_header}")
        .hide(axis="index")
    )
    return styled_table

def _snapshot_algebras_(style=None, use_latex=None):
    """
    Returns a Pandas DataFrame summarizing the finite algebra systems in dgcv's VMF.

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
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']
    def convert_to_greek(var_name):
        # Assume greek_letters is defined module-wide as a dict mapping strings to LaTeX-valid Greek strings.
        # Example: {'alpha': '\\alpha', 'beta': '\\beta', ...}
        for name, greek in greek_letters.items():
            if var_name.lower().startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def process_basis_label(label, ul):
        if ul is True:
            # Use regex to separate alphabetic and numeric parts.
            match = re.match(r"(.*?)(\d+)?$", label)
            if match:
                basis_elem_name = match.group(1).replace("_", "")
                basis_elem_number = match.group(2)
                basis_elem_name = convert_to_greek(basis_elem_name)
                if basis_elem_number:
                    return f"${basis_elem_name}_{{{basis_elem_number}}}$"
                else:
                    return f"${basis_elem_name}$"
            else:
                return f'${label}$ '
        else:
            return label

    registry = get_variable_registry()
    data = []

    finite_algebras = registry.get("finite_algebra_systems", {})
    for label in sorted(finite_algebras.keys()):
        system = finite_algebras[label]
        family_names = system.get("family_names", ())
        if isinstance(family_names, (list, tuple)):
            num_basis = len(family_names)
            if num_basis > 5:
                basis_str = f"{process_basis_label(family_names[0],ul=use_latex)}, ..., {process_basis_label(family_names[-1],ul=use_latex)}"
            else:
                basis_str = ", ".join(process_basis_label(x,ul=use_latex) for x in family_names)
        else:
            num_basis = 1
            basis_str = process_basis_label(family_names,ul=use_latex)
        # For the algebra label, use the module-wide format_variable_name with system_type "algebra".
        if use_latex and label in _cached_caller_globals:
            formatted_label = _cached_caller_globals[label]._repr_latex_(abbrev=True)
        else:
            formatted_label = label

        # Add grading column
        if label in _cached_caller_globals:
            alg = _cached_caller_globals[label]
            grading = alg.grading
            if isinstance(grading, (list, tuple)) and len(grading) > 0 and all(isinstance(g, (list, tuple)) for g in grading) and any(g for g in grading):
                grading_str = ", ".join(f"({', '.join(map(str, g))})" for g in grading)
            else:
                grading_str = "None"
        else:
            grading_str = "None"

        data.append([formatted_label, basis_str, num_basis, grading_str])

    columns = ["Algebra Label", "Basis", "Dimension", "Grading"]
    df = pd.DataFrame(data, columns=columns)

    table_styles = get_style(style)+[{'selector': 'td', 'props': [('text-align', 'left')]}]+[{'selector': 'th', 'props': [('text-align', 'left')]}]
    styled_df = df.style.set_table_styles(table_styles)
    styled_df = (
        df.style
        .set_table_styles(table_styles)
        .set_caption("Initialized Finite-dimensional Algebras")
        .hide(axis="index")
    )
    return styled_df

def _snapshot_eds_atoms_(style=None, use_latex=None):
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
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']
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

def _snapshot_coframes_(style=None, use_latex=None):
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

    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']
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

