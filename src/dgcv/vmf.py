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

License:
    MIT License
"""

import warnings

import sympy as sp
from sympy import I

from dgcv._tables import build_plain_table

from ._config import (
    _cached_caller_globals,
    get_dgcv_settings_registry,
    get_variable_registry,
    greek_letters,
    latex_in_html,
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

def variableSummary(*args, **kwargs):
    warnings.warn(
        "variableSummary() is deprecated and will be removed in a future version. "
        "Please use vmf_summary() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return vmf_summary(*args, **kwargs)

def vmf_summary(style=None, use_latex=None, complete_report=None):
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

    force_report = True if complete_report is True else False
    if complete_report is None:
        complete_report = True

    vr = get_variable_registry()
    builders = []
    if vr['standard_variable_systems'] or vr['complex_variable_systems'] or force_report:
        builders.append(_snapshot_coor_)
    if vr['finite_algebra_systems'] or force_report:
        builders.append(_snapshot_algebras_)
    if vr['eds']['atoms'] or force_report:
        builders.append(_snapshot_eds_atoms_)
    if vr['eds']['coframes'] or force_report:
        builders.append(_snapshot_coframes_)

    if not builders and not force_report:
        print("There are no objects currently registered in the dgcv VMF.")
        return

    container_id = "dgcv-vmf-summary"

    html_parts = []
    total = len(builders)
    for i, builder in enumerate(builders):
        is_first = (i == 0)
        is_last = (i == total - 1)
        corner_kwargs = {}
        if is_first and is_last:
            corner_kwargs = {}
        elif is_first:
            corner_kwargs = {"lr": 0, "ll": 0}
        elif is_last:
            corner_kwargs = {"ur": 0, "ul": 0}
        else:
            corner_kwargs = {"ur": 0, "lr": 0, "ll": 0, "ul": 0}

        view = builder(style=style, use_latex=use_latex, **corner_kwargs)
        html_parts.append(f'<div class="section">{view.to_html()}</div>')

    combined_html = f"""
<div id="{container_id}">
  <style>
    #{container_id} .stack {{
      display: flex;
      flex-direction: column;
      gap: 16px;
      align-items: stretch;
      width: 100%;
      margin: 0;
    }}
    #{container_id} .section {{
      width: 100%;
    }}
    #{container_id} .section table {{
      width: 100%;
      table-layout: fixed;
    }}
  </style>
  <div class="stack">
    {''.join(html_parts)}
  </div>
</div>
""".strip()

    class _HTMLWrapper:
        def __init__(self, html): self._html = html
        def to_html(self, *args, **kwargs): return self._html
        def _repr_html_(self):
            return self._html

    return latex_in_html(_HTMLWrapper(combined_html))

def _snapshot_coor_(style=None, use_latex=None, **kwargs):
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

    def convert_to_greek(var_name):
        for name, greek in greek_letters.items():
            if var_name.lower().startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def format_latex_subscripts(var_name, nest_braces=False):
        if var_name and var_name[-1] == '_':
            var_name = var_name[:-1]
        parts = var_name.split("_")
        if len(parts) == 1:
            return convert_to_greek(var_name)
        base = convert_to_greek(parts[0])
        subscript = ", ".join(parts[1:])
        return f"{{{base}_{{{subscript}}}}}" if nest_braces else f"{base}_{{{subscript}}}"

    def latex_symbol_with_overline_if_needed(name: str, index: int | None = None) -> str:
        is_bar = isinstance(name, str) and name.startswith("BAR")
        base = name[3:] if is_bar else name
        if index is None:
            inner = format_latex_subscripts(base)          # e.g., z or z_{i} (if base had underscores)
        else:
            inner = f"{convert_to_greek(base)}_{{{index}}}" # e.g., z_{i}
        return f"\\overline{{{inner}}}" if is_bar else inner

    def format_variable_name(var_name, system_type, use_latex=False):
        vr = variable_registry
        if system_type == "standard":
            info = vr["standard_variable_systems"].get(var_name, {})
            family_names = info.get("family_names", var_name)
            initial_index = info.get("initial_index", 1)
        elif system_type == "complex":
            info = vr["complex_variable_systems"].get(var_name, {})
            family_names = info.get("family_names", ())
            if family_names and isinstance(family_names, (list, tuple)) and len(family_names) > 0:
                family_names = family_names[0]
            else:
                family_names = var_name
            initial_index = info.get("initial_index", 1)
        elif system_type == "algebra":
            info = vr["finite_algebra_systems"].get(var_name, {})
            family_names = info.get("family_names", var_name)
            initial_index = None
        else:
            family_names, initial_index = var_name, None

        if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
            if use_latex:
                if initial_index is not None:
                    content = (f"{format_latex_subscripts(var_name)} = "
                               f"\\left( {format_latex_subscripts(var_name, nest_braces=True)}_{{{initial_index}}}, "
                               f"\\ldots, {format_latex_subscripts(var_name, nest_braces=True)}_{{{initial_index + len(family_names) - 1}}} \\right)")
                else:
                    content = f"{convert_to_greek(var_name)} = {convert_to_greek(var_name)}"
            else:
                content = f"{var_name} = ({family_names[0]}, ..., {family_names[-1]})"
        else:
            content = f"{convert_to_greek(var_name)}" if use_latex else f"{var_name}"
        return f"${content}$" if use_latex else content

    def build_object_string(obj_type, var_name, start_index, tuple_len, system_type, use_latex=False):
        if tuple_len == 1:
            if use_latex:
                sym = latex_symbol_with_overline_if_needed(var_name)
                return f"$\\frac{{\\partial}}{{\\partial {sym}}}$" if obj_type == 'D' else f"$\\operatorname{{d}} {sym}$"
            return f"{obj_type}_{var_name}"
        else:
            if use_latex:
                left  = latex_symbol_with_overline_if_needed(var_name, start_index)
                right = latex_symbol_with_overline_if_needed(var_name, start_index + tuple_len - 1)
                if obj_type == 'D':
                    s = f"\\frac{{\\partial}}{{\\partial {left}}}, \\ldots, \\frac{{\\partial}}{{\\partial {right}}}"
                else:
                    s = f"\\operatorname{{d}} {left}, \\ldots, \\operatorname{{d}} {right}"
                return f"${s}$"
            return f"{obj_type}_{var_name}{start_index},...,{obj_type}_{var_name}{start_index + tuple_len - 1}"

    def build_object_string_for_complex(obj_type, family_houses, family_names, start_index, use_latex=False):
        parts = []
        if isinstance(family_names, (list, tuple)) and len(family_names) == 4:
            for i, part in enumerate(family_houses):
                part_names = family_names[i]
                if isinstance(part_names, (list, tuple)) and len(part_names) > 1:
                    if use_latex:
                        left  = latex_symbol_with_overline_if_needed(part, start_index)
                        right = latex_symbol_with_overline_if_needed(part, start_index + len(part_names) - 1)
                        if obj_type == "D":
                            core = (f"\\frac{{\\partial}}{{\\partial {left}}}, \\ldots, "
                                    f"\\frac{{\\partial}}{{\\partial {right}}}")
                        else:
                            core = (f"\\operatorname{{d}} {left}, \\ldots, "
                                    f"\\operatorname{{d}} {right}")
                        parts.append(f"${core}$")
                    else:
                        parts.append(f"{obj_type}_{part}{start_index},...,{obj_type}_{part}{start_index + len(part_names) - 1}")
                else:
                    if use_latex:
                        sym = latex_symbol_with_overline_if_needed(part)
                        parts.append(f"${obj_type}_{sym}$")
                    else:
                        parts.append(f"{obj_type}_{part}")
        else:
            if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
                if use_latex:
                    left  = latex_symbol_with_overline_if_needed(family_houses[0], start_index)
                    right = latex_symbol_with_overline_if_needed(family_houses[0], start_index + len(family_names) - 1)
                    parts.append(f"$\\frac{{\\partial}}{{\\partial {left}}}$, "
                                 f"$\\ldots$, "
                                 f"$\\frac{{\\partial}}{{\\partial {right}}}$" if obj_type=="D"
                                 else f"$\\operatorname{{d}} {left}$, $\\ldots$, $\\operatorname{{d}} {right}$")
                else:
                    parts.append(f"{obj_type}_{family_houses[0]}{start_index},...,{obj_type}_{family_houses[0]}{start_index + len(family_names) - 1}")
            else:
                if use_latex:
                    sym = latex_symbol_with_overline_if_needed(family_houses[0])
                    parts.append(f"${obj_type}_{sym}$")
                else:
                    parts.append(f"{obj_type}_{family_houses[0]}")
        return ", ".join(parts)

    variable_registry = get_variable_registry()

    data = []
    var_system_labels = []

    # Complex systems
    for var_name in sorted(variable_registry.get("complex_variable_systems", {}).keys()):
        system = variable_registry["complex_variable_systems"][var_name]
        fn = system.get("family_names", ())
        hol_names = fn[0] if (fn and isinstance(fn, (list, tuple)) and len(fn) == 4) else []
        tuple_len = len(hol_names) if isinstance(hol_names, (list, tuple)) else 1
        start_index = system.get("initial_index", 1)
        formatted_label = format_variable_name(var_name, "complex", use_latex=use_latex)
        var_system_labels.append(formatted_label)
        family_houses = system.get("family_houses", ("N/A", "N/A", "N/A", "N/A"))
        if isinstance(fn, (list, tuple)) and len(fn) == 4:
            real_names = fn[2]
            imag_names = fn[3]
        else:
            real_names, imag_names = "N/A", "N/A"
        if use_latex:
            real_part = (f"${format_latex_subscripts(family_houses[2])} = "
                         f"\\left( {format_latex_subscripts(family_houses[2], nest_braces=True)}_{{{start_index}}}, "
                         f"\\ldots, {format_latex_subscripts(family_houses[2], nest_braces=True)}_{{{start_index + len(real_names) - 1}}} \\right)$"
                         if isinstance(real_names, (list, tuple)) and len(real_names) > 1
                         else f"${format_latex_subscripts(family_houses[2])}$")
            imag_part = (f"${format_latex_subscripts(family_houses[3])} = "
                         f"\\left( {format_latex_subscripts(family_houses[3], nest_braces=True)}_{{{start_index}}}, "
                         f"\\ldots, {format_latex_subscripts(family_houses[3], nest_braces=True)}_{{{start_index + len(imag_names) - 1}}} \\right)$"
                         if isinstance(imag_names, (list, tuple)) and len(imag_names) > 1
                         else f"${format_latex_subscripts(family_houses[3])}$")
        else:
            real_part = (f"{family_houses[2]} = ({real_names[0]}, ..., {real_names[-1]})"
                         if isinstance(real_names, (list, tuple)) and len(real_names) > 1
                         else f"{family_houses[2]}")
            imag_part = (f"{family_houses[3]} = ({imag_names[0]}, ..., {imag_names[-1]})"
                         if isinstance(imag_names, (list, tuple)) and len(imag_names) > 1
                         else f"{family_houses[3]}")
        vf_str = build_object_string_for_complex("D", family_houses, fn, start_index, use_latex)
        df_str = build_object_string_for_complex("d", family_houses, fn, start_index, use_latex)
        data.append([tuple_len, real_part, imag_part, vf_str, df_str])

    # Standard systems
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

    combined_data = [[label] + row for label, row in zip(var_system_labels, data)]
    columns = ["Coordinate System", "# of Variables", "Real Part", "Imaginary Part", "Vector Fields", "Differential Forms"]

    loc_style = get_style(style)
    def _get_prop(sel, prop):
        for sd in loc_style:
            if sd.get("selector") == sel:
                for k, v in sd.get("props", []):
                    if k == prop:
                        return v
        return None

    caption_ff = _get_prop("th.col_heading.level0", "font-family") or _get_prop("thead th", "font-family") or "inherit"
    caption_fs = _get_prop("th.col_heading.level0", "font-size")   or _get_prop("thead th", "font-size")   or "inherit"

    extra = [
        {"selector": "table",   "props": [("border-collapse", "collapse"), ("width","100%"), ("table-layout","fixed")]},
        {"selector": "td",      "props": [("text-align", "left")]},
        {"selector": "th",      "props": [("text-align", "left")]},
        {"selector": "caption", "props": [
            ("caption-side", "top"),
            ("text-align", "left"),
            ("margin", "0 0 6px 0"),
            ("font-family", caption_ff),
            ("font-size", caption_fs),
            ("font-weight", "bold"),
        ]},
    ]
    view = build_plain_table(
        columns=columns,
        rows=combined_data,
        caption="Initialized Coordinate Systems",
        theme_styles=loc_style,
        extra_styles=extra,
        table_attrs='style="table-layout:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        truncate_chars=None,
        **kwargs
    )
    return view

def _snapshot_algebras_(style=None, use_latex=None, **kwargs):
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

    registry = get_variable_registry()
    finite_algebras = registry.get("finite_algebra_systems", {}) or {}

    def _basis_label(x):
        if use_latex:
            try:
                return f"${x._repr_latex_(raw=True)}$"
            except Exception:
                return f"${str(x)}$"
        return repr(x)

    def _format_basis(values):
        if isinstance(values, (list, tuple)):
            n = len(values)
            if n == 0:
                return "—" if not use_latex else "$\\text{—}$"
            if n > 5:
                return f"{_basis_label(values[0])}, ..., {_basis_label(values[-1])}"
            return ", ".join(_basis_label(v) for v in values)
        # single element
        return _basis_label(values)

    def _format_alg_label(label):
        try:
            if use_latex and '_cached_caller_globals' in globals() and label in _cached_caller_globals:
                return _cached_caller_globals[label]._repr_latex_(abbrev=True)
        except Exception:
            pass
        return label

    def _format_grading(label):
        try:
            if '_cached_caller_globals' in globals() and label in _cached_caller_globals:
                alg = _cached_caller_globals[label]
                grading = getattr(alg, "grading", None)
                if isinstance(grading, (list, tuple)) and grading and all(isinstance(g, (list, tuple)) for g in grading) and any(g for g in grading):
                    return ", ".join(f"({', '.join(map(str, g))})" for g in grading)
        except Exception:
            pass
        return "None"

    rows = []
    for label in sorted(finite_algebras.keys()):
        system = finite_algebras[label] or {}
        family_values = system.get("family_values", ())
        basis_str = _format_basis(family_values)

        if isinstance(family_values, (list, tuple)):
            dim = len(family_values)
        else:
            dim = 1

        alg_label = _format_alg_label(label)
        if use_latex and isinstance(alg_label, str) and not alg_label.startswith("$"):
            alg_label = f"${alg_label}$"

        grading_str = _format_grading(label)

        rows.append([alg_label, basis_str, dim, grading_str])

    columns = ["Algebra Label", "Basis", "Dimension", "Grading"]

    loc_style = get_style(style)
    def _get_prop(sel, prop):
        for sd in loc_style:
            if sd.get("selector") == sel:
                for k, v in sd.get("props", []):
                    if k == prop:
                        return v
        return None

    caption_ff = _get_prop("th.col_heading.level0", "font-family") or _get_prop("thead th", "font-family") or "inherit"
    caption_fs = _get_prop("th.col_heading.level0", "font-size")   or _get_prop("thead th", "font-size")   or "inherit"

    extra = [
        {"selector": "table",   "props": [("border-collapse", "collapse"), ("width","100%"), ("table-layout","fixed")]},
        {"selector": "td",      "props": [("text-align", "left")]},
        {"selector": "th",      "props": [("text-align", "left")]},
        {"selector": "caption", "props": [
            ("caption-side", "top"),
            ("text-align", "left"),
            ("margin", "0 0 6px 0"),
            ("font-family", caption_ff),
            ("font-size", caption_fs),
            ("font-weight", "bold"),
        ]},
    ]

    view = build_plain_table(
        columns=columns,
        rows=rows,
        caption="Initialized Finite-dimensional Algebras",
        theme_styles=loc_style,
        extra_styles=extra,
        table_attrs='style="table-layout:auto; overflow-x:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        truncate_chars=None,
        **kwargs
    )
    return view

def _snapshot_eds_atoms_(style=None, use_latex=None, **kwargs):
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

    vr = get_variable_registry()
    eds_atoms_registry = (vr.get("eds", {}) or {}).get("atoms", {}) or {}

    columns = ["DF System", "Degree", "# Elements", "Differential Forms", "Conjugate Forms", "Primary Coframe"]
    rows = []

    if not eds_atoms_registry:
        loc_style = get_style(style)
        def _get_prop(sel, prop):
            for sd in loc_style:
                if sd.get("selector") == sel:
                    for k, v in sd.get("props", []):
                        if k == prop:
                            return v
            return None

        caption_ff = _get_prop("th.col_heading.level0", "font-family") or _get_prop("thead th", "font-family") or "inherit"
        caption_fs = _get_prop("th.col_heading.level0", "font-size")   or _get_prop("thead th", "font-size")   or "inherit"

        extra = [
            {"selector": "table",   "props": [("border-collapse", "collapse"), ("width","100%"), ("table-layout","fixed")]},
            {"selector": "td",      "props": [("text-align", "left")]},
            {"selector": "th",      "props": [("text-align", "left")]},
            {"selector": "caption", "props": [
                ("caption-side", "top"),
                ("text-align", "left"),
                ("margin", "0 0 6px 0"),
                ("font-family", caption_ff),
                ("font-size", caption_fs),
                ("font-weight", "bold"),
            ]},
        ]

        return build_plain_table(
            columns=columns,
            rows=[],
            caption="Initialized abstract differential forms in the VMF scope",
            theme_styles=loc_style,
            extra_styles=extra,
            table_attrs='style="table-layout:auto; overflow-x:auto;"',
            cell_align=None,
            escape_cells=False,
            escape_headers=True,
            nowrap=False,
            truncate_chars=None,
        )

    for label, system in sorted(eds_atoms_registry.items()):
        df_system = label
        degree = system.get("degree", "----")
        family_values = system.get("family_values", ())
        num_elements = len(family_values) if isinstance(family_values, tuple) else 1

        # Differential Forms
        if isinstance(family_values, tuple):
            if len(family_values) > 3:
                diff_forms = f"{family_values[0]}, ..., {family_values[-1]}"
            else:
                diff_forms = ", ".join(str(x) for x in family_values)
        else:
            diff_forms = str(family_values)

        if use_latex and family_values:
            if isinstance(family_values, tuple) and len(family_values) > 3:
                left = family_values[0]._latex() if hasattr(family_values[0], "_latex") else str(family_values[0])
                right = family_values[-1]._latex() if hasattr(family_values[-1], "_latex") else str(family_values[-1])
                diff_forms = f"$ {left}, ..., {right} $"
            elif isinstance(family_values, tuple):
                inner = ", ".join((x._latex() if hasattr(x, "_latex") else str(x)) for x in family_values)
                diff_forms = f"$ {inner} $"

        # Conjugate Forms
        real_status = system.get("real", False)
        if real_status:
            conjugate_forms = "----"
        else:
            conjugates = system.get("conjugates", {})
            if conjugates:
                conj_list = list(conjugates.values())
                if len(conj_list) > 3:
                    conjugate_forms = f"{conj_list[0]}, ..., {conj_list[-1]}"
                else:
                    conjugate_forms = ", ".join(str(x) for x in conj_list)
                if use_latex and conj_list and hasattr(conj_list[0], "_latex"):
                    if len(conj_list) > 3:
                        left = conj_list[0]._latex()
                        right = conj_list[-1]._latex()
                        conjugate_forms = f"$ {left}, ..., {right} $"
                    else:
                        inner = ", ".join(x._latex() for x in conj_list)
                        conjugate_forms = f"$ {inner} $"
            else:
                conjugate_forms = "----"

        # Primary Coframe
        primary_coframe = system.get("primary_coframe", None)
        if primary_coframe is None:
            primary_coframe_str = "----"
        else:
            primary_coframe_str = (
                primary_coframe._latex() if use_latex and hasattr(primary_coframe, "_latex")
                else repr(primary_coframe)
            )

        rows.append([df_system, degree, num_elements, diff_forms, conjugate_forms, primary_coframe_str])

    loc_style = get_style(style)
    def _get_prop(sel, prop):
        for sd in loc_style:
            if sd.get("selector") == sel:
                for k, v in sd.get("props", []):
                    if k == prop:
                        return v
        return None

    caption_ff = _get_prop("th.col_heading.level0", "font-family") or _get_prop("thead th", "font-family") or "inherit"
    caption_fs = _get_prop("th.col_heading.level0", "font-size")   or _get_prop("thead th", "font-size")   or "inherit"

    extra = [
        {"selector": "table",   "props": [("border-collapse", "collapse"), ("width","100%"), ("table-layout","fixed")]},
        {"selector": "td",      "props": [("text-align", "left")]},
        {"selector": "th",      "props": [("text-align", "left")]},
        {"selector": "caption", "props": [
            ("caption-side", "top"),
            ("text-align", "left"),
            ("margin", "0 0 6px 0"),
            ("font-family", caption_ff),
            ("font-size", caption_fs),
            ("font-weight", "bold"),
        ]},
    ]

    return build_plain_table(
        columns=columns,
        rows=rows,
        caption="Initialized abstract differential forms in the VMF scope",
        theme_styles=loc_style,
        extra_styles=extra,
        table_attrs='style="table-layout:auto; overflow-x:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        truncate_chars=None,
        **kwargs
    )

def _snapshot_coframes_(style=None, use_latex=None, **kwargs):
    """
    Returns a summary table listing coframes in the VMF scope
    """
    if style is None:
        style = get_dgcv_settings_registry()['theme']
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()['use_latex']

    vr = get_variable_registry()
    coframes_registry = (vr.get("eds", {}) or {}).get("coframes", {}) or {}

    def _latex_of(obj):
        try:
            if hasattr(obj, "_repr_latex_"):
                s = obj._repr_latex_(raw=True)
                if s is not None:
                    return s.strip()
            if hasattr(obj, "_latex"):
                s = obj._latex()
                if s is not None:
                    return s.strip()
        except Exception:
            pass
        return None

    def _fmt_one(x):
        if use_latex:
            s = _latex_of(x)
            if s is not None:
                return f"${s}$"
        return repr(x)

    def _fmt_list(xs):
        xs = list(xs or [])
        if not xs:
            return "$\\varnothing$" if use_latex else "∅"
        if len(xs) > 3:
            return f"{_fmt_one(xs[0])}, ..., {_fmt_one(xs[-1])}"
        return ", ".join(_fmt_one(x) for x in xs)

    rows = []
    for label, system in sorted(coframes_registry.items()):
        coframe_obj = _cached_caller_globals.get(label, label)
        coframe_label = label

        # Coframe 1-forms
        if isinstance(coframe_obj, str):
            children = list(system.get("children", []) or [])
            if children and all(ch in _cached_caller_globals for ch in children):
                forms = [_cached_caller_globals[ch] for ch in children]
            else:
                forms = []
        else:
            forms = list(getattr(coframe_obj, "forms", []) or [])
        forms_cell = _fmt_list(forms)

        # Structure coefficients (cousins)
        cousins = list(system.get("cousins_vals", []) or [])
        cousins_cell = _fmt_list(cousins)

        rows.append([coframe_label, forms_cell, cousins_cell])

    columns = ["Coframe Label", "Coframe 1-Forms", "Structure Coefficients"]

    loc_style = get_style(style)
    def _get_prop(sel, prop):
        for sd in loc_style:
            if sd.get("selector") == sel:
                for k, v in sd.get("props", []):
                    if k == prop:
                        return v
        return None

    caption_ff = _get_prop("th.col_heading.level0", "font-family") or _get_prop("thead th", "font-family") or "inherit"
    caption_fs = _get_prop("th.col_heading.level0", "font-size")   or _get_prop("thead th", "font-size")   or "inherit"

    extra = [
        {"selector": "table",   "props": [("border-collapse", "collapse"), ("width","100%"), ("table-layout","fixed")]},
        {"selector": "td",      "props": [("text-align", "left")]},
        {"selector": "th",      "props": [("text-align", "left")]},
        {"selector": "caption", "props": [
            ("caption-side", "top"),
            ("text-align", "left"),
            ("margin", "0 0 6px 0"),
            ("font-family", caption_ff),
            ("font-size", caption_fs),
            ("font-weight", "bold"),
        ]},
    ]

    from dgcv._tables import build_plain_table
    view = build_plain_table(
        columns=columns,
        rows=rows,  # may be empty; we still render an empty themed table
        caption="Initialized Abstract Coframes",
        theme_styles=loc_style,
        extra_styles=extra,
        table_attrs='style="table-layout:auto; overflow-x:auto;"', # table-layout:auto;
        cell_align=None,
        escape_cells=False,   # allow $...$ LaTeX in cells
        escape_headers=True,
        nowrap=False,
        truncate_chars=None,
        **kwargs
    )
    return view
