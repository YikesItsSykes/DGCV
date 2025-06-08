import random
import string
import warnings

from ._config import (
    _cached_caller_globals,
    get_dgcv_settings_registry,
    get_variable_registry,
)
from .vmf import clearVar

_passkey = "".join(random.choices(string.ascii_letters + string.digits, k=16))
public_key = "".join(random.choices(string.ascii_letters + string.digits, k=8))

def get_dgcv_category(obj):
    try:
        if getattr(obj, '_dgcv_class_check', None) == _passkey:
            return getattr(obj, '_dgcv_category', None)
    except Exception:
        pass
    return None

def check_dgcv_category(obj):
    return getattr(obj, '_dgcv_class_check', None) == _passkey

def create_key(prefix=None, avoid_caller_globals=False, key_length = 8):
    """
    Generates a unique alphanumeric key with an optional prefix.

    Parameters
    ----------
    prefix : str, optional
        A string to prepend to the generated key. Defaults to an empty string if not provided.
    avoid_caller_globals : bool, optional
        If True, ensures the key does not conflict with existing keys in the caller's global namespace.

    Returns
    -------
    str
        An alphanumeric key.
    """
    if prefix is None:
        prefix = ""
    if not isinstance(prefix, str):
        prefix = ""

    # Get the caller's globals if avoid_caller_globals is True
    caller_globals = {}
    if avoid_caller_globals:
        caller_globals = _cached_caller_globals

    # Generate a new key
    while True:
        key = prefix + "".join(
            random.choices(string.ascii_letters + string.digits, k=key_length)
        )
        if not avoid_caller_globals or key not in caller_globals:
            return key

def retrieve_passkey():
    """
    Returns the internal passkey for use within dgcv functions.
    """
    return _passkey

def retrieve_public_key():
    """
    Returns the public key for use in function and variable names.
    """
    return public_key

def protected_caller_globals():
    """
    Returns a set of globally protected variable labels that should not be overwritten.
    These include standard Python built-ins, special variables, and common modules.

    Returns
    -------
    set
        A set of protected global variable names.
    """
    return {
        # Built-in functions
        "print",
        "len",
        "sum",
        "max",
        "min",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "open",
        "range",
        "enumerate",
        "map",
        "filter",
        # Common modules and objects
        "math",
        "numpy",
        "sympy",
        "os",
        "sys",
        "config",
        "inspect",
        "re",
        # Special variables
        "__name__",
        "__file__",
        "__doc__",
        "__builtins__",
        "__package__",
    }

def validate_label_list(basis_labels):
    """
    Validates a list of basis labels by checking if they are already present in _cached_caller_globals.
    If a label exists and is found in the variable_registry, provides detailed instructions for clearing it.

    Parameters
    ----------
    basis_labels : list of str
        The list of basis labels to validate.

    Raises
    ------
    ValueError
        If any label in basis_labels is already present in _cached_caller_globals, with additional info if found in the variable_registry.
    """
    existing_labels = []
    to_clear = []
    detailed_message = ""
    if get_dgcv_settings_registry()['ask_before_overwriting_objects_in_vmf'] is False:
        overwritePermissionGranted = True
    else:
        overwritePermissionGranted = False

    # Loop through each label to check if it exists in _cached_caller_globals
    for label in basis_labels:
        if label in _cached_caller_globals:
            existing_labels.append(label)

            # Check if the label is a parent or child variable in variable_registry
            variable_registry = get_variable_registry()

            # Check standard, complex, and finite algebra systems
            for system_type, sub_type, system_type_name, family_names_address in [
                ("standard_variable_systems","","standard coordinate","variable_relatives"),
                ("complex_variable_systems","","complex coordinate","variable_relatives"),
                ("finite_algebra_systems","","finite-dimensional algebra","family_names"),
                ("eds","atoms","atomic differential forms","family_relatives"),
                ("eds","coframes","exterior differential","family_relatives"),
            ]:
                if system_type in variable_registry:
                    if sub_type!="" and sub_type in variable_registry[system_type]:
                        innerVR = variable_registry[system_type][sub_type]
                    else:
                        innerVR = variable_registry[system_type]
                    if label in innerVR:
                        # The label is a parent variable
                        if overwritePermissionGranted is True:
                            detailed_message += (
                                f"\n •Label'{label}' was already assigned as the label for a {system_type_name} system.\n"
                                rf"    The old object (and all of its dependant relatives) was deleted from the VMF ( and"
                                f"the global namespace) to resolve the environment for re-assignments."
                            )
                            to_clear.append(label)
                        else:
                            detailed_message += (
                                f"\n`validate_basis_labels` detected '{label}' within the dgcv Variable Management Framework "
                                f"assigned as the label for a {system_type_name} system.\n"
                                f"Apply the dgcv function `clearVar('{label}')` to clear the obstructing objects."
                            )
                    else:
                        # Check if the label is a child variable
                        for parent_label, parent_data in innerVR.items():
                            if (family_names_address in parent_data and label in parent_data[family_names_address]):
                                # The label is a child variable
                                if overwritePermissionGranted is True:
                                    detailed_message += (
                                        f"\n • Label '{label}' was already assigned as the label for an object in a {system_type_name} system.\n"
                                        r"    The old object (and all of its dependant relatives) was deleted from the "
                                        f"VMF (and the global namespace) to resolve the environment for re-assignments. "
                                    )
                                    to_clear.append(parent_label)
                                else:
                                    detailed_message += (
                                        f"\n`validate_basis_labels` detected '{label}' within the dgcv Variable Management Framework "
                                        f"associated with the {system_type_name} system '{parent_label}'.\n Apply"
                                        f"the dgcv function `clearVar('{parent_label}')` to first clear the obstructing objects."
                                    )

    if len(existing_labels)>0:
        if overwritePermissionGranted is True:
            warning_message = (
                f"The following basis labels were already defined in the current namespace: {existing_labels}. "
                "Since `set_dgcv_settings(ask_before_overwriting_objects_in_vmf=True)` was run during the current session, "
                "these previously used labels have been re-assigned to new objects."
            )
        else:
            warning_message = (
                f"Warning: The following basis labels are already defined in the current namespace: {existing_labels}. "
                "By default, `dgcv` creator functions will not overwrite such objects.  Either clear them from the "
                " global namespace before attempting label re-assignment or set the over-riding setting "
                "`set_dgcv_settings(ask_before_overwriting_objects_in_vmf=True)`."
            )

        if detailed_message:
            warning_message += detailed_message

        if overwritePermissionGranted is True:
            if get_dgcv_settings_registry()['forgo_warnings'] is not True:
                warnings.warn(warning_message+"\n To suppress warnings such as this, set `set_dgcv_settings(forgo_warnings=True)`.",UserWarning)
                clearVar(*to_clear)
            else:
                clearVar(*to_clear,report=False)
        else:
            raise ValueError(warning_message)

def protect_variable_relatives():
    variable_registry = get_variable_registry()
    return sum(
        [
            variable_registry["complex_variable_systems"][k]["family_names"][j]
            for k in variable_registry["complex_variable_systems"]
            for j in [2, 3]
        ],
        (),
    )

def validate_label(label, remove_guardrails=False):
    """
    Checks if the provided variable label starts with 'BAR', and reformats it to 'anti_' if necessary.
    Also checks if the label is a protected global or in 'protected_variables', unless remove_guardrails is True.

    Parameters
    ----------
    label : str
        A string representing the variable label to be validated.
    remove_guardrails : bool, optional
        If True, skips the check for protected global names and 'protected_variables' (default is False).

    Returns
    -------
    str
        The reformatted label.

    Raises
    ------
    ValueError
        If the label is a protected global name or is in 'protected_variables', unless remove_guardrails is True.
    """
    variable_registry = get_variable_registry()

    # Check if the label is a protected global, unless guardrails are removed
    if not remove_guardrails:
        if label in protected_caller_globals():
            raise ValueError(
                f"dgcv recognizes label '{label}' as a protected global name and recommends not using it as a variable name. Set remove_guardrails=True to force it."
            )

        # Check if the label is a child of a parent in 'protected_variables' in the variable_registry
        if label in protect_variable_relatives():
            # If protected, search through complex_variable_systems for the parent label
            if "complex_variable_systems" in variable_registry:
                for parent_label, parent_data in variable_registry[
                    "complex_variable_systems"
                ].items():
                    if (
                        "variable_relatives" in parent_data
                        and label in parent_data["variable_relatives"]
                    ):
                        # Found the parent label associated with the protected variable
                        raise ValueError(
                            f"Label '{label}' is protected within the current dgcv Variable Management Framework, "
                            f"as it is associated with the complex variable system '{parent_label}'.\n"
                            f"It is recommended to use the dgcv function `clearVar('{parent_label}')` to clear the protected variable "
                            f"from the dgcv Variable Management Framework (VMF) before reassigning this label.\n"
                            f"Or set remove_guardrails=True in the relevant dgcv object creator to force the use of this label, "
                            f"but note this can limit available features from the VMF."
                        )

        # Check if the label is in 'protected_variables' in the variable_registry
        if (
            "protected_variables" in variable_registry
            and label in variable_registry["protected_variables"]
        ):
            # If protected, search through complex_variable_systems for the parent label
            if "complex_variable_systems" in variable_registry:
                for parent_label, parent_data in variable_registry[
                    "complex_variable_systems"
                ].items():
                    if (
                        "family_houses" in parent_data
                        and label in parent_data["family_houses"]
                    ):
                        # Found the parent label associated with the protected variable
                        raise ValueError(
                            f"Label '{label}' is protected within the current dgcv Variable Management Framework, "
                            f"as it is associated with the complex variable system '{parent_label}'.\n"
                            f"It is recommended to use the dgcv function `clearVar('{parent_label}')` to clear the protected variable "
                            f"from the dgcv Variable Management Framework (VMF) before reassigning this label.\n"
                            f"Or set remove_guardrails=True in the relevant dgcv object creator to force the use of this label, "
                            f"but note this can limit available features from the VMF."
                        )

    # Check if the label starts with "BAR" and reformat if necessary
    if label.startswith("BAR"):
        reformatted_label = "anti_" + label[3:]
        warnings.warn(
            f"Label '{label}' starts with 'BAR', which has special meaning in dgcv. It has been automatically reformatted to '{reformatted_label}'."
        )
    else:
        reformatted_label = label

    return reformatted_label
