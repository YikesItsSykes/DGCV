import warnings
import random
import string
from .config import _cached_caller_globals, get_variable_registry

# Generate a dynamic passkey using random
_passkey = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

# Generate a dynamic public key using random
public_key = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def create_key():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def retrieve_passkey():
    """
    Returns the internal passkey for use within DGCV functions.
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
        'print', 'len', 'sum', 'max', 'min', 'str', 'int', 'float', 'list', 
        'dict', 'set', 'tuple', 'open', 'range', 'enumerate', 'map', 'filter',
        
        # Common modules and objects
        'math', 'numpy', 'sympy', 'os', 'sys', 'config', 'inspect', 're',
        
        # Special variables
        '__name__', '__file__', '__doc__', '__builtins__', '__package__'
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
    detailed_message = ""

    # Loop through each label to check if it exists in _cached_caller_globals
    for label in basis_labels:
        if label in _cached_caller_globals:
            existing_labels.append(label)

            # Check if the label is a parent or child variable in variable_registry
            variable_registry = get_variable_registry()

            # Check standard, complex, and finite algebra systems
            for system_type in ['standard_variable_systems', 'complex_variable_systems', 'finite_algebra_systems']:
                if system_type in variable_registry:
                    if label in variable_registry[system_type]:
                        # The label is a parent variable
                        system_name = system_type.replace('_variable_systems', '').replace('_', ' ')
                        detailed_message += (
                            f"\n`validate_basis_labels` detected '{label}' within the DGCV Variable Management Framework "
                            f"assigned as the label for a {system_name} system.\n"
                            f"Apply the DGCV function `clearVar('{label}')` to clear the obstructing objects."
                        )
                    else:
                        # Check if the label is a child variable
                        for parent_label, parent_data in variable_registry[system_type].items():
                            if 'variable_relatives' in parent_data and label in parent_data['variable_relatives']:
                                # The label is a child variable
                                system_name = system_type.replace('_variable_systems', '').replace('_', ' ')
                                detailed_message += (
                                    f"\n`validate_basis_labels` detected '{label}' within the DGCV Variable Management Framework "
                                    f"associated with the {system_name} system '{parent_label}'.\n"
                                    f"Apply the DGCV function `clearVar('{parent_label}')` to clear the obstructing objects."
                                )

    if existing_labels:
        # Include a detailed warning before raising the error
        warning_message = (
            f"Warning: The following basis labels are already defined in the current namespace: {existing_labels}. "
            "These labels may be associated with existing objects, and `validate_basis_labels` was not designed to overwrite such objects. "
            "Please clear them from the namespace before using `validate_basis_labels` to reassign the label."
        )
        
        # Combine the warning message with any detailed information about variable registry involvement
        if detailed_message:
            warning_message += detailed_message
        
        # Raise the error with the combined message
        raise ValueError(warning_message)

def protect_variable_relatives():
    variable_registry = get_variable_registry()
    return sum([variable_registry['complex_variable_systems'][k]['family_names'][j] for k in variable_registry['complex_variable_systems'] for j in [2,3]],())


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
            raise ValueError(f"DGCV recognizes label '{label}' as a protected global name and recommends not using it as a variable name. Set remove_guardrails=True to force it.")
        
        # Check if the label is a child of a parent in 'protected_variables' in the variable_registry
        if label in protect_variable_relatives():
            # If protected, search through complex_variable_systems for the parent label
            if 'complex_variable_systems' in variable_registry:
                for parent_label, parent_data in variable_registry['complex_variable_systems'].items():
                    if 'variable_relatives' in parent_data and label in parent_data['variable_relatives']:
                        # Found the parent label associated with the protected variable
                        raise ValueError(
                            f"Label '{label}' is protected within the current DGCV Variable Management Framework, "
                            f"as it is associated with the complex variable system '{parent_label}'.\n"
                            f"It is recommended to use the DGCV function `clearVar('{parent_label}')` to clear the protected variable "
                            f"from the DGCV Variable Management Framework (VMF) before reassigning this label.\n"
                            f"Or set remove_guardrails=True in the relevant DGCV object creator to force the use of this label, "
                            f"but note this can limit available features from the VMF."
                        )
                    
        # Check if the label is in 'protected_variables' in the variable_registry
        if 'protected_variables' in variable_registry and label in variable_registry['protected_variables']:
            # If protected, search through complex_variable_systems for the parent label
            if 'complex_variable_systems' in variable_registry:
                for parent_label, parent_data in variable_registry['complex_variable_systems'].items():
                    if 'family_houses' in parent_data and label in parent_data['family_houses']:
                        # Found the parent label associated with the protected variable
                        raise ValueError(
                            f"Label '{label}' is protected within the current DGCV Variable Management Framework, "
                            f"as it is associated with the complex variable system '{parent_label}'.\n"
                            f"It is recommended to use the DGCV function `clearVar('{parent_label}')` to clear the protected variable "
                            f"from the DGCV Variable Management Framework (VMF) before reassigning this label.\n"
                            f"Or set remove_guardrails=True in the relevant DGCV object creator to force the use of this label, "
                            f"but note this can limit available features from the VMF."
                        )
    
    # Check if the label starts with "BAR" and reformat if necessary
    if label.startswith("BAR"):
        reformatted_label = "anti_" + label[3:]
        warnings.warn(f"Label '{label}' starts with 'BAR', which has special meaning in DGCV. It has been automatically reformatted to '{reformatted_label}'.")
    else:
        reformatted_label = label

    return reformatted_label