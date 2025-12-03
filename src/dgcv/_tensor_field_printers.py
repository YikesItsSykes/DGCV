import re

import sympy as sp

from ._config import from_vsr, greek_letters
from .backends._caches import _get_expr_types

joinders = {'symmetric':{'latex':'\\odot ','plain':'&'},'skew':{'latex':'\\wedge ','plain':'*'}}
def _shape_joinders(shape,format):
    if shape in ['symmetric','skew']:
        return joinders[shape][format]
    if format == 'latex':
        return '\\otimes '
    return '@'

def tensor_field_printer(tensor):
    terms = tensor.coeff_dict
    valence = tensor.valence

    def coeff_formatter(scalar):
        """
        Formats scalar coefficients for printing.
        """
        if scalar == 1:
            return ""
        if scalar == -1:
            return "-"
        scalar_str = str(scalar)
        if isinstance(scalar, sp.Basic) and len(scalar.args) > 0:
            return f"({scalar_str})*"
        return scalar_str+'*'

    formatted_terms = []
    for vec, scalar in terms.items():
        if scalar != 0:
            basis_elements = [
                f"D_{tensor.varSpace[vec[j]]}" if valence[j] == 1 else f"d_{tensor.varSpace[vec[j]]}"
                for j in range(len(valence))
            ]
            basis_element = _shape_joinders(tensor.data_shape,'plain').join(basis_elements)
            formatted_terms.append(f"{coeff_formatter(scalar)}{basis_element}")

    if not formatted_terms:  # If all scalars are zero
        pref_index = tensor._simplifyKW['preferred_basis_element'] if tensor._simplifyKW['preferred_basis_element'] else (0,) * len(valence)
        basis_elements = [
            f"D_{tensor.varSpace[pref_index[j]]}" if valence[j] == 1 else f"d_{tensor.varSpace[pref_index[j]]}"
            for j in range(len(valence))
        ]
        formatted_terms.append(f"0{_shape_joinders(tensor.data_shape,'plain').join(basis_elements)}")

    # join terms
    result = formatted_terms[0]
    for term in formatted_terms[1:]:
        if term.startswith("-"):
            result += term
        else:
            result += f"+{term}"

    return result

def tensor_field_latex(tensor,raw=False):
    """
    Generates LaTeX representation of a tensorField based on its data shape.
    """
    terms = tensor.coeff_dict
    valence = tensor.valence

    def coeff_formatter(scalar):
        """
        Formats scalar coefficients for LaTeX.

        - Wraps scalar expressions in parentheses if they contain multiple terms.
        - Handles special cases like 1 and -1.
        """
        if scalar == 1:
            return ""
        if scalar == -1:
            return "-"

        scalar_str = sp.latex(scalar)

        # Use is_Add to check for multi-term expressions
        if sp.sympify(scalar).is_Add:
            return f"\\left({scalar_str}\\right)"

        return scalar_str

    formatted_terms = []

    for vec, scalar in terms.items():
        if scalar != 0:
            basis_elements = [
                f"\\frac{{\\partial}}{{\\partial {_process_var_label(tensor.varSpace[index])}}}"
                if valence[j] == 1
                else f"d {_process_var_label(tensor.varSpace[index])}"
                for j, index in enumerate(vec)
            ]
            basis_element = _shape_joinders(tensor.data_shape, 'latex').join(basis_elements)
            formatted_terms.append(f"{coeff_formatter(scalar)} {basis_element}")

    # Handle case where all coefficients are zero
    if not terms or all(scalar == 0 for scalar in terms.values()):
        pref_index = tensor._simplifyKW['preferred_basis_element'] if tensor._simplifyKW['preferred_basis_element'] else (0,) * len(valence)
        basis_elements = [
            f"\\frac{{\\partial}}{{\\partial {_process_var_label(tensor.varSpace[pref_index[j]])}}}"
            if valence[j] == 1
            else f"d {_process_var_label(tensor.varSpace[pref_index[j]])}"
            for j in range(len(valence))
        ]
        formatted_terms.append("0" + _shape_joinders(tensor.data_shape, 'latex').join(basis_elements))

    # Join terms with "+" and handle "+-" cases
    latex_str = " + ".join(formatted_terms).replace("+ -", "- ")

    return latex_str if raw is True else f"${latex_str}$"

# for tensors.py
def tensor_VS_printer(tp):
    terms = tp.coeff_dict

    def coeff_formatter(scalar):
        """
        Formats scalar coefficients for printing.
        """
        if scalar == 1:
            return ""
        if scalar == -1:
            return "-"
        scalar_str = str(scalar)
        if isinstance(scalar, sp.Basic) and len(scalar.args) > 0:
            return f"({scalar_str})*"
        return scalar_str+'*'

    BL={}
    def labler(idx,vsidx,BLdict):
        if vsidx not in BLdict:
            vsl=from_vsr(vsidx)
            BLdict[vsidx] = vsl.basis_labels or [f"VS{vsl.vector_spaces.index(vsidx)}_{j+1}" for j in range(vsl.dimension)]
        return BLdict[vsidx][idx]

    formatted_terms = []
    for vec, scalar in terms.items():
        if scalar != 0:
            valence = vec[len(vec)//3:2*len(vec)//3]
            vs = vec[2*len(vec)//3:]
            idx = vec[:len(vec)//3]
            basis_elements = [
                f"{labler(idx[j],vs[j],BL)}"
                if valence[j] == 1
                else f"{labler(idx[j],vs[j],BL)}^''"
                for j in range(len(valence))
            ]
            basis_element = '@'.join(basis_elements)
            formatted_terms.append(f"{coeff_formatter(scalar)}{basis_element}")

    if not formatted_terms:  # If all scalars are zero
        vec=list(terms.keys())[0]   # find more performant way here
        vs = vec[2*len(vec)//3:]
        idx = vec[:len(vec)//3]
        basis_elements = [f"{labler(j,k,BL)}" for j,k in zip(idx,vs)]
        formatted_terms.append(f"0{'@'.join(basis_elements)}")

    # join terms
    result = formatted_terms[0]
    for term in formatted_terms[1:]:
        if term.startswith("-"):
            result += term
        else:
            result += f"+{term}"

    return result

def tensor_VS_latex(tp):
    """
    Generates LaTeX representation of a tensorProduct.
    """
    terms = tp.coeff_dict

    def coeff_formatter(scalar,bypass=None):
        """
        Formats scalar coefficients for LaTeX.

        Wraps scalar expressions in parentheses if they have more than one term.
        """
        if scalar==1:
            if bypass=='':
                return 1
            return ""
        if scalar==-1:
            if bypass=='':
                return -1
            return "-"
        scalar_str = sp.latex(scalar)
        if isinstance(scalar, _get_expr_types()) and len(scalar.args) > 0:
            return f"\\left({scalar_str}\\right)"
        return scalar_str
    basis_labels = tp.vector_space.basis_labels or [f"e_{i+1}" for i in range(tp.vector_space.dimension)]
    formatted_terms = []
    for vec, scalar in terms.items():
        valence = vec[len(vec)//2:]
        vec = vec[:len(vec)//2]
        if scalar != 0:
            basis_elements = [
                f"{_process_var_label(basis_labels[vec[j]])}"
                if valence[j] == 1
                else f"{_process_var_label(basis_labels[vec[j]])}^*"
                for j in range(len(valence))
            ]
            basis_element = '\\otimes '.join(basis_elements)
            formatted_terms.append(f"{coeff_formatter(scalar,basis_element)} {basis_element}")

    if not formatted_terms:  # If all scalars are zero
        basis_elements = [
            f"{_process_var_label(basis_labels[vec[j]])}"
            for j in range(tp.max_degree)
        ]
        formatted_terms.append("0" + '\\otimes '.join(basis_elements))

    # Join terms
    latex_str = formatted_terms[0]
    for term in formatted_terms[1:]:
        if term.startswith("-"):
            latex_str += term
        else:
            latex_str += f" + {term}"

    return f"${latex_str}$"

# for tensors.py
def tensor_latex_helper(tp):
    """
    Generates LaTeX representation of a tensorProduct instances.
    """
    terms = tp.coeff_dict

    def coeff_formatter(scalar,bypass=None):
        """
        Formats scalar coefficients for LaTeX.

        Wraps scalar expressions in parentheses if they have more than one term.
        """
        if scalar == 1:
            if bypass=='':
                return 1
            return ""
        if scalar == -1:
            if bypass=='':
                return -1
            return "-"
        scalar_str = sp.latex(scalar)
        if isinstance(scalar, _get_expr_types()) and len(scalar.args) > 0:
            return f"\\left({scalar_str}\\right)"
        return scalar_str

    def labler(idx,vsidx):
        vsl=from_vsr(vsidx)
        return vsl.basis[idx]._repr_latex_(raw=True)

    formatted_terms = []
    for vec, scalar in terms.items():
        if scalar != 0:
            valence = vec[len(vec)//3:2*len(vec)//3]
            vs = vec[2*len(vec)//3:]
            idxs = vec[:len(vec)//3]
            def pwrap(x):
                if "^" in x and x[-1]!=")":
                    return f"\\left({x}\\right)"
                return x
            basis_elements = [
                f"{labler(idxs[j],vs[j])}"
                if valence[j] == 1
                else f"{pwrap(labler(idxs[j],vs[j]))}^*"
                for j in range(len(valence))
            ]
            basis_element = '\\otimes '.join(basis_elements)
            formatted_terms.append(rf"{coeff_formatter(scalar,basis_element)} {basis_element}")

    if not formatted_terms:
        vec=list(terms.keys())[0]   # find more performant way here
        vs = vec[2*len(vec)//3:]
        idx = vec[:len(vec)//3]
        basis_elements = [f"{labler(j,k)}" for j,k in zip(idx,vs)]
        formatted_terms.append("0" + '\\otimes '.join(basis_elements))

    latex_str = formatted_terms[0]
    for term in formatted_terms[1:]:
        if term.startswith("-"):
            latex_str += term
        else:
            latex_str += f" + {term}"

    return latex_str

def _convert_to_greek(var_name):
    for name, greek in greek_letters.items():
        if var_name.startswith(name):
            return var_name.replace(name, greek, 1)
    return var_name

def _process_var_label(var,bypass=False):
    """
    Converts variable names into LaTeX-compatible labels.

    - Recognizes Greek letter names and converts them to proper LaTeX.
    - If the variable has a numerical subscript (e.g., "theta1"), it formats as "\theta_1".
    - Handles "BAR" prefix for conjugates, wrapping the final output in \\overline{}.
    """
    if bypass is True:
        return var
    var_str = str(var)
    is_conjugate = False

    # Handle "BAR" prefix
    if var_str.startswith("BAR"):
        var_str = var_str[3:]  # Remove "BAR"
        is_conjugate = True

    match = re.match(r"(.*?)(\d*)$", var_str)

    if match:
        label_part = match.group(1).rstrip("_")  # Remove trailing underscores
        number_part = match.group(2)  # Extract number if present

        label_part = _convert_to_greek(label_part)

        formatted_label = f"{label_part}_{{{number_part}}}" if number_part else label_part

        return f"\\overline{{{formatted_label}}}" if is_conjugate else formatted_label

    return var_str
