import numbers

from ..._deprecated.dgcv_formatter import process_basis_label
from ..._utilities._config import get_dgcv_settings_registry


def conjugation_prefix():
    return get_dgcv_settings_registry().get("conjugation_prefix", "_c_")


def ext_der_latex(string, order):
    if order == 1:
        return f"D\\left({string}\\right)"
    if order is not None and order > 1:
        return f"D^{order}\\left({string}\\right)"
    return string


def split_conjugation_prefix(label):
    pref = conjugation_prefix()
    if label and label[0 : len(pref)] == pref:
        return label[len(pref) :], True
    return label, False


def ext_der_str(string, order):
    if isinstance(order, numbers.Integral) and order != 0:
        return f"{string}_extD_{order}"
    return string


def bar_label_latex(label):
    base, conjugated = split_conjugation_prefix(label)
    if conjugated:
        to_print = process_basis_label(base)
        if "_" in to_print:
            return f"\\overline{{{to_print}".replace("_", "}^", 1)
        return f"\\overline{{{to_print}}}"
    return process_basis_label(label).replace("_", "^", 1)


def ordinal_latex(count):
    if count == 0:
        return r"0^\text{th}"
    if count == 1:
        return r"1^\text{st}"
    if count == 2:
        return r"2^\text{nd}"
    if count == 3:
        return r"3^\text{rd}"
    return str(count) + r"^\text{th}"


def ext_der_repr(string, order):
    if isinstance(order, numbers.Integral) and order > 0:
        return f"extDer({string},order = {order})"
    return string
