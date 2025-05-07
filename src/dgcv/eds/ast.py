# src/dgcv/eds/ast.py

import sympy as sp

from .eds import abstDFAtom, abstract_ZF

_math_functions_registry = {
    'sub': {
        'arguments': 2,
        'derivative': None,
        'simplify': lambda expr: expr,  # placeholder
        'latex': lambda args: f"{args[0]} - {args[1]}"
    },
    'log': {
        'arguments': 1,
        'derivative': lambda arg: 1 / arg,
        'simplify': lambda expr: expr,
        'latex': lambda args: f"\\log\\left({args[0]}\\right)"
    },
    'sin': {
        'arguments': 1,
        'derivative': lambda arg: dgcv_AST_assembler('cos', arg),
        'simplify': lambda expr: expr,
        'latex': lambda args: f"\\sin\\left({args[0]}\\right)"
    },
}

def dgcv_AST_assembler(func_name, *args):
    """
    Constructs an abstract syntax tree node representing a symbolic function call.

    Parameters:
        func_name (str): The function name (must be in _math_functions_registry).
        *args: Positional arguments to the symbolic function.

    Returns:
        tuple: An AST node represented as (func_name, *args).
    """
    if func_name not in _math_functions_registry:
        raise ValueError(f"Function '{func_name}' is not registered in _math_functions_registry.")

    expected_arity = _math_functions_registry[func_name]['arguments']
    if len(args) != expected_arity:
        raise ValueError(f"Function '{func_name}' expects {expected_arity} argument(s), got {len(args)}.")

    return abstract_ZF((func_name, *args))



def _add_ID(base):
    # remove zero terms: 0 + x → x
    if isinstance(base, tuple) and base[0] == "add":
        args = [arg for arg in base[1:] if arg != 0]
        if not args:
            return 0
        if len(args) == 1:
            return args[0]
        return tuple(["add"] + args)
    return base

def _add_flatten(base):
    # flatten nested adds: add(add(a,b),c) → add(a,b,c)
    if isinstance(base, tuple) and base[0] == "add":
        new_args = []
        for arg in base[1:]:
            if isinstance(arg, tuple) and arg[0] == "add":
                new_args.extend(arg[1:])
            else:
                new_args.append(arg)
        return tuple(["add"] + new_args)
    return base

def _mul_ID(base):
    # remove ones: 1 * x → x
    if isinstance(base, tuple) and base[0] == "mul":
        args = [arg for arg in base[1:] if arg != 1]
        if not args:
            return 1
        if len(args) == 1:
            return args[0]
        return tuple(["mul"] + args)
    return base

def _mul_zero(base):
    # zero annihilator: 0 * x → 0
    if isinstance(base, tuple) and base[0] == "mul":
        if any(arg == 0 for arg in base[1:]):
            return 0
    return base

def _pow_one(base):
    # x**1 → x
    if isinstance(base, tuple) and base[0] == "pow":
        _, b, e = base
        if e == 1:
            return b
    return base

def _pow_zero(base):
    # x**0 → 1
    if isinstance(base, tuple) and base[0] == "pow":
        _, b, e = base
        if e == 0:
            return 1
    return base

_simplification_rules_registry = {
    # Additive normalizations
    "a1": _add_ID,
    "a2": _add_flatten,
    # Multiplicative normalizations
    "m1": _mul_ID,
    "m2": _mul_zero,
    # Power rules
    "p1": _pow_one,
    "p2": _pow_zero,
}

class abstract_ZF_test(sp.Basic):
    """
    Experimental version of abstract_ZF supporting a symbolic function registry and planned simplification logic.
    """
    def __new__(cls, base):
        if base is None or base == list() or base == tuple():
            base = 0
        if isinstance(base, list):
            base = tuple(base)
        if isinstance(base, abstract_ZF_test):
            base = base.base
        if isinstance(base, abstDFAtom) and base.degree == 0:
            base = base.coeff
        if isinstance(base, tuple):
            op, *args = base
            new_args = []
            for arg in args:
                if isinstance(arg, abstDFAtom) and arg.degree == 0:
                    new_args.append(arg.coeff)
                elif isinstance(arg, abstract_ZF_test):
                    new_args.append(arg.base)
                else:
                    new_args.append(arg)
            return sp.Basic.__new__(cls, (op, *new_args))
        return sp.Basic.__new__(cls, base)

    def __init__(self, base):
        self.base = base

    def _repr_latex_(self):
        return f"${self._latex(self.base)}$"

    def _latex(self, expr):
        if isinstance(expr, tuple):
            op, *args = expr
            if op in _math_functions_registry and "latex" in _math_functions_registry[op]:
                latex_func = _math_functions_registry[op]["latex"]
                return latex_func([self._latex(arg) for arg in args])
            return f"{op}({', '.join(map(self._latex, args))})"
        return sp.latex(expr)
