import warnings
from collections import Counter
from functools import total_ordering

import sympy as sp

from ._safeguards import validate_label
from .combinatorics import weightedPermSign
from .config import _cached_caller_globals, get_variable_registry
from .DGCVFormatter import process_basis_label


@total_ordering
class SortableObj:
    def __init__(self, pair: tuple):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Input must be a tuple of (value, sortable_key)")
        self.pair = pair
        self.value = pair[0]
        self.place = pair[1]

    def __lt__(self, other):
        if not isinstance(other, SortableObj):
            return NotImplemented
        return self.place < other.place  # Comparison is based on the sortable key

    def __eq__(self, other):
        if not isinstance(other, SortableObj):
            return NotImplemented
        return self.place == other.place  # Equality is based on the sortable key

    def __repr__(self):
        return f"SortableObj(value={self.value}, place={self.place})"

@total_ordering
class barSortedStr:
    def __init__(self, label=None):
        if label:
            if not isinstance(label, str):
                raise ValueError(f"Input must be a str. Recieved {label}")
            self.str = label
        else:
            self.str = None

    def __lt__(self, other):
        if self.str is None:
            return True
        if other is None:
            return False
        if isinstance(other,barSortedStr):
            other = other.str
            if other is None:
                return False
        if not isinstance(other, str):
            return NotImplemented
        if self.str[0:3]=="BAR":
            if other[0:3]=="BAR":
                return self.str<other
            else:
                return False
        else:
            if other[0:3]=="BAR":
                return True
            else:
                return self.str<other

    def __eq__(self, other):
        if self.str is None:
            if other is None or (isinstance(other,barSortedStr) and other.str is None):
                return True
            else:
                return False
        if isinstance(other,barSortedStr):
            other = other.str
            if other is None:
                return False
        if not isinstance(other, str):
            return NotImplemented
        return self.str == other  # Equality is based on str label

    def __repr__(self):
        return f"barSortedStr(value={self.str})"

def _custom_conj(expr):
    if isinstance(expr,(abstract_ZF,zeroFormAtom,abstDFAtom,abstDFMonom,abstract_DF)):
        return expr._eval_conjugate()
    else:
        return sp.conjugate(expr)

class zeroFormAtom(sp.Basic):
    def __new__(cls, label, partials_orders=None, coframe=None, _markers=frozenset(), coframe_independants=frozenset()):
        """
        Create a new zeroFormAtom instance.

        Parameters:
        - label (str): The base function label.
        - partials_orders (tuple[int], optional): Tuple of non-negative integers representing partial derivatives.
          Defaults to an empty tuple (no derivatives applied).
        """
        if not isinstance(label, str):
            raise TypeError(f"label must be type `str`. Instead received `{type(label)}`\n given label: {label}")
        if coframe is None:
            coframe = abst_coframe(tuple(), {})
        elif not isinstance(coframe, abst_coframe):
            raise TypeError('Expected given `coframe` to be None or have type `abst_coframe`')

        if partials_orders is None:
            partials_orders = (0,) * len(coframe.forms)
        elif not all(isinstance(order, int) and order >= 0 for order in partials_orders) or len(partials_orders) != len(coframe.forms):
            raise ValueError(f"partials_orders must be a tuple of non-negative integers of length matching the associated coframe basis (if any).\n given `partials_orders`: {partials_orders} \n given coframe basis: {coframe.forms}")

        # Using SymPy's Basic constructor
        obj = sp.Basic.__new__(cls, label, partials_orders, coframe)
        return obj

    def __init__(self, label, partials_orders=None, coframe=None, _markers=frozenset(),coframe_independants=frozenset()):
        """
        Initialize attributes (already set by __new__).
        """
        if coframe is None:
            coframe = abst_coframe(tuple(), {})
        if partials_orders is None:
            partials_orders = (0,) * len(coframe.forms)
        self.label = label
        self.partials_orders = tuple(partials_orders)
        self.coframe = coframe
        self._markers = _markers
        self.coframe_independants = coframe_independants
        self.is_constant = 'constant' in _markers
        self.is_one = self.label == '_1' and self.is_constant
        self._is_zero = self.label == '_0' and self.is_constant

    @property
    def is_zero(self):
        """Property to safely expose the zero check."""
        return self._is_zero

    def __eq__(self, other):
        """
        Check equality of two zeroFormAtom instances.
        """
        if not isinstance(other, zeroFormAtom):
            return NotImplemented
        return self.label == other.label and self.partials_orders == other.partials_orders

    def __hash__(self):
        """
        Hash the zeroFormAtom instance based on its label and partials_orders.
        """
        return hash((self.label, self.partials_orders))

    def _eval_conjugate(self):
        """
        Define how `sympy.conjugate()` should behave for zeroFormAtom instances.
        """
        if "real" in self._markers:
            conjugated_label = self.label
        elif self.label.startswith("BAR"):
            conjugated_label = self.label[3:]  # Remove "BAR" prefix
        else:
            conjugated_label = f"BAR{self.label}"  # Add "BAR" prefix

        newPO = tuple(self.partials_orders[self.coframe.conj_rules[j]] for j in range(len(self.partials_orders)))

        # Return a new zeroFormAtom with the conjugated label
        return zeroFormAtom(conjugated_label, newPO, self.coframe, self._markers, self.coframe_independants)

    def __mul__(self, other):
        """
        Multiplication of zeroFormAtom:
        - With another zeroFormAtom --> Becomes a structured `abstract_ZF` multiplication.
        - With a scalar (int/float/sympy.Expr) --> Wraps in `abstract_ZF`.
        """
        if self.is_one:
            return other
        if self.is_zero:
            return abstract_ZF(0)
        if isinstance(other, (zeroFormAtom, abstract_ZF, int, float, sp.Expr)):
            return abstract_ZF(("mul", self, other))
        return NotImplemented

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        """
        Division of zeroFormAtom:
        - Returns a structured `abstract_ZF` division.
        """
        if isinstance(other, (zeroFormAtom, int, float, sp.Expr)):
            return abstract_ZF(("div", self, other))
        return NotImplemented

    def __pow__(self, exp):
        """
        Exponentiation of zeroFormAtom:
        - Returns a structured `abstract_ZF` exponentiation.
        """
        if isinstance(exp, (int, float, sp.Expr, zeroFormAtom, abstract_ZF)):
            return abstract_ZF(('pow',self,exp))
        return NotImplemented

    def __add__(self, other):
        """
        Addition of zeroFormAtom:
        - Returns a structured `abstract_ZF` addition.
        """
        if isinstance(other, (zeroFormAtom, int, float, sp.Expr, abstract_ZF)):
            return abstract_ZF(("add", self, other))
        return NotImplemented

    def __sub__(self, other):
        """
        Subtraction of zeroFormAtom:
        - Returns a structured `abstract_ZF` subtraction.
        """
        if isinstance(other, (zeroFormAtom, int, float, sp.Expr, abstract_ZF)):
            return abstract_ZF(("sub", self, other))
        return NotImplemented

    def subs(self, data):
        """
        Symbolic substitution in zeroFormAtom.
        """
        if isinstance(data, (list, tuple)) and all(isinstance(j, tuple) and len(j) == 2 for j in data):
            l1 = len(data)
            data = dict(data)
            if len(data) < l1:
                warnings.warn('Provided substitution rules had repeat keys, and only one was used.')

        if isinstance(data, dict):
            if self in data:
                new_value = data[self]
                if isinstance(new_value, (zeroFormAtom, abstract_ZF, int, float, sp.Expr)):
                    return new_value
                else:
                    raise TypeError(f'subs() cannot replace a `zeroFormAtom` with object type {type(new_value)}.')
            else:
                return self
        else:
            raise TypeError('`zeroFormAtom.subs()` received unsupported subs data.')

    def __repr__(self):
        return (f"zeroFormAtom({self.label!r})")

    def __str__(self):
        """
        Fallback string representation.
        """
        if self.is_one:
            return '1'
        if self.is_zero:
            return '0'

        if self.partials_orders and any(j!=0 for j in self.partials_orders):
            partials_str = "_".join(map(str, self.partials_orders))
            return f"D_{partials_str}({self.label})"

        return self.label


    def _latex(self, printer=None):
        """
        LaTeX representation for zeroFormAtom.
        """
        if self.is_one:
            return '1'
        if self.is_zero:
            return '0'

        base_label = self.label
        conjugated = False
        if base_label.startswith("BAR"):
            base_label = base_label[3:]
            conjugated = True

        index_start = None
        if "_low_" in base_label:
            index_start = base_label.index("_low_")
        elif "_hi_" in base_label:
            index_start = base_label.index("_hi_")

        if index_start is not None:
            first_part = base_label[:index_start]
            index_part = base_label[index_start:]
        else:
            first_part = base_label
            index_part = ""

        # Process the base part
        formatted_label = f"\\bar{{{process_basis_label(first_part)}}}" if conjugated else process_basis_label(first_part)

        if "_" in formatted_label and index_part is not None and self.partials_orders and any(j!=0 for j in self.partials_orders):
            formatted_label = f"\\left({formatted_label}\\right)"

        # Extract lower and upper indices
        lower_list, upper_list = [], []
        if "_low_" in index_part:
            lower_part = index_part.split("_low_")[1]
            if "_hi_" in lower_part:
                lower_part, upper_part = lower_part.split("_hi_")
                upper_list = upper_part.split("_")
            lower_list = lower_part.split("_")
        elif "_hi_" in index_part:
            upper_part = index_part.split("_hi_")[1]
            upper_list = upper_part.split("_")

        # Conjugate index formatter
        def cIdx(idx):
            idx = int(idx)
            if idx - 1 in self.coframe.inverted_conj_rules:
                return f'\\bar{1 + self.coframe.inverted_conj_rules[idx - 1]}'
            else:
                return f"{idx}"

        # Convert string indices to LaTeX-compatible integers
        lower_list = [cIdx(idx) for idx in lower_list if idx]
        upper_list = [cIdx(idx) for idx in upper_list if idx]

        # Extract partial derivative indices
        partials_indices = []
        if self.partials_orders is not None:
            for j, count in enumerate(self.partials_orders):
                partials_indices.extend([j + 1] * count)

        # Combine indices into the LaTeX string
        lower_str = ",".join(lower_list)
        partials_str = ",".join(map(cIdx, partials_indices))
        upper_str = ",".join(upper_list)

        indices_str = ""
        if lower_str or 'verbose' in self._markers:
            indices_str += f"_{{{lower_str};{partials_str}}}".replace(";}", "}")  # Omit ';' if no partials
        elif partials_str:
            indices_str += f"_{{{partials_str}}}"
        if upper_str:
            indices_str += f"^{{{upper_str}}}"

        return f"{formatted_label}{indices_str}"

    def _repr_latex_(self):
        """
        Jupyter Notebook LaTeX representation for zeroFormAtom.
        """
        return f"${sp.latex(self)}$"

class abstract_ZF(sp.Basic):
    """
    Symbolic expression class that represents all operations (+, -, *, /, **)
    built on top of zeroFormAtom.
    """
    def __new__(cls, base):
        """
        Creates a new abstract_ZF instance.
        """
        if base is None or base==list() or base==tuple():
            base = 0
        if isinstance(base,list):
            base = tuple(base)
        if isinstance(base,abstract_ZF):
            base = base.base
        if isinstance(base,tuple):
            op, *args = base  # Extract operator and operands
            new_args = []
            for arg in args:
                if isinstance(arg,abstract_ZF):
                    new_args += [arg.base]
                else:
                    new_args += [arg]
            args = new_args


        # Define sorting hierarchy: lower index = lower precedence
        type_hierarchy = {int: 0, float: 0, sp.Expr: 1, zeroFormAtom: 2, abstract_ZF: 3, tuple: 4}

        # Helper function to get the hierarchy value for sorting
        def hierarchy_rank(obj):
            th = type_hierarchy.get(type(obj), -1)
            if th<1:
                return (th,th)
            if th==1:
                return (th,th)
            elif th == 2:
                return (th,obj.label)
            else:
                return (th,th)

        def is_zero_check(x):
            """Helper function to check if x is zero."""
            return x == 0 or x == 0.0 or (hasattr(x, "is_zero") and x.is_zero)
        def is_one_check(x):
            """Helper function to check if x is zero."""
            return x == 1 or x == 1.0 or x==sp.sympify(1) or (hasattr(x, "is_one") and x.is_one)

        # If `base` is a tuple, process it
        if isinstance(base, tuple):
            op, *args = base  # Extract operator and operands
            if op == 'sub':
                if all(isinstance(j,(int,float,sp.Expr)) for j in args):
                    base = args[0]-args[1]
                elif args[0]==args[1]:
                    base = 0
                elif abstract_ZF(args[0]).is_zero:
                    base = ('mul',-1,args[1])
                elif abstract_ZF(args[1]).is_zero:
                    base = args[0]
            elif op == 'div':
                if all(isinstance(j,(int,float,sp.Expr)) for j in args):
                    base = sp.Rational(args[0],args[1])
                elif args[0]==args[1] and (not abstract_ZF(args[0]).is_zero):
                    base = 1
                elif abstract_ZF(args[0]).is_zero:
                    base = 0
            elif op in {"add", "mul"}:
                # Flatten nested structures (("add", ("add", x, y), z) --> ("add", x, y, z))
                flat_args = []
                for arg in args:
                    if isinstance(arg, abstract_ZF) and isinstance(arg.base, tuple) and arg.base[0] == op:
                        flat_args.extend(arg.base[1:])  # Expand nested elements
                    elif isinstance(arg, tuple) and arg[0] == op:
                        flat_args.extend(arg[1:])  # Expand nested elements
                    else:
                        flat_args.append(arg)

                # Sort operands by hierarchy
                flat_args.sort(key=hierarchy_rank)

                # Combine leading numeric terms (int, float, sp.Expr) into a single term
                numeric_terms = [arg for arg in flat_args if hierarchy_rank(arg)[0] < 2]
                other_terms = [arg for arg in flat_args if hierarchy_rank(arg)[0] >= 2]

                if op=="mul" and (any((j==0 or j==0.0) for j in numeric_terms) or any(j.is_zero for j in other_terms if not isinstance(j,tuple))):
                    base = 0
                else:
                    if op=="mul":
                        other_terms = [j for j in other_terms if isinstance(j,tuple) or not j.is_one]

                    if op == "add":
                        new_other_terms = {}

                        for term in other_terms:
                            if isinstance(term, abstract_ZF):
                                term = term.base  # Extract base representation

                            # Case 1: Standalone atomic term (e.g., A_low_1_2_hi_1)
                            if isinstance(term, zeroFormAtom):
                                new_other_terms[term] = new_other_terms.get(term, 0) + 1

                            # Case 2: Multiplication structure (e.g., ('mul', 1, A))
                            elif isinstance(term, tuple) and term[0] == "mul" and len(term) == 3 and isinstance(term[1], (int, float, sp.Expr)):
                                coeff, base_term = term[1], term[2]
                                new_other_terms[base_term] = new_other_terms.get(base_term, 0) + coeff

                            # Case 3: Any other term, store as-is
                            else:
                                new_other_terms[term] = new_other_terms.get(term, 0) + 1

                        # Reconstruct terms, applying simplifications
                        other_terms = []
                        for key, coeff in new_other_terms.items():
                            if coeff == 0:
                                continue  # Skip zero terms

                            if coeff == 1:
                                other_terms.append(key)
                            elif coeff == -1:
                                other_terms.append(("mul", -1, key))
                            else:
                                other_terms.append(("mul", coeff, key))

                    # Combine numeric terms into a single sum/prod and insert if nonzero
                    numeric_terms = [j for j in numeric_terms if j is not None]
                    if numeric_terms:
                        if op == "add":
                            combined_numeric = sp.sympify(sum(numeric_terms))
                            if combined_numeric != 0:
                                other_terms.insert(0, combined_numeric)
                            elif (combined_numeric==0 or combined_numeric==0.0) and len(other_terms)==0:
                                other_terms = [combined_numeric]
                        elif op == "mul":
                            combined_numeric = sp.prod(numeric_terms)
                            if combined_numeric == 0 or combined_numeric==0.0:
                                other_terms = [combined_numeric]
                            elif combined_numeric !=1 and combined_numeric!=1.0:
                                other_terms.insert(0, combined_numeric)
                            elif (combined_numeric==1 or combined_numeric==1.0) and len(other_terms)==0:
                                other_terms = [combined_numeric]

                    # Update base
                    if len(other_terms) > 1:
                        base = (op, *other_terms)
                    elif len(other_terms)==0:
                        base = 0
                    elif hasattr(other_terms[0],'base'):
                        base = other_terms[0].base
                    else:
                        base = other_terms[0]

            elif op == "pow":
                left, right = args

                if is_zero_check(right):
                    if not is_zero_check(left):
                        base = 1
                elif is_one_check(right):
                    base = left
                else:
                    if is_zero_check(left):
                        base = 0
                    elif is_one_check(left):
                        base = 1
                    elif isinstance(left,(int,float,sp.Expr)) and isinstance(right,(int,float,sp.Expr)):
                        base =left**right
                    elif isinstance(left, abstract_ZF) and isinstance(left.base, tuple) and left.base[0] == "pow":
                        inner_base, inner_exp = left.base[1], left.base[2]

                        base = ("pow", inner_base,  ( "mul", inner_exp, right))

            # Assign updated base
            if isinstance(base,(tuple,list)): 
                base = tuple([j.base if isinstance(j,abstract_ZF) else j for j in base])
            obj = super().__new__(cls, base)
            obj.base = base 
            return obj

        elif not isinstance(base, (zeroFormAtom, abstract_ZF, int, float, sp.Expr)):
            raise TypeError("Base must be zeroFormAtom, int, float, sympy.Expr, abstract_ZF, or an operation tuple.")

        # Call sp.Basic constructor
        obj = super().__new__(cls, base)
        obj.base = base
        return obj

    def __init__(self, base):
        self._is_zero = (self.base == 0 or self.base == 0.0 or
                        (isinstance(self.base, zeroFormAtom) and self.base.is_zero) or
                        (isinstance(self.base, abstract_ZF) and self.base.is_zero))

        self.is_one = (self.base == 1 or self.base == 1.0 or
                    (isinstance(self.base, zeroFormAtom) and self.base.is_one) or
                    (isinstance(self.base, abstract_ZF) and self.base.is_one))

    @property
    def is_zero(self):
        """Returns True if the expression simplifies to zero."""
        return self._is_zero 

    def __hash__(self):
        """
        Hash the abstract_ZF instance for use in sets and dicts.
        """
        return hash(self.base)

    def __eq__(self, other):
        """
        Check equality of two abstract_ZF instances.
        """
        if not isinstance(other, abstract_ZF):
            return NotImplemented
        return self.base == other.base

    def subs(self, data):
        """
        Symbolic substitution in abstract_ZF.
        """
        if isinstance(self.base,(int,float)):
            return self
        if isinstance(data, (list, tuple)) and all(isinstance(j, tuple) and len(j) == 2 for j in data):
            l1 = len(data)
            data = dict(data)
            if len(data) < l1:
                warnings.warn('Provided substitution rules had repeat keys, and only one was used.')
        if isinstance(self.base,sp.Expr):
            new_subs = dict()
            for k,v in data.items():
                if isinstance(k,sp.Expr):
                    new_subs[k] = v
            return abstract_ZF(self.base.subs(new_subs))
        if isinstance(self.base,zeroFormAtom):
            return abstract_ZF(self.base.subs(data))
        if isinstance(self.base,tuple):
            op,*args = self.base
            def sub_process(arg,sub_data):
                if isinstance(arg,sp.Expr):
                    new_subs = dict()
                    for k,v in sub_data.items():
                        if isinstance(k,sp.Expr):
                            new_subs[k] = v
                    return arg.subs(new_subs)
                if isinstance(arg, (zeroFormAtom,abstract_ZF)):
                    return arg.subs(sub_data)
                return arg
            new_base = (op,)+tuple([sub_process(arg,data) for arg in args])                
            return abstract_ZF(new_base)

    def _eval_conjugate(self):
        def recursive_conjugate(expr):
            if isinstance(expr,tuple):
                op, *args = expr
                return tuple([op]+[recursive_conjugate(arg) for arg in args])
            else:
                return _custom_conj(expr)
        return abstract_ZF(recursive_conjugate(self.base))


    def __add__(self, other):
        """
        Addition of abstract_ZF instances.
        Supports addition with int, float, and sympy.Expr.
        """
        if not isinstance(other, (abstract_ZF, int, float, sp.Expr, zeroFormAtom)):
            return NotImplemented
        if other == 0:
            return self
        return abstract_ZF(("add", self, other))

    def __radd__(self,other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction of abstract_ZF instances.
        Supports subtraction with int, float, and sympy.Expr.
        """
        if not isinstance(other, (abstract_ZF, int, float, sp.Expr)):
            return NotImplemented
        if other == 0:
            return self
        return abstract_ZF(("sub", self, other))

    def __mul__(self, other):
        """
        Multiplication of abstract_ZF instances.
        Supports multiplication with int, float, and sympy.Expr.
        """
        if isinstance(other, abstract_ZF):
            # If multiplying same base, add exponents (x^a * x^b --> x^(a + b))
            if (
                isinstance(self.base, tuple) and self.base[0] == "pow" and
                isinstance(other.base, tuple) and other.base[0] == "pow"
            ):
                base1, exp1 = self.base[1], self.base[2]
                base2, exp2 = other.base[1], other.base[2]
                if base1 == base2:
                    return abstract_ZF(("pow", base1, ("add", exp1, exp2)))  # x^(a+b)

            return abstract_ZF(("mul", self.base, other.base))  # Default multiplication for abstract_ZF instances

        elif isinstance(other, (int, float, sp.Expr)):
            if isinstance(self.base,tuple) and self.base[0]=='mul' and isinstance(self.base[1],(int,float,sp.Expr)):
                factors = tuple(['mul']+[other*f if count==0 else f for count,f in enumerate(self.base[1:])])
            else: 
                factors = ("mul", other, self.base)
            return abstract_ZF(factors)
        elif isinstance(other, zeroFormAtom):
            return abstract_ZF(("mul", self.base, other))

        return NotImplemented 

    def __rmul__(self, other):
        """
        Multiplication of abstract_ZF instances.
        Supports multiplication with int, float, and sympy.Expr.
        """
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        """
        Division of abstract_ZF instances.
        Supports division with int, float, and sympy.Expr.
        """
        if isinstance(other, (int, float, sp.Expr)):
            return abstract_ZF(("div", self, other))

        if not isinstance(other, abstract_ZF):
            return NotImplemented

        return abstract_ZF(("div", self, other))

    def __pow__(self, other):
        """
        Exponentiation of abstract_ZF instances.
        Supports exponentiation with int, float, and sympy.Expr.
        """
        if isinstance(other, (int, float, sp.Expr)):
            return abstract_ZF(("pow", self, other))

        if not isinstance(other, abstract_ZF):
            return NotImplemented

        return abstract_ZF(("pow", self, other))

    def _eval_simplify(self, ratio=None, measure=None, inverse=True, doit=True, rational=True, expand=False, **kwargs):
        """
        Simplifies the abstract_ZF instance using algebraic rules.
        """
        if isinstance(self.base, tuple):
            op, *args = self.base

            # Recursively simplify operands with the same keyword arguments
            args = [arg._eval_simplify(ratio=ratio, measure=measure, inverse=inverse, 
                                    doit=doit, rational=rational, expand=expand, **kwargs) 
                    if isinstance(arg, abstract_ZF) else arg for arg in args]

            # Apply `expand=True` to distribute multiplication over addition
            if expand:
                if op == "mul":
                    expanded_terms = []
                for term in args:
                    if isinstance(term, abstract_ZF) and isinstance(term.base, tuple) and term.base[0] == "add":
                        expanded_products = [
                            abstract_ZF(("mul", term2, *args[:i], *args[i+1:]))
                            for i, term2 in enumerate(term.base[1:])
                        ]
                        expanded_terms.append(abstract_ZF(("add", *expanded_products)))
                    elif isinstance(term, tuple) and term[0] == "add":
                        expanded_products = [
                            abstract_ZF(("mul", *args[:i], term2, *args[i+1:]))
                            for i, term2 in enumerate(term[1:])
                        ]
                        expanded_terms.append(abstract_ZF(("add", *expanded_products)))
                    else:
                        expanded_terms.append(term)
                    return abstract_ZF(("mul", *expanded_terms))

                if op == "pow":
                    base, exp = args
                    if isinstance(exp, int) and exp>0:
                        if isinstance(base, abstract_ZF) and isinstance(base.base, tuple) and base.base[0] == "add":
                            base = base.base
                        if isinstance(base, tuple) and base[0] == "add":
                            terms = base[1:]
                            expanded = [('mul',term) for term in terms]
                        for j in range(exp):
                            expanded = [prod+(term,) for prod in expanded for term in terms]

                        expanded = abstract_ZF(('sum',)+tuple(expanded))
                        return expanded._eval_simplify()
                    else:
                        return abstract_ZF(('pow',base,exp))

            # Use `ratio` to decide whether to factor (a * b + a * c --> a * (b + c))
            if op == "add" and ratio is not None:
                common_factors = set()
                for term in args:
                    if isinstance(term, abstract_ZF) and isinstance(term.base, tuple) and term.base[0] == "mul":
                        term = term.base
                    if isinstance(term, tuple) and term[0] == "mul":
                        factors = set(term[1:])
                        if not common_factors:
                            common_factors = factors
                        else:
                            common_factors &= factors  # Intersect common factors
                    elif isinstance(term,(int,float,sp.Expr,zeroFormAtom,abstract_ZF)):
                        if not common_factors:
                            common_factors = set([term])
                        else:
                            common_factors &= set([term])

                if common_factors and ratio > 0.1:  # Custom threshold for factoring
                    remaining_terms = []
                    for term in args:
                        if isinstance(term, abstract_ZF) and isinstance(term.base, tuple) and term.base[0] == "mul":
                            factors = set(term.base[1:])
                        elif isinstance(term, tuple) and term[0] == "mul":
                            factors = set(term[1:])
                        else:
                            factors = {term}

                        reduced_factors = factors - common_factors
                        remaining_terms.append(("mul", *reduced_factors) if reduced_factors else 1)

                    return abstract_ZF(("mul", *common_factors, ("add", *remaining_terms)))

            # Return simplified operation
            return abstract_ZF((op, *args))

        elif isinstance(self.base, sp.Expr):
            return abstract_ZF(self.base.simplify(ratio=ratio, measure=measure, rational=rational))

        return self  # Return unchanged if base is not an operation

    def __repr__(self):
        """
        Returns a detailed string representation showing the AST.
        """
        if isinstance(self.base, tuple):
            return f"abstract_ZF({self.base})"
        return f"abstract_ZF({repr(self.base)})"

    def __str__(self):
        """
        Returns a reading-friendly string representation of the expression.
        """
        if isinstance(self.base, tuple):
            op, *args = self.base

            def needs_parentheses(expr, context_op, position):
                """
                Determines whether an expression needs parentheses based on its operator.
                """
                expr_str = str(expr)
                if context_op == "mul" and ("+" in expr_str or "-" in expr_str):
                    if position == 0 and ('-' not in expr_str[1:] and '+' not in expr_str[1:]):
                        return False
                    return True  # Wrap sums inside products
                if context_op == "pow" and any(j in expr_str for j in {"+", "-", "*", "/"}) and position == 0:
                    return True  # base of expontents with sums/products/divs should be wrapped
                if context_op == "div" and position == 0 and (any(j in expr_str for j in {"+", "/"}) or '-' in expr_str[1:]):
                    return True  # base of expontents with sums/products/divs should be wrapped
                if context_op == "sub" and ("+" in expr_str or "-" in expr_str):
                    return True
                return False

            formatted_args = []
            for count, arg in enumerate(args):
                arg_str = str(arg)
                if needs_parentheses(arg, op, count):
                    arg_str = f"({arg_str})"
                formatted_args.append(arg_str)

            if op == "add":
                formatted_str = " + ".join(formatted_args)
                formatted_str = formatted_str.replace("+ -", "-")
            elif op == "sub":
                return " - ".join(formatted_args)
            elif op == "mul":
                return " * ".join(formatted_args)
            elif op == "div":
                return f"({formatted_args[0]}) / ({formatted_args[1]})"
            elif op == "pow":
                return f"{formatted_args[0]}**{formatted_args[1]}"

        return str(self.base)

    def _latex(self, printer=None):
        """
        Returns a LaTeX representation of the expression.
        """
        if isinstance(self.base, tuple):
            op, *args = self.base

            def needs_parentheses(expr, context_op, position):
                """
                Determines whether an expression needs parentheses based on its operator.
                """
                expr_str = str(expr)
                if context_op == "mul" and any(j in expr_str for j in {"+", "-","add","sub"}):
                    if position == 0 and ('-' not in expr_str[1:] and '+' not in expr_str[1:]):
                        return False
                    return True  # Wrap sums inside products
                if context_op == "pow" and any(j in expr_str for j in {"+", "-", "*", "/","add","sub","mul","div"}) and position == 0:
                    return True  # base of expontents with sums/products/divs should be wrapped
                if context_op == "div" and position == 0 and (any(j in expr_str for j in {"+", "/","add","div"}) or '-' in expr_str[1:] or 'sub' in expr_str[1:]):
                    return True  # base of expontents with sums/products/divs should be wrapped
                if context_op == "sub" and (j in expr_str for j in {"+", "-","add","sub"}):
                    return True
                return False

            formatted_args = []
            for count, arg in enumerate(args):
                if count==0 and op=='mul' and arg in {1,1.0, -1, -1.0}:
                    if arg in {1,1.0}:
                        if len(args)==1:
                            formatted_args.append('1')
                    elif arg in {-1,-1.0}:
                        if len(args)==1:
                            formatted_args.append('-1')
                        else:
                            formatted_args.append('-')
                elif op=='pow' and isinstance(args[1],sp.Rational) and all(isinstance(j,int) and j>0 for j in [args[1].numerator,args[1].denominator-1]):
                    op = '_handled'
                    if isinstance(arg,tuple):
                        arg_latex = f"{{{abstract_ZF(arg)._latex(printer=printer)}}}"
                    else:
                        arg_latex = f"{{{arg._latex(printer=printer)}}}" if hasattr(arg, "_latex") else sp.latex(arg)
                    if args[1].numerator==1:
                        if args[1].denominator==2:
                            formatted_str = f'\\sqrt{{{arg_latex}}}'
                        else:
                            formatted_str = f'\\sqrt[{args[1].denominator}]{{{arg_latex}}}'
                    else:
                        if args[1].denominator==2:
                            formatted_str = f'\\left(\\sqrt{{{arg_latex}}}\\right)^{{{args[1].numerator}}}'
                        else:
                            formatted_str = f'\\left(\\sqrt[{args[1].denominator}]{{{arg_latex}}}\\right)^{{{args[1].numerator}}}'
                else:
                    if isinstance(arg,tuple):
                        arg_latex = f"{{{abstract_ZF(arg)._latex(printer=printer)}}}"
                    else:
                        arg_latex = f"{{{arg._latex(printer=printer)}}}" if hasattr(arg, "_latex") else sp.latex(arg)
                    if needs_parentheses(arg, op, count):
                        arg_latex = f"\\left({arg_latex}\\right)"
                    formatted_args.append(arg_latex)

            if op == "add":
                formatted_str = " + ".join(formatted_args)
                formatted_str = formatted_str.replace("+ -", "-")
            elif op == "sub":
                formatted_str = " - ".join(formatted_args)
            elif op == "mul":
                formatted_str = " ".join(formatted_args)
            elif op == "div":
                formatted_str = f"\\frac{{{formatted_args[0]}}}{{{formatted_args[1]}}}"
            elif op == "pow":
                formatted_str = f"{formatted_args[0]}^{{{formatted_args[1]}}}"

            return formatted_str.replace("+ {\\left(-1\\right) ", "- { ").replace("+ -", "-").replace("+ {-", "-{")

        return sp.latex(self.base)

    def _repr_latex_(self):
        """
        Jupyter Notebook LaTeX representation for abstract_ZF.
        """
        return f"${sp.latex(self)}$"

class abstDFAtom():

    # def __new__(cls, coeff, degree, label=None, ext_deriv_order=None, _markers=frozenset()):
    #     # Type validations
    #     if not isinstance(coeff, (int, float, sp.Basic)):
    #         raise TypeError(f"Expected 'coeff' to be int, float, or sympy expression, got {type(coeff).__name__}")

    #     if not isinstance(degree, int):
    #         raise TypeError(f"Expected 'degree' to be an int, got {type(degree).__name__}")

    #     if label is not None and not isinstance(label, str):
    #         raise TypeError(f"Expected 'label' to be a str or None, got {type(label).__name__}")

    #     if ext_deriv_order is not None and not isinstance(ext_deriv_order, int):
    #         raise TypeError(f"Expected 'ext_deriv_order' to be an int or None, got {type(ext_deriv_order).__name__}")

    #     if not isinstance(_markers, (set, frozenset)):
    #         raise TypeError(f"Expected '_markers' to be a set or frozenset, got {type(_markers).__name__}")

    #     obj = super().__new__(cls, coeff, degree)
    #     return obj

    def __init__(self, coeff, degree, label=None, ext_deriv_order=None, _markers=frozenset()):
        if hasattr(coeff,'is_zero') and coeff.is_zero:
            coeff = 0
        elif hasattr(coeff,'is_one') and coeff.is_one:
            coeff = 1
        if isinstance(coeff,(int,float)):
            coeff = sp.sympify(coeff)
        self.coeff = coeff
        self.degree = degree
        self.label = label
        self.ext_deriv_order = ext_deriv_order
        self._markers=_markers

    def __eq__(self, other):
        """
        Check equality of two abstDFAtom instances.
        """
        if not isinstance(other, abstDFAtom):
            return NotImplemented
        return (
            self.coeff == other.coeff
            and self.degree == other.degree
            and self.label == other.label
            and self.ext_deriv_order == other.ext_deriv_order
        )

    def __hash__(self):
        """
        Hash the abstDFAtom instance based on its attributes.
        """
        return hash((self.coeff, self.degree, self.label, self.ext_deriv_order))

    def _eval_conjugate(self):
        if self.label:
            if "real" in self._markers:
                label = self.label
            elif self.label[0:3]=="BAR":
                label=self.label[3:]
            else:
                label=f"BAR{self.label}"
        else:
            label = None
        def cMarkers(marker):
            if marker == "holomorphic":
                return "antiholomorphic"
            if marker == "antiholomorphic":
                return "holomorphic"
            return marker
        new_markers = frozenset([cMarkers(j) for j in self._markers])
        coeff = _custom_conj(self.coeff)
        return abstDFAtom(coeff,self.degree,label=label,ext_deriv_order=self.ext_deriv_order,_markers=new_markers)

    def __repr__(self):
        """String representation for abstDFAtom."""
        def extDerFormat(string):
            if self.ext_deriv_order==1:
                return f'D({string})'
            elif self.ext_deriv_order is not None and self.ext_deriv_order>1:
                return f'D^{self.ext_deriv_order}({string})'
            else:
                return string
        if isinstance(self.coeff,(zeroFormAtom,abstract_ZF)):
            return extDerFormat(self.coeff.__repr__())
        coeff_sympy = sp.sympify(self.coeff)
        if len(coeff_sympy.free_symbols)==0 and self.ext_deriv_order is not None and self.ext_deriv_order>0:
            return 0
        if coeff_sympy == 1:
            return str(self.label) if self.label else "1"
        elif coeff_sympy == -1:
            return f"-{self.label}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = f"({coeff_sympy})" if len(coeff_sympy.as_ordered_terms()) > 1 else str(coeff_sympy)
            return extDerFormat(f"{coeff_str}{self.label}") if self.label else extDerFormat(coeff_str)

    def _latex(self, printer=None):
        """LaTeX representation for abstDFAtom."""
        def extDerFormat(string):
            if self.ext_deriv_order==1:
                return f'D\\left({string}\\right)'
            elif self.ext_deriv_order is not None and self.ext_deriv_order>1:
                return f'D^{self.ext_deriv_order}\\left({string}\\right)'
            else:
                return string
        def bar_labeling(label):
            if label[0:3]=="BAR":
                to_print =  process_basis_label(label[3:])
                if "_" in to_print:
                    return f"\\bar{{{to_print}".replace("_", "}^", 1)
                else:
                    return f"\\bar{{{to_print}}}"
            else:
                return process_basis_label(label).replace("_", "^", 1)
        if isinstance(self.coeff,(zeroFormAtom,abstract_ZF)):
            return extDerFormat(self.coeff._latex(printer=printer))
        coeff_sympy = sp.sympify(self.coeff)
        if len(coeff_sympy.free_symbols)==0 and self.ext_deriv_order is not None and self.ext_deriv_order>0:
            return 0
        if coeff_sympy == 1:
            return bar_labeling(self.label) if self.label else "1"
        elif coeff_sympy == -1:
            return f"-{bar_labeling(self.label)}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_latex = f"\\left({sp.latex(coeff_sympy)}\\right)" if len(coeff_sympy.as_ordered_terms()) > 1 else sp.latex(coeff_sympy)
            return extDerFormat(f"{coeff_latex}{bar_labeling(self.label)}") if self.label else coeff_latex

    def _repr_latex_(self):
        return f"${sp.latex(self)}$"

    def __str__(self):
        """Fallback to the string representation."""
        return self.__repr__()

    def has_common_factor(self, other):
        if not isinstance(other, (abstDFAtom, abstDFMonom)):
            return False

        if isinstance(other, abstDFAtom):
            # Special case: label None and degree 0
            if self.label is None and self.degree == 0:
                return other.label is None and other.degree == 0
            # Otherwise, compare labels
            return self.label == other.label

        elif isinstance(other, abstDFMonom):
            # Match against all factors in the monomial
            return any(self.has_common_factor(factor) for factor in other.factors_sorted)

        return False

    def _eval_simplify(self, ratio=None, measure=None, inverse=True, doit=True, rational=True, expand=False, **kwargs):
        return self

    def subs(self,subs_data):
        if isinstance(subs_data, (list, tuple)) and all(isinstance(j, tuple) and len(j) == 2 for j in subs_data):
            l1 = len(subs_data)
            subs_data = dict(subs_data)
            if len(subs_data) < l1:
                warnings.warn('Provided substitution rules had repeat keys, and only one was used.')

        if isinstance(self.coeff,(zeroFormAtom,abstract_ZF)):
            return abstDFAtom((self.coeff).subs(subs_data),self.degree,self.label,self.ext_deriv_order,_markers=self._markers)
        if isinstance(self.coeff,sp.Expr):
            new_subs = dict()
            for k,v in subs_data.items():
                if isinstance(k,sp.Expr):
                    new_subs[k] = v
            return abstDFAtom((self.coeff).subs(new_subs),self.degree,self.label,self.ext_deriv_order,_markers=self._markers)
        return self

    def __mul__(self, other):
        """Handle left multiplication."""
        if isinstance(other, abstDFAtom):
            # Combine two atoms into a monomial product
            return abstDFMonom([self, other])
        elif isinstance(other, abstDFMonom):
            # Prepend this atom as a factor to the monomial's factors
            return abstDFMonom([self] + other.factors_sorted)
        elif isinstance(other, (int, float, sp.Expr, zeroFormAtom, abstract_ZF)):
            return abstDFAtom(self.coeff * other, self.degree, self.label, _markers=self._markers)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Handle right multiplication."""
        if isinstance(other, abstDFMonom):
            # Append this atom as a factor to the monomial
            return abstDFMonom(other.factors_sorted + [self])
        elif isinstance(other, (int, float, sp.Expr, zeroFormAtom, abstract_ZF)):
            return abstDFAtom(self.coeff * other, self.degree, self.label,_markers=self._markers)
        else:
            return NotImplemented

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """Addition with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([abstDFMonom([self]), other])
        elif isinstance(other, zeroFormAtom):
            return abstract_DF([abstDFMonom([self]), abstDFMonom([abstDFAtom(other,0,_marker=other._markers)])])
        elif isinstance(other, abstract_ZF):
            return abstract_DF([abstDFMonom([self]), abstDFMonom([abstDFAtom(other,0)])])
        elif isinstance(other, abstract_DF):
            return abstract_DF((abstDFMonom([self]),) + tuple(other.terms))
        else:
            raise TypeError(f"Unsupported operand type {type(other)} for + with `abstDFAtom`")

    def __sub__(self, other):
        """Subtraction with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([abstDFMonom([self]), -1 * other])
        elif isinstance(other, abstract_DF):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_DF([abstDFMonom([self])] + negated_terms)
        elif isinstance(other, (zeroFormAtom,abstract_ZF)):
            return abstract_DF([abstDFMonom([self]), abstDFMonom([abstDFAtom(-1*other,0)])])
        else:
            raise TypeError(f"Unsupported operand type for - with `abstDFAtom`: {type(other)}")

    def __lt__(self, other):
        """
        Lexicographic comparison: First by degree, then by label.
        """
        if not isinstance(other, abstDFAtom):
            return NotImplemented

        # Primary comparison: degree
        if self.degree != other.degree:
            return self.degree < other.degree

        # Secondary comparison: label (None precedes any string)
        if self.label is None:
            return True
        if other.label is None:
            return False
        return self.label < other.label

class abstDFMonom(sp.Basic):
    def __new__(cls, factors):
        if not isinstance(factors, (list, tuple)):
            raise TypeError('`abstDFMonom` expects `factors` to be a list or tuple')
        if not all(isinstance(elem, abstDFAtom) for elem in factors):
            raise TypeError('`abstDFMonom` expects `factors` to be a list of `abstDFAtom`')

        return sp.Basic.__new__(cls, *factors)

    def __init__(self, factors):

        class DegreeLabelSortable:
            def __init__(self, atom):
                self.degree = atom.degree
                self.label = atom.label

            def __lt__(self, other):
                if self.degree != other.degree:
                    return self.degree < other.degree
                if self.label is None:
                    return True
                if other.label is None:
                    return False
                return self.label < other.label

            def __eq__(self, other):
                return self.degree == other.degree and self.label == other.label

            def __le__(self, other):
                return self < other or self == other

        weighted_objs = [SortableObj((j, (j.degree,barSortedStr(j.label)))) for j in factors]
        parity, objs_sorted, _ = weightedPermSign(
            weighted_objs, [DegreeLabelSortable(j) for j in factors], returnSorted=True, use_degree_attribute=True
        )

        if parity == -1:
            objs_sorted = [SortableObj((abstDFAtom(-1, 0), (0,None)))] + objs_sorted

        coeffFactor = 1
        new_objs = []

        for j in objs_sorted:
            if coeffFactor!=0:
                coeffFactor = coeffFactor * j.value.coeff if j.value.coeff else 0
            if j.place[0] != 0:
                new_objs.append(abstDFAtom(1, j.value.degree, j.value.label,j.value.ext_deriv_order,j.value._markers))

        consolidated_factor = abstDFAtom(coeffFactor, 0) ### check !!!
        if coeffFactor==0:
            self.factors_sorted = [abstDFAtom(0,0)]
            self._coeff = 0
        else:
            self.factors_sorted = [consolidated_factor] + new_objs
            self._coeff = coeffFactor
        if len(self.factors_sorted)!=len(set(self.factors_sorted)) or len(self.factors_sorted)==0:
            self.factors_sorted = [abstDFAtom(0,0)]
        self.factors = factors
        self.str_ids = tuple(
            "<Coeff>" if i == 0 and factor.label is None else (factor.label if factor.label is not None else "<None>")
            for i, factor in enumerate(self.factors_sorted)
        )
        self.degree = sum(factor.degree for factor in self.factors if factor.coeff!=0)

    def __eq__(self, other):
        """
        Check equality of two abstDFMonom instances.
        """
        if not isinstance(other, abstDFMonom):
            return NotImplemented
        return self.factors_sorted == other.factors_sorted and self.degree == other.degree

    def __hash__(self):
        """
        Hash the abstDFMonom instance based on sorted factors.
        """
        return hash((tuple(self.factors_sorted), self.degree))

    @property
    def is_zero(self):
        return self._coeff==0
    def _eval_conjugate(self):
        return abstDFMonom([j._eval_conjugate() for j in self.factors])

    def _eval_simplify(self, ratio=None, measure=None, inverse=True, doit=True, rational=True, expand=False, **kwargs):
        return abstDFMonom([j._eval_simplify() for j in self.factors])

    def subs(self,subs_data):
        return abstDFMonom([j.subs(subs_data) for j in self.factors_sorted])

    def _repr_latex_(self):
        """
        Define how the abstDFMonom is displayed in LaTeX in IPython (Jupyter Notebook).
        """
        return f"${sp.latex(self)}$"

    def _latex(self, printer=None):
        """
        LaTeX representation
        """
        # Handle the leading degree 0 factor (coefficient)
        coeff0 = self.factors_sorted[0]
        if isinstance(coeff0,abstDFAtom):
            coeff_inner = coeff0.coeff
        else:
            coeff_inner = coeff0
        if isinstance(coeff_inner,zeroFormAtom):
            if coeff_inner.is_one:
                coeff_latex = ''
            elif ((-1)*coeff_inner).is_one:
                coeff_latex = '-'
            elif coeff_inner.is_zero:
                coeff_latex = '0'
            else:
                coeff_latex = sp.latex(coeff0)
        elif isinstance(coeff_inner,abstract_ZF):
            if coeff_inner.is_one:
                coeff_latex = ''
            elif (-1*coeff_inner).is_one:
                coeff_latex = '-'
            elif coeff_inner.is_zero:
                coeff_latex = '0'
            else:
                coeff_latex = sp.latex(coeff0)
                if isinstance(coeff_inner.base,tuple) and coeff_inner.base[0] in {'sub','add'}:
                    coeff_latex = f'\\left({coeff_latex}\\right)'
        else:
            if coeff_inner==1 or coeff_inner==1.0:
                coeff_latex = ''
            elif coeff_inner==-1 or coeff_inner==-1.0:
                coeff_latex ='-'
            elif coeff_inner==0 or coeff_inner==0.0:
                coeff_latex = '0'
            else:
                coeff_latex = sp.latex(coeff0)

        # Join the remaining factors using '\wedge'
        if len(self.factors_sorted) > 1:
            # Generate LaTeX for all non-coefficient factors
            factors_latex = " \\wedge ".join(factor._latex(printer=printer) for factor in self.factors_sorted[1:])

            # Combine coefficient and factors, but omit '\cdot' if coeff_latex is empty or "-"
            if coeff_latex in ["", "-"]:
                return f"{coeff_latex}{factors_latex}"
            else:
                return f"{coeff_latex} \\cdot {factors_latex}"
        else:
            # Only the degree 0 factor (no other factors to join)
            if coeff_latex == "":
                return '1'
            elif coeff_latex == "-":
                return '-1'
            return coeff_latex

    def __repr__(self):
        """
        String representation for abstDFMonom.
        """
        # Handle the coefficient (first factor)
        coeff0 = self.factors_sorted[0]
        if isinstance(coeff0,abstDFAtom):
            coeff_inner = coeff0.coeff
        else:
            coeff_inner = coeff0
        if isinstance(coeff_inner,zeroFormAtom):
            coeff_str = coeff0.__str__()
        elif isinstance(coeff_inner,abstract_ZF):
            coeff_str = coeff0.__str__()
            if isinstance(coeff_inner.base,tuple) and coeff_inner.base[0] in {'sub','add'}:
                coeff_str = f'({coeff_str})'
        else:
            coeff_str = str(coeff0)
        # Join the other factors using '*'
        if len(self.factors_sorted) > 1:
            factors_str = "*".join(str(factor) for factor in self.factors_sorted[1:])

            # Combine coefficient and factors, but omit '*' if coeff_str is empty or "-"
            if coeff_str in ["", "-"]:
                return f"{coeff_str}{factors_str}"
            else:
                return f"{coeff_str}*{factors_str}"
        else:
            # Only the degree 0 factor (no other factors to join)
            return coeff_str

    def __str__(self):
        """Fallback to the string representation."""
        return self.__repr__()

    def __mul__(self, other):
        """Handle left multiplication."""
        if isinstance(other, abstDFMonom):
            # Combine factors of both monomials
            return abstDFMonom(self.factors_sorted + other.factors_sorted)
        elif isinstance(other, abstDFAtom):
            # Append the atom as a factor to this monomial
            return abstDFMonom(self.factors_sorted + [other])
        elif isinstance(other, (int, float, sp.Expr,abstract_ZF,zeroFormAtom)):
            # Scalar multiplication (prepend as an atom with degree 0)
            other_sympy = sp.sympify(other)
            return abstDFMonom([abstDFAtom(other_sympy, 0)] + self.factors_sorted)
        elif isinstance(other, (abstract_ZF,zeroFormAtom)):
            return abstDFMonom([abstDFAtom(other, 0)] + self.factors_sorted)
        else:
            # Allow Python to try __rmul__ of the other operand
            return NotImplemented

    def __rmul__(self, other):
        """Handle right multiplication (symmetrically supports scalar * abstDFMonom)."""
        if isinstance(other, (int, float, sp.Expr)):
            other_sympy = sp.sympify(other)
            return abstDFMonom([abstDFAtom(other_sympy, 0)] + self.factors_sorted)
        elif isinstance(other, (abstract_ZF,zeroFormAtom)):
            return abstDFMonom([abstDFAtom(other, 0)] + self.factors_sorted)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'abstDFMonom'")

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """Addition with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([self, other])
        elif isinstance(other, abstract_DF):
            return abstract_DF([self] + other.terms)
        elif isinstance(other, (abstract_ZF,zeroFormAtom)):
            return abstract_DF([self, abstDFAtom(other, 0)])
        elif isinstance(other, (int, float, sp.Expr)):
            other_sympy = sp.sympify(other)
            return abstract_DF([self, abstDFAtom(other_sympy, 0)])
        else:
            raise TypeError("Unsupported operand type for + with `abstDFMonom`")

    def __sub__(self, other):
        """Subtraction with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([self, -1 * other])
        elif isinstance(other, abstract_DF):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_DF([self] + negated_terms)
        elif isinstance(other, (abstract_ZF,zeroFormAtom)):
            return abstract_DF([self, abstDFAtom(-1*other, 0)])
        elif isinstance(other, (int, float, sp.Expr)):
            other_sympy = -1*sp.sympify(other)
            return abstract_DF([self, abstDFAtom(other_sympy, 0)])
        else:
            raise TypeError("Unsupported operand type for - with `abstDFMonom`")

class abstract_DF(sp.Basic):
    def __new__(cls, terms):
        # Validate terms input
        if not isinstance(terms, (list, tuple)):
            raise TypeError('`abstract_DF` expects `terms` to be a list or tuple')
        if not all(isinstance(elem, (abstDFMonom, abstDFAtom)) for elem in terms):
            raise TypeError('`abstract_DF` expects `terms` to be a list of `abstDFMonom` or `abstDFAtom`')
        return super().__new__(cls, *terms)

    def __init__(self, terms):
        """
        Initialize abstract_DF with a simplified list of terms.
        """
        def process_abstDF(elem):
            """
            Check that all elements are abstDFMonom/Atom instances.
            """
            if isinstance(elem, abstDFMonom):
                return elem
            elif isinstance(elem, abstDFAtom):
                return abstDFMonom([elem])
            else:
                raise TypeError("`abstract_DF` initializer expects `abstDFMonom` or `abstDFAtom` instances")

        # Process terms into abstDFMonom instances
        processed_terms = [process_abstDF(term) for term in terms if not process_abstDF(term).is_zero]
        # Handle empty terms: default to trivial zero form
        if not terms:
            terms = [abstDFAtom(0, 0)]

        # Simplify terms by combining like terms and removing zeros
        collected_terms = tuple(self.simplify_terms(processed_terms))
        if len(collected_terms)==0:
            collected_terms=(abstDFMonom([abstDFAtom(0,0)]),)
        self.terms = collected_terms
        self.degree = self.terms[0].degree if all(j.degree == self.terms[0].degree for j in self.terms) else None

    def simplify_terms(self, terms):
        """
        Simplify a list of terms by combining like terms and removing zero terms.
        Terms with "<None>" in their `str_ids` will not be combined.
        """
        term_dict = {}

        for term in terms:
            # Use str_ids as the key for grouping like terms
            key = tuple(term.str_ids)

            # Skip combining terms that contain "<None>" in their key
            if "<None>" in key:
                # Treat these terms individually
                term_dict[id(term)] = {"coeff": term.factors_sorted[0].coeff, "tail": term.factors_sorted[1:]}
                continue

            # Extract the leading coefficient and trailing factors
            coeff = term.factors_sorted[0].coeff  # Leading coefficient
            tail = term.factors_sorted[1:]       # Trailing factors (actual abstDFAtom instances)

            # Combine coefficients for like terms
            if key in term_dict:
                term_dict[key]["coeff"] = coeff + term_dict[key]["coeff"]
            else:
                term_dict[key] = {"coeff": coeff, "tail": tail}

        # Rebuild simplified terms list
        simplified_terms = [
            abstDFMonom([abstDFAtom(data["coeff"], 0)] + data["tail"])
            for key, data in term_dict.items() if data["coeff"] != 0
        ]

        if not simplified_terms:
            return [abstDFMonom([abstDFAtom(0, 0)])]

        # Return the simplified, sorted terms
        return tuple(sorted(simplified_terms, key=lambda t: t.factors_sorted))

    def __eq__(self, other):
        """
        Check equality of two abstract_DF instances.
        """
        if not isinstance(other, abstract_DF):
            return NotImplemented
        return self.terms == other.terms and self.degree == other.degree

    def __hash__(self):
        """
        Hash the abstract_DF instance based on its terms.
        """
        return hash((self.terms,self.degree))

    def _eval_conjugate(self):
        return abstract_DF([j._eval_conjugate() for j in self.terms])

    def _eval_simplify(self, ratio=None, measure=None, inverse=True, doit=True, rational=True, expand=False, **kwargs):
        return abstract_DF([j._eval_simplify() for j in self.terms])

    def subs(self,subs_data):
        return abstract_DF([j.subs(subs_data) for j in self.terms])

    def __add__(self, other):
        if isinstance(other, (abstract_ZF,zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, (int, float, sp.Expr)):
            other = abstDFAtom(sp.sympify(other), 0)
        if other is None:
            return self
        elif isinstance(other, abstract_DF):
            return abstract_DF(self.terms + other.terms)
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF(self.terms + (other,))
        else:
            raise TypeError("Unsupported operand type for + with `abstract_DF`")

    def __sub__(self, other):
        if isinstance(other, (abstract_ZF,zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, (int, float, sp.Expr)):
            other = abstDFAtom(sp.sympify(other), 0)
        if other is None:
            return self
        elif isinstance(other, abstract_DF):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_DF(self.terms + negated_terms)
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return self + (-1 * other)
        else:
            raise TypeError("Unsupported operand type for - with `abstract_DF`")

    def __mul__(self, other):
        if isinstance(other, (abstract_ZF,zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, (int, float, sp.Expr)):
            other = abstDFAtom(sp.sympify(other), 0)
        if isinstance(other, (int, float, sp.Expr)):
            # Scalar multiplication
            return abstract_DF([term * other for term in self.terms])
        if isinstance(other, abstract_DF):
            # Distribute over terms
            new_terms = [t1 * t2 for t1 in self.terms for t2 in other.terms]
            return abstract_DF(new_terms)
        if isinstance(other, abstDFAtom):
            other = abstDFMonom([other])
        if isinstance(other, abstDFMonom):
            # Multiply each term by the monomial
            return abstract_DF([term * other for term in self.terms])
        else:
            NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (abstract_ZF,zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, (int, float, sp.Expr)):
            other = abstDFAtom(sp.sympify(other), 0)
        if isinstance(other, (int, float, sp.Expr)):
            return self * other
        if isinstance(other, abstDFAtom):
            other = abstDFMonom([other])
        if isinstance(other, abstDFMonom):
            other = abstract_DF([other])
        if isinstance(other, abstract_DF):
            return other.__mul__(self)
        else:
            NotImplemented

    def __neg__(self):
        return -1 * self

    def __repr__(self):
        """String representation for abstract_DF."""
        if len(self.terms) == 1:
            return repr(self.terms[0])

        # Build the string for terms
        terms_repr = [repr(term) for term in self.terms]

        result = terms_repr[0] if len(terms_repr)>0 else ""
        for term in terms_repr[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"  # Add space before "-" and strip the leading "-"
            else:
                result += f" + {term}"      # Add "+" before positive terms
        return result

    def _latex(self, printer=None):
        """LaTeX representation for SymPy's LaTeX printer."""
        if len(self.terms) == 1:
            return self.terms[0]._latex(printer=printer)

        # Build the LaTeX string for terms
        terms_latex = [term._latex(printer=printer) for term in self.terms]

        result = terms_latex[0] if len(terms_latex)>0 else ""
        for term in terms_latex[1:]:
            if term.startswith("-"):
                result += " - " + term[1:]  # Add space before "-" and strip the leading "-"
            else:
                result +=  " + " + term      # Add "+" before positive terms
        return result

    def _repr_latex_(self):
        return f"${sp.latex(self)}$"

    def __str__(self):
        return self.__repr__()


class abst_coframe(sp.Basic):
    def __new__(cls, coframe_basis, structure_equations, min_conj_rules={}):
        """
        Create a new abst_coframe instance.

        Parameters:
        ==========
            - structure_equations (dict): A dictionary where keys are abstDFAtom instances (representing 1-forms), and values are either abstract_DF instances (representing the differential) or None.
        """

        # Validate basis
        if not isinstance(coframe_basis,(list,tuple)):
            raise TypeError(f"Given `coframe_basis` must be a list/tuple. Instead recieved type {type(coframe_basis)}")
        if len(coframe_basis)!= len(set(coframe_basis)):
            counter = Counter(coframe_basis)
            repeated_elements = [item for item, count in counter.items() if count > 1]
            raise TypeError(f"`coframe_basis` should not have repeated elements\nRepeated elements: {repeated_elements}")
        # Validate basis elements
        for df in coframe_basis:
            if not isinstance(df,abstDFAtom):
                raise TypeError(f"Given `coframe_basis` should contain only `abstDFAtom` instances. Instead recieved {type(df)}")

        # Validate structure_equations input
        if not isinstance(structure_equations, dict):
            raise TypeError("structure_equations must be a dictionary.")

        # Validate keys and values
        for key, value in structure_equations.items():
            if key not in coframe_basis:
                raise TypeError(f"Keys in `structure_equations` dict must be also be in `coframe_basis`.\nErroneus key recieved{key}")
            if value is not None and not isinstance(value, abstract_DF):
                raise TypeError("Values in structure_equations dict must be abstract_DF instances or None.")


        forms_tuple = tuple(coframe_basis)
        #### the following may be a useful alternative for resolving ordering ambiguity
        # forms_tuple = tuple(sorted(structure_equations.keys(), key=lambda x: (x.degree, x.label)))

        conj_atoms = list(min_conj_rules.keys())+list(min_conj_rules.values())
        if len(conj_atoms)!=len(set(conj_atoms)) or not all(isinstance(j,int) for j in conj_atoms) or not all (j in range(len(forms_tuple)) for j in conj_atoms):
            raise ValueError(f'`conj_rules` should be an invertible dict containing `int` indices in the range 0 to {len(forms_tuple)}')
        # Create the instance
        obj = super().__new__(cls, forms_tuple)
        obj.forms = forms_tuple  # Immutable tuple of keys
        obj.structure_equations = structure_equations  # Original dictionary (mutable)
        obj.min_conj_rules = min_conj_rules
        obj.inverted_conj_rules = {v: k for k, v in min_conj_rules.items()}
        obj.conj_rules = min_conj_rules | {v: k for k, v in min_conj_rules.items()} | {j:j for j in range(len(forms_tuple)) if j not in conj_atoms}
        return obj

    def __init__(self, *args, **kwargs):
        """
        Override the default __init__ to do nothing.
        Initialization is fully handled in __new__.
        """
        pass

    def __eq__(self, other):
        """
        Check equality of two abst_coframe instances.
        """
        if not isinstance(other, abst_coframe):
            return NotImplemented
        return self.forms == other.forms

    def __hash__(self):
        """
        Hash the abst_coframe instance based on its immutable tuple of forms.
        """
        return hash(self.forms)

    def __repr__(self):
        """
        String representation of the coframe as a list of 1-forms.
        """
        return f"abst_coframe({', '.join(map(str, self.forms))})"

    def _latex(self, printer=None):
        """
        LaTeX representation of the coframe as a list of 1-forms.
        """
        return r"\{" + r", ".join(form._latex(printer=printer) for form in self.forms) + r"\}"

    def update_structure_equations(self, replace_symbols = {}, replace_eqns = {}, simplify=True):
        """
        Update the structure equation for a specific form.

        Parameters:
        - form: An abstDFAtom instance representing the 1-form to update.
        - equation: An abstract_DF instance or None representing the new structure equation.
        """
        if not (isinstance(replace_symbols,dict) and isinstance(replace_eqns,dict)):
            raise TypeError('If specified, `replace_symbols` and `replace_eqns` should be `dict` type')
        for key, value in self.structure_equations.items():
            if hasattr(value, 'subs') and callable(getattr(value, 'subs')):
                if simplify:
                    self.structure_equations[key]=sp.simplify(value.subs(replace_symbols))
                else:
                    self.structure_equations[key]=value.subs(replace_symbols)
        for key, value in replace_eqns.items():
            if key in self.forms:
                self.structure_equations[key]=value
        if simplify:
            for key, value in self.structure_equations.items():
                if isinstance(value,abstDFAtom):
                    self.structure_equations[key]=key._eval_simplify(value)
                else:
                    self.structure_equations[key]=sp.simplify(value)

def create_coframe(label, coframe_labels, str_eqns=None, str_eqns_labels=None, complete_to_complex_cf = None,  integrable_complex_struct=False, markers=dict(),remove_guardrails=False):
    """
    Create a coframe with specified 1-forms and structure equations.

    Parameters:
    - label (str): The name of the coframe.
    - coframe_labels (list of str): Labels for 1-forms in the coframe.
    - str_eqns (dict, optional): Pre-populated structure equations, keyed by (i, j, k) tuples.
    - str_eqns_labels (str, optional): Prefix for generating labels for missing terms in str_eqns.
    - markers (dict, optional): a dict whose key are strings from `coframe_labels`, and values of sets of properties associated with each coframe element
    - complete_to_complex_cf (any, optional): builds a larger coframe by include complex conjugate duals of coframe elements marked as holomorphic/antiholomorphic
    - remove_guardrails (bool, optional): Pass to validate_label for customization.
    """

    # Initialize str_eqns and conjugation dict
    if str_eqns is None:
        str_eqns = {}
    min_conj_rules = {}

    coframe_labels = list(coframe_labels)

    if complete_to_complex_cf is None:
        complete_to_complex_cf = ['standard']*len(coframe_labels)
    elif complete_to_complex_cf is True:
        complete_to_complex_cf = []
        for coVec in coframe_labels:
            if 'holomorphic' in markers.get(coVec,{}):
                complete_to_complex_cf += ['holomorphic']
                if 'antiholomorphic' in markers.get(coVec,{}):
                    warnings.warn('A coframe label was given to `create_coframe` with conflicting property markers \"holomorphic\" and \"antiholomorphic\". The coframe was created assuming \"holomorphic\" is correct')
                if 'real' in markers.get(coVec,{}):
                    warnings.warn('A coframe label was given to `create_coframe` with conflicting property markers \"holomorphic\" and \"real\". The coframe was created assuming \"holomorphic\" is correct')
            elif 'antiholomorphic' in markers.get(coVec,{}):
                complete_to_complex_cf += ['antiholomorphic']
                if 'real' in markers.get(coVec,{}):
                    warnings.warn('A coframe label was given to `create_coframe` with conflicting property markers \"antiholomorphic\" and \"real\". The coframe was created assuming \"antiholomorphic\" is correct')
            elif 'real' in markers.get(coVec,{}):
                complete_to_complex_cf += ['real']
            else:
                complete_to_complex_cf += ['standard']
    elif complete_to_complex_cf=="fromHol":
        complete_to_complex_cf = ['holomorphic']*len(coframe_labels)
    elif complete_to_complex_cf=="fromAntihol":
        complete_to_complex_cf = ['antiholomorphic']*len(coframe_labels)
    elif not isinstance(complete_to_complex_cf,(list,tuple)) or len(complete_to_complex_cf)!=len(coframe_labels):
        warnings.warn('`create_coframe` was given an unexpected value for `complete_to_complex_cf`. Proceeding as if `complete_to_complex_cf=None`.')
        complete_to_complex_cf = ['standard']*len(coframe_labels)

    closed_assumptions = [next(iter(markers[j].intersection({'closed'})),None) if j in markers else None for j in coframe_labels]

    # Create abstDFAtom instances for coframe labels
    elem_list = []
    conjugates_list = []
    conjugates_labels = []
    augments_counter = 0
    for count, coframe_label in enumerate(coframe_labels):
        if complete_to_complex_cf[count]=="real":
            coframe_label = validate_label(coframe_label, remove_guardrails=remove_guardrails)
            elem = abstDFAtom(1, 1, label=coframe_label, _markers=frozenset(["real"]))
            _cached_caller_globals[coframe_label] = elem
            elem_list.append(elem)
        elif complete_to_complex_cf[count]=="holomorphic" or complete_to_complex_cf[count]=="antiholomorphic":
            if coframe_label[0:3]=="BAR":
                coframe_label = f"BAR{validate_label(coframe_label[3:], remove_guardrails=remove_guardrails)}"
            else:
                coframe_label = validate_label(coframe_label, remove_guardrails=remove_guardrails)
            elem = abstDFAtom(1, 1, label=coframe_label, _markers=frozenset([complete_to_complex_cf[count]]))
            cElem = _custom_conj(elem)
            _cached_caller_globals[coframe_label] = elem
            _cached_caller_globals[cElem.label] = cElem
            elem_list.append(elem)
            conjugates_list.append(cElem)
            conjugates_labels.append(cElem.label)
            min_conj_rules = min_conj_rules | {count:len(coframe_labels)+augments_counter}
            augments_counter += 1
        else:
            coframe_label = validate_label(coframe_label, remove_guardrails=remove_guardrails)
            elem = abstDFAtom(1, 1, label=coframe_label)
            _cached_caller_globals[coframe_label] = elem
            elem_list.append(elem)

    init_dimension = len(coframe_labels)
    coframe_labels+=conjugates_labels
    elem_list+=conjugates_list

    # Build the coframe_dict
    coframe_dict = {elem:None for elem in elem_list}

    # Register the coframe in the caller's globals
    coframe = abst_coframe(elem_list,coframe_dict,min_conj_rules)
    label = validate_label(label, remove_guardrails=remove_guardrails)
    _cached_caller_globals[label] = coframe

    # Populate missing terms in str_eqns
    coeff_labels = []
    if str_eqns_labels is not None:
        if integrable_complex_struct:
            first_index_bound = init_dimension
        else:
            first_index_bound = len(coframe_labels)
        for i in range(first_index_bound):
            for j in range(i + 1, len(coframe_labels)):
                for k in range(init_dimension):
                    if (i, j, k) not in str_eqns or str_eqns[(i, j, k)] is None:
                        scale = 0 if closed_assumptions[k]=='closed' else 1
                        # Generate and validate the coefficient label
                        if str_eqns_labels not in _cached_caller_globals:
                            _cached_caller_globals[str_eqns_labels] = tuple()
                        coeff_label = f"{str_eqns_labels}_low_{i+1}_{j+1}_hi_{k+1}"
                        coeff_label = validate_label(coeff_label, remove_guardrails=remove_guardrails)

                        # Create a zeroFormAtom and register it
                        _cached_caller_globals[coeff_label] = zeroFormAtom(label=coeff_label,coframe=_cached_caller_globals[label])
                        _cached_caller_globals[str_eqns_labels] += (_cached_caller_globals[coeff_label],)
                        coeff_labels.append(coeff_label)

                        # Update str_eqns
                        str_eqns[(i, j, k)] = abstDFAtom(_cached_caller_globals[coeff_label], 0)*scale

    else:
        # Fill missing terms with None
        for i in range(len(coframe_labels)):
            for j in range(i + 1, len(coframe_labels)):
                for k in range(len(coframe_labels)):
                    if (i, j, k) not in str_eqns:
                        str_eqns[(i, j, k)] = None


    # Update coframe!!!!
    update_dict = {}
    for k in range(init_dimension):
        kth_term_list = []
        for i in range(len(coframe_labels)):
            for j in range(i + 1, len(coframe_labels)):
                if (i, j, k) in str_eqns and str_eqns[(i, j, k)] is not None:
                    kth_term_list.append(str_eqns[(i, j, k)] * elem_list[i] * elem_list[j])
        if kth_term_list:
            update_dict[elem_list[k]] = abstract_DF(kth_term_list)
    inv_dict = {v:k for k,v in min_conj_rules.items()}
    for v in range(init_dimension,len(coframe_labels)):
        if update_dict[elem_list[inv_dict[v]]]:
            update_dict[elem_list[v]] = update_dict[elem_list[inv_dict[v]]]._eval_conjugate()

    _cached_caller_globals[label].update_structure_equations(replace_eqns=update_dict)

    if init_dimension<len(coframe_labels):
        BAR_str_eqns_labels = str_eqns_labels[3:] if str_eqns_labels[:3]=='BAR' else 'BAR'+str_eqns_labels
        barVars = []
        for j in _cached_caller_globals[str_eqns_labels]:
            conj_j = _custom_conj(j)
            _cached_caller_globals[conj_j.label]=conj_j
            barVars += [conj_j]
        _cached_caller_globals[BAR_str_eqns_labels]=tuple(barVars)

    # Add the labels to the variable registry
    vr = get_variable_registry()
    vr["misc"][label] ={"children": coframe_labels, "cousins": coeff_labels}


def coframe_derivative(df, coframe, cfIndex):
    """
    Compute the coframe derivative of an expression with respect to `coframe.forms[cfIndex]`.

    Parameters:
    - df: A `zeroFormAtom` or `abstract_ZF` instance.
    - coframe: The coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - The coframe derivative of `df`.
    """
    if not isinstance(cfIndex, int) or cfIndex < 0:
        raise ValueError("`cfIndex` must be a non-negative integer.")
    if cfIndex >= len(coframe.forms):
        raise IndexError(f"`cfIndex` {cfIndex} is out of bounds for coframe with {len(coframe.forms)} forms.")

    if isinstance(df, zeroFormAtom):
        return _cofrDer_zeroFormAtom(df, coframe, cfIndex)
    elif isinstance(df, abstract_ZF):
        return _cofrDer_abstract_ZF(df, coframe, cfIndex)
    else:
        raise TypeError(f"`coframe_derivative` does not support type `{type(df).__name__}`.")

def extDer(df, coframe=None, order=1):
    """
    Exterior derivative operator `extDer()` for various differential forms.

    Parameters:
    - df: The differential form (`zeroFormAtom`,
          `abstDFAtom`, `abstDFMonom`, or `abstract_DF`).
    - coframe: Optional `abst_coframe` object representing the coframe.
    - order: Optional positive integer, denoting the number of times `extDer` is applied.

    Returns:
    - The exterior derivative of the form.
    """
    if coframe is None:
        raise KeyError('Applying `extDer` without specifying a coframe is not supported in the version of DGCV due to a bug in it\'s algorithm that will be patched in a later version.\n\n Suggestion apply `extDer(df, coframe="coframe_name")` instead of `extDer(df)`')
    if not isinstance(order, int) or order < 1:
        raise ValueError("`order` must be a positive integer.")

    # Recursive case for order > 1
    if order > 1:
        return extDer(extDer(df, coframe=coframe), coframe=coframe, order=order - 1)

    # Dispatch based on the type of `df`
    if isinstance(df, zeroFormAtom):
        return _extDer_zeroFormAtom(df, coframe)
    if isinstance(df, abstract_ZF):
        return _extDer_abstract_ZF(df, coframe)
    elif isinstance(df, abstDFAtom):
        return _extDer_abstDFAtom(df, coframe)
    elif isinstance(df, abstDFMonom):
        return _extDer_abstDFMonom(df, coframe)
    elif isinstance(df, abstract_DF):
        return _extDer_abstract_DF(df, coframe)
    elif isinstance(df,(int,float, sp.Expr)):
        return 0
    else:
        raise TypeError(f"`extDer` does not support type `{type(df).__name__}`.")

def _extDer_zeroFormAtom(df, coframe):
    """
    Compute the exterior derivative for zeroFormAtom.
    """
    return _extDer_abstract_ZF(abstract_ZF(df), coframe)

def _extDer_abstract_ZF(df, coframe):
    """
    Compute the exterior derivative of an `abstract_ZF` expression.

    Parameters:
    - df: An instance of `abstract_ZF`.
    - coframe: The coframe basis (optional).

    Returns:
    - The exterior derivative as an `abstract_DF` expression.
    """
    if coframe is None:
        return abstDFAtom(df, 1, ext_deriv_order=1,
                          _markers=frozenset([j for j in df._markers if j not in {"holomorphic", "antiholomorphic"}]))

    # Compute one-form terms using `coframe_derivative`
    oneForms = [coframe_derivative(df, coframe, j) * coframe.forms[j] for j in range(len(coframe.forms))]

    # Sum the terms
    return sum(oneForms[1:], oneForms[0])

def _extDer_abstDFAtom(df, coframe):
    """
    Compute the exterior derivative for abstDFAtom.
    """
    if coframe is None:
        order = df.ext_deriv_order+1 if df.ext_deriv_order else 1
        return abstDFAtom(df.coeff,df.degree+1,df.label,order,_markers=frozenset([j for j in df._markers if (j!="holomorphic" and j!="antiholomorphic")]))
    str_eqns = coframe.structure_equations
    if df in str_eqns:
        return str_eqns[df]
    if isinstance(df.coeff,(zeroFormAtom,abstract_ZF)):
        new_markers = frozenset([j for j in df._markers if (j!="holomorphic" and j!="antiholomorphic")])
        return extDer(df.coeff,coframe=coframe)*abstDFAtom(1,df.degree,df.label,df.ext_deriv_order,_markers=new_markers)+(df.coeff)*extDer(abstDFAtom(1,df.degree,df.label,df.ext_deriv_order,_markers=df._markers),coframe=coframe)
    if isinstance(df.coeff, sp.Expr) and len((df.coeff).free_symbols)>0:
        new_markers = frozenset([j for j in df._markers if (j!="holomorphic" and j!="antiholomorphic")])
        return abstDFAtom(df.coeff,1,ext_deriv_order=1)*abstDFAtom(1,df.degree,df.label,df.ext_deriv_order,_markers=new_markers)+(df.coeff)*extDer(abstDFAtom(1,df.degree,df.label,df.ext_deriv_order,_markers=df._markers),coframe=coframe)
    if df.label:
        order = df.ext_deriv_order+1 if df.ext_deriv_order else 1
        return (df.coeff)*abstDFAtom(1,df.degree,df.label,order,_markers=df._markers)
    return abstDFAtom(0,df.degree+1)

def _extDer_abstDFMonom(df, coframe):
    """
    Compute the exterior derivative for abstDFMonom.
    """
    result = abstract_DF([])
    for idx, factor in enumerate(df.factors_sorted):
        first_part = df.factors_sorted[:idx]
        last_part = df.factors_sorted[idx + 1:]
        if len(df.factors_sorted)==1:
            term = extDer(factor, coframe=coframe)
        elif idx==0:
            term = extDer(factor, coframe=coframe) * abstDFMonom(last_part)
        elif idx == len(df.factors_sorted) - 1:
            term = abstDFMonom(first_part) * extDer(factor, coframe=coframe) 
        else:
            term = abstDFMonom(first_part) * extDer(factor, coframe=coframe) * abstDFMonom(last_part)
        result += term
    return result

def _extDer_abstract_DF(df, coframe):
    """
    Compute the exterior derivative for abstract_DF.
    """
    result = abstract_DF([])
    for term in df.terms:
        result += extDer(term, coframe=coframe)
    return result

def _cofrDer_zeroFormAtom(zf, cf, cfIndex):
    """
    Compute the coframe derivative of a `zeroFormAtom` with respect to `cf.forms[cfIndex]`.

    Parameters:
    - zf: The `zeroFormAtom` instance to differentiate.
    - cf: The `abst_coframe` representing the coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - A new `zeroFormAtom` representing the coframe derivative.
    """
    if not isinstance(zf, zeroFormAtom):
        raise TypeError("`zf` must be an instance of `zeroFormAtom`.")
    if not isinstance(cf, abst_coframe):
        raise TypeError("`cf` must be an instance of `abst_coframe`.")
    if not isinstance(cfIndex, int) or cfIndex < 0:
        raise ValueError("`cfIndex` must be a non-negative integer.")

    if zf.coframe is not None and zf.coframe != cf:
        raise ValueError("Cannot differentiate a `zeroFormAtom` with respect to a different coframe.")

    if cfIndex >= len(cf.forms):
        raise IndexError(f"`cfIndex` {cfIndex} is out of bounds for coframe with {len(cf.forms)} forms.")

    if cfIndex in zf.coframe_independants:
        return 0*zf

    # Helper function to increment the partial derivative orders
    def raise_indices(int_list, int_index):
        new_list = list(int_list)
        new_list[int_index] += 1
        return tuple(new_list)

    # Extract or initialize `partials_orders`
    orders_list = zf.partials_orders if zf.partials_orders else (0,) * len(cf.forms)

    # Compute the new derivative
    return zeroFormAtom(
        zf.label,
        partials_orders=raise_indices(orders_list, cfIndex),
        coframe=zf.coframe,
        _markers=zf._markers,
        coframe_independants=zf.coframe_independants
    )

def _cofrDer_abstract_ZF(df, cf, cfIndex):
    """
    Compute the coframe derivative of an `abstract_ZF` expression with respect to `cf.forms[cfIndex]`.

    Parameters:
    - df: An instance of `abstract_ZF`.
    - cf: The coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - The coframe derivative as an `abstract_ZF` expression.
    """
    if isinstance(df, (int, float, sp.Expr)):  # Scalars differentiate to 0
        return 0
    if isinstance(df, zeroFormAtom):
        return _cofrDer_zeroFormAtom(df, cf, cfIndex)
    if isinstance(df.base, zeroFormAtom):  # Differentiate zeroFormAtom
        return _cofrDer_zeroFormAtom(df.base, cf, cfIndex)

    if isinstance(df.base, tuple):
        op, *args = df.base

        if op == "add":  # d(f + g) = df + dg
            return abstract_ZF(("add", *[_cofrDer_abstract_ZF(arg, cf, cfIndex) for arg in args]))

        elif op == "mul":  # Product Rule: d(fg) = df g + f dg
            terms = []
            for i, term in enumerate(args):
                other_factors = args[:i] + args[i+1:]
                terms.append(abstract_ZF(("mul", _cofrDer_abstract_ZF(term, cf, cfIndex), *other_factors)))
            return abstract_ZF(("add", *terms))

        elif op == "div":  # Quotient Rule: d(f/g) = (df g - f dg) / g²
            num, denom = args
            dnum = _cofrDer_abstract_ZF(num, cf, cfIndex)
            ddenom = _cofrDer_abstract_ZF(denom, cf, cfIndex)
            return abstract_ZF(("div", ("sub", abstract_ZF(("mul", dnum, denom)), abstract_ZF(("mul", num, ddenom))), ("pow", denom, 2)))

        elif op == "pow":  # Power Rule: d(f^g)
            base, exponent = args
            dbase = _cofrDer_abstract_ZF(base, cf, cfIndex)

            if isinstance(exponent, (int, float, sp.Expr)):
                new_exp = exponent-1
                return abstract_ZF(("mul", exponent, ("pow", base, new_exp), dbase))
            else:
                raise NotImplementedError(f'coframe derivatives are not implemented for type {type(base)} raised to type {type(exponent)}')

    return 0  # If df is constant, return 0

