from .._safeguards import retrieve_passkey
from ..backends._caches import _get_expr_num_types
from ..solvers import simplify_dgcv
from .eds import abst_coframe, abstract_ZF, coframe_derivative, zeroFormAtom


class abstract_VF_atom:
    def __init__(self,label:str,dual_label=None,coframe_identifier={'coframe':None,'index':0},):
        self.label=label
        CF=coframe_identifier.get('coframe',None)
        if isinstance(CF,abst_coframe):
            self.dual_coframe=CF
            self.dual_coframe_index=coframe_identifier.get('index',0)
        else:
            if dual_label is None or not isinstance(dual_label,str):
                dual_label='d_'+label
            self.dual_coframe = abst_coframe('_CF_auto_'+label,[dual_label],'_CF_'+label)
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "abstract_vector_field_atom"

    def __repr__(self):
        return self.label
    def __str__(self):
        return self.label

    def __call__(self, arg):
        if isinstance(arg,_get_expr_num_types()):
            return 0
        if isinstance(arg, (abstract_ZF,zeroFormAtom)):
            return coframe_derivative(arg,self.dual_coframe,self.dual_coframe_index)

    # --- promote to abstract_VF for algebraic ops ---
    def _as_vf(self):
        return abstract_VF.from_atom(self)

    def __add__(self, other):
        """Atom + (VF|atom) -> abstract_VF add node."""
        other_vf = abstract_VF._VFify(other)
        return abstract_VF(('sum', self._as_vf(), other_vf))

    def __radd__(self, other):
        # Support sum([...], start=0)
        if other == 0:
            return self._as_vf()
        other_vf = abstract_VF._VFify(other)
        return abstract_VF(('sum', other_vf, self._as_vf()))

    def __mul__(self, other):
        """
        Overload `*` for both scaling and Lie bracket:
          - if `other` is scalar (per abstract_VF._is_scalar), return ('scale', other, self)
          - else treat as bracket: ('bracket', self, other)
        """
        if abstract_VF._is_scalar(other):
            return abstract_VF(('scale', other, self._as_vf()))
        other_vf = abstract_VF._VFify(other)
        return abstract_VF(('bracket', self._as_vf(), other_vf))

    def __rmul__(self, other):
        # scalar * atom -> scale; if `other` is VF, Python will use other's __mul__
        if abstract_VF._is_scalar(other):
            return abstract_VF(('scale', other, self._as_vf()))
        return NotImplemented

    def _repr_latex_(self, raw: bool = False):
        s = self.label  # keep label as-is; customize if you want styling
        return s if raw else f"${s}$"

class abstract_VF:
    """
    Abstract Vector Field (AST-based)

    Nodes are tuples with first entry an operator tag among:
      - ('atom', <abstract_VF_atom>)
      - ('sum', <vf_1>, <vf_2>, ... , <vf_k>)   # k >= 2 after normalization, flattened
      - ('scale', <scalar>, <vf>)               # scalar in _get_expr_num_types()
      - ('bracket', <vf_left>, <vf_right>)
      - ('zero',)
      - ('ad', X, Y, n)

    Design goals:
      * Keep a *normalized* AST: flatten adds, combine nested scales, extract scalar factors from brackets.
      * Provide Python operators: +, * (scalar and bracket).
      * Keep implementation minimal while leaving hooks for future evaluation on abstract_ZF.
    """

    __slots__ = ("_node", "_order", "_dgcv_class_check", "_dgcv_category", "favor_factoring")

    # --------- constructors / factories ---------
    def __init__(self, node, favor_factoring=False):
        self.favor_factoring = True if favor_factoring is True else False
        self._node = self._normalize(node)
        self._order = self._compute_order(self._node)
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "abstract_vector_field"

    def _compute_order(self, n):
        """Return max nesting depth of Lie brackets along any path.
        Conventions:
          - 'zero', 'atom': 0
          - 'sum': max(children)
          - 'scale': depth(child)
          - 'bracket': 1 + max(depth(left), depth(right))
          - 'ad': n + max(depth(X), depth(Y))
        """
        tag = n[0]
        if tag in ('zero', 'atom'):
            return 0
        if tag in ('sum', 'sum'):
            if len(n) == 1:
                return 0
            return max(self._compute_order(ch) for ch in n[1:])
        if tag == 'scale':
            return self._compute_order(n[2])
        if tag == 'bracket':
            return 1 + max(self._compute_order(n[1]), self._compute_order(n[2]))
        if tag == 'ad':
            try:
                k = int(n[3])
            except Exception:
                k = 0
            return max(k + self._compute_order(n[2]), self._compute_order(n[1]))
        # Fallback
        return 0

    @classmethod
    def from_atom(cls, atom: "abstract_VF_atom"):
        if not isinstance(atom, abstract_VF_atom):
            raise TypeError("from_atom expects an abstract_VF_atom")
        return cls(('atom', atom))

    @classmethod
    def zero(cls):
        return cls(('zero',))

    # --------- basic predicates & coercions ---------
    @staticmethod
    def _is_scalar(x):
        return isinstance(x, _get_expr_num_types()+(abstract_ZF, zeroFormAtom))

    @staticmethod
    def _is_function_scalar(x):
        return isinstance(x, (abstract_ZF, zeroFormAtom))

    @classmethod
    def _VFify(cls, x: "abstract_VF | abstract_VF_atom | tuple") -> "abstract_VF":
        if isinstance(x, cls):
            return x
        if isinstance(x, abstract_VF_atom):
            return cls.from_atom(x)
        # Accept a raw AST node tuple and wrap it
        if isinstance(x, tuple) and x:
            tag = x[0]
            if tag in {'atom', 'sum', 'scale', 'bracket', 'zero', 'ad'}:
                return cls(x)
        raise TypeError(f"Expected abstract_VF or abstract_VF_atom, got {type(x)!r}")

    # --------- normalization logic ---------
    def _normalize(self, node):
        if isinstance(node, abstract_VF):
            return node._node

        # Leaf coercions
        if isinstance(node, abstract_VF_atom):
            return ('atom', node)

        if not isinstance(node, tuple):
            raise TypeError("abstract_VF expects a node tuple or an abstract_VF_atom")

        tag = node[0]

        if tag == 'zero':
            return ('zero',)

        if tag == 'atom':
            atom = node[1]
            if not isinstance(atom, abstract_VF_atom):
                raise TypeError("('atom', ...) must contain an abstract_VF_atom")
            return ('atom', atom)

        if tag == 'sum':
            # Flatten and normalize children
            terms = []
            for t in node[1:]:
                t_node = self._VFify(t)._node
                if t_node[0] == 'zero':
                    continue
                if t_node[0] == 'sum':
                    terms.extend(t_node[1:])
                else:
                    terms.append(t_node)
            if not terms:
                return ('zero',)
            if len(terms) == 1:
                return terms[0]
            return ('sum', *terms)

        if tag == 'scale':
            if len(node) != 3:
                raise ValueError("('scale', scalar, vf) requires exactly 2 operands")
            c, v = node[1], node[2]
            if not self._is_scalar(c):
                raise TypeError("Scale coefficient must be numeric/scalar by _get_expr_num_types()")
            v_node = self._VFify(v)._node
            # If the child is zero, the whole product is zero
            if v_node[0] == 'zero':
                return ('zero',)
            # 0 * X = 0, 1 * X = X
            try:
                if c == 0:
                    return ('zero',)
                if c == 1:
                    return v_node
            except Exception:
                pass
            # combine nested scales: c*(d*X) -> (c*d)*X
            if v_node[0] == 'scale':
                inner_c, inner_v = v_node[1], v_node[2]
                try:
                    new_c = c * inner_c
                except Exception:
                    # Fallback: if scalar multiplication fails for some symbolic type, leave as a nested scale
                    return ('scale', c, v_node)
                return self._normalize(('scale', new_c, inner_v))
            return ('scale', c, v_node)

        if tag == 'bracket':
            if len(node) != 3:
                raise ValueError("('bracket', X, Y) requires exactly 2 operands")
            X = self._VFify(node[1])._node
            Y = self._VFify(node[2])._node
            # Annihilation by zero arguments
            if X[0] == 'zero' or Y[0] == 'zero':
                return ('zero',)
            # Antisymmetry simplification: [X,X] = 0
            if X == Y:
                return ('zero',)

            # Helper: split out only numeric scales; keep function-scalars inside
            def split_numeric_scale(n):
                if isinstance(n, tuple) and n and n[0] == 'scale':
                    c, core = n[1], n[2]
                    if isinstance(c, _get_expr_num_types()):
                        return c, core
                return 1, n

            a, Xb = split_numeric_scale(X)
            b, Yb = split_numeric_scale(Y)

            if self.favor_factoring is not True:
                # Prepare distribution over sums (treat 'sum' same as 'sum')
                def add_terms(n):
                    if isinstance(n, tuple) and n and n[0]=='sum':
                        return list(n[1:])
                    return None

                X_terms = add_terms(Xb)
                Y_terms = add_terms(Yb)

                def make_sum(terms):
                    # Normalize each term; the enclosing 'sum' case will flatten as needed
                    normed = [self._normalize(t) for t in terms]
                    if not normed:
                        return ('zero',)
                    if len(normed) == 1:
                        return normed[0]
                    return ('sum', *normed)

                # Distribute bracket over sums
                if X_terms is not None and Y_terms is not None:
                    core = make_sum([('bracket', xi, yj) for xi in X_terms for yj in Y_terms])
                elif X_terms is not None:
                    core = make_sum([('bracket', xi, Yb) for xi in X_terms])
                elif Y_terms is not None:
                    core = make_sum([('bracket', Xb, yj) for yj in Y_terms])
                else:
                    core = ('bracket', Xb, Yb)
            else:
                core = ('bracket', Xb, Yb)

            # Re-attach numeric scalar product if present
            try:
                ab = a * b
            except Exception:
                ab = None

            if ab in (0, 0.0):
                return ('zero',)
            if ab is None or ab == 1:
                return core
            return ('scale', ab, self._VFify(core)._node)

        if tag == 'ad':
            # node layout: ('ad', X, Y, n)
            if len(node) != 4:
                raise ValueError("('ad', X, Y, n) requires exactly 3 operands and an exponent n")
            Xn = self._VFify(node[1])._node
            Yn = self._VFify(node[2])._node
            n = node[3]
            try:
                n_int = int(n)
            except Exception:
                raise TypeError("ad power n must be an integer")
            if n_int < 0:
                raise ValueError("ad power n must be >= 0")
            if n_int == 0:
                return Yn
            if n_int == 1:
                return self._normalize(('bracket', Xn, Yn))
            # For n > 1, keep compact 'ad' node to consolidate formulas
            return ('ad', Xn, Yn, n_int)

        raise ValueError(f"Unknown abstract_VF node tag: {tag!r}")

    # --------- public helpers ---------
    @property
    def node(self):
        """Return normalized AST node (immutable tuple)."""
        return self._node

    @property
    def order(self) -> int:
        """Maximum bracket nesting depth in this VF."""
        return self._order

    def is_zero(self) -> bool:
        return self._node[0] == 'zero'

    # --------- pretty printing (minimal for now) ---------
    def __repr__(self):
        return f"abstract_VF({self._node!r})"

    def __str__(self):
        return self._to_str(self._node)

    def _to_str(self, n):
        tag = n[0]
        if tag == 'zero':
            return '0'
        if tag == 'atom':
            return getattr(n[1], 'label', repr(n[1]))
        if tag == 'scale':
            c = n[1]
            child = n[2]
            child_str = self._to_str(child)
            # Parenthesize sums to avoid ambiguity: (f)*(A + B)
            if isinstance(child, tuple) and child and child[0] == 'sum':
                return f"({c})*({child_str})"
            return f"({c})*{child_str}"
        if tag == 'sum':
            return " + ".join(self._to_str(t) for t in n[1:])
        if tag == 'bracket':
            return f"[{self._to_str(n[1])}, {self._to_str(n[2])}]"
        if tag == 'ad':
            Xs = self._to_str(n[1]) if isinstance(n[1], tuple) else str(n[1])
            Ys = self._to_str(n[2]) if isinstance(n[2], tuple) else str(n[2])
            power = n[3]
            return f"ad_{Xs}_{power}({Ys})"
        return repr(n)

    def _scalar_to_latex(self, c):
        try:
            if isinstance(c, abstract_ZF):
                return c._repr_latex_(raw=True)
            if isinstance(c, zeroFormAtom):
                return c._repr_latex_(raw=True)
        except Exception:
            pass
        # best-effort fallback
        try:
            from sympy import latex as _sp_latex  # lazy, optional
            return _sp_latex(c)
        except Exception:
            return str(c)

    def _to_latex(self, n):
        tag = n[0]
        if tag == 'zero':
            return '0'
        if tag == 'atom':
            leaf = n[1]
            if hasattr(leaf, '_repr_latex_'):
                return leaf._repr_latex_(raw=True)
            return getattr(leaf, 'label', 'X')
        if tag == 'scale':
            c_raw = n[1]
            c_tex = self._scalar_to_latex(c_raw)
            child = n[2]
            child_tex = self._to_latex(child)
            # Parenthesize sums to avoid ambiguity: (f)\,(A + B)
            if isinstance(child, tuple) and child and child[0] == 'sum':
                return f"({c_tex})\\, ( {child_tex} )"
            return f"({c_tex})\\, {child_tex}"
        if tag == 'sum':
            return " \\,+\\, ".join(self._to_latex(t) for t in n[1:])
        if tag == 'bracket':
            return f"\\left[ {self._to_latex(n[1])} \\, , \\, {self._to_latex(n[2])} \\right]"
        if tag == 'ad':
            Xtex = self._to_latex(n[1]) if isinstance(n[1], tuple) else str(n[1])
            Ytex = self._to_latex(n[2]) if isinstance(n[2], tuple) else str(n[2])
            power = n[3]
            return f"\\operatorname{{ad}}_{{{Xtex}}}^{{{power}}}({Ytex})"
        return str(n)

    def _repr_latex_(self, raw: bool = False):
        s = self._to_latex(self._node)
        return s if raw else f"${s}$"

    # --------- algebraic operators ---------
    def __add__(self, other):
        other_vf = self._VFify(other)
        return type(self)(('sum', self, other_vf))

    def __radd__(self, other):
        # Support sum([...], start=0)
        if other == 0:
            return self
        other_vf = self._VFify(other)
        return type(self)(('sum', other_vf, self))

    def __mul__(self, other):
        """Use `*` for both scaling and Lie bracket.
        If `other` is scalar (per _is_scalar), return scale; otherwise bracket.
        """
        if self._is_scalar(other):
            return type(self)(('scale', other, self))
        other_vf = self._VFify(other)
        return type(self)(('bracket', self, other_vf))

    def __rmul__(self, other):
        # Only scalars hit __rmul__; VFs on the left use their own __mul__
        if not self._is_scalar(other):
            return NotImplemented
        return type(self)(('scale', other, self))

    def bracket(self, other: "abstract_VF | abstract_VF_atom"):
        other_vf = self._VFify(other)
        return type(self)(('bracket', self, other_vf))

    def _eval_simplify(self, aggressive=False, **kws):
        """
        Simplify the vector field AST.

        Steps:
          1) Canonicalize structure via VF_expand (bottom-up expansion of ad and
             bracket rules), yielding a shallow tree with shape add -> scale -> bracket/atom/zero.
          2) Simplify scalar coefficients in every ('scale', c, ·) using simplify_dgcv(c).
          3) Inside each bracket, consolidate nested brackets into compact ('ad', X, Y, k)
             whenever possible, starting from patterns like ('bracket', X, ('bracket', X, Y))
             and ('bracket', X, ('bracket', Y, X)), and then fold chains to increase k.
        """
        if aggressive is True:
            return VF_factor(self,aggressive=True)._eval_simplify()

        # Phase 1: canonicalize using VF_expand
        canon_vf = VF_expand(self)
        root = canon_vf._node

        # Helpers
        def simp_coeff(c):
            try:
                return simplify_dgcv(c)
            except Exception:
                return c

        def nodes_equal(a, b):
            return a == b

        def fold_ad(node):
            """Normalize ('ad', X, Y, k) by pushing ad-power into nested right operands.
            If Y is ('ad', X, Z, m) -> ('ad', X, Z, k+m)
            If Y is ('bracket', X, Z) -> ('ad', X, Z, k+1)
            If Y is ('bracket', Z, X) -> ('scale', -1, ('ad', X, Z, k+1))
            Recurse until fixed point.
            """
            if not (isinstance(node, tuple) and node and node[0] == 'ad'):
                return node
            Xn, Yn, k = node[1], node[2], int(node[3])

            changed = True
            cur_X, cur_Y, cur_k = Xn, Yn, k
            while changed:
                changed = False
                if isinstance(cur_Y, tuple):
                    tag = cur_Y[0]
                    if tag == 'ad':
                        X2, Z, m = cur_Y[1], cur_Y[2], int(cur_Y[3])
                        if nodes_equal(cur_X, X2):
                            cur_Y = Z
                            cur_k = cur_k + m
                            changed = True
                            continue
                    if tag == 'bracket':
                        A, B = cur_Y[1], cur_Y[2]
                        if nodes_equal(A, cur_X):
                            cur_Y = B
                            cur_k = cur_k + 1
                            changed = True
                            continue
                        if nodes_equal(B, cur_X):
                            # ad_X^{k}( [Z, X] ) = - ad_X^{k+1}(Z)
                            Z = A
                            cur_Y = ('scale', -1, ('ad', cur_X, Z, cur_k + 1))
                            # One more normalization pass after embedding a scale
                            cur_Y = normalize(cur_Y)
                            changed = True
                            continue
            return ('ad', cur_X, cur_Y, cur_k)

        def normalize(n):
            """Use class normalization to clean up structure after local rewrites."""
            return abstract_VF(n)._node

        def simplify_node(n):
            tag = n[0]
            if tag in ('zero', 'atom'):
                return n
            if tag in ('sum', 'sum'):
                parts = [simplify_node(ch) for ch in n[1:]]
                return normalize(('sum', *parts))
            if tag == 'scale':
                c = simp_coeff(n[1])
                child = simplify_node(n[2])
                return normalize(('scale', c, child))
            if tag == 'bracket':
                L = simplify_node(n[1])
                R = simplify_node(n[2])
                # Pattern 0: ('bracket', X, ('ad', X, Y, k)) -> ('ad', X, Y, k+1)
                if isinstance(R, tuple) and R and R[0] == 'ad' and nodes_equal(L, R[1]):
                    candidate = ('ad', L, R[2], int(R[3]) + 1)
                    return normalize(fold_ad(candidate))
                # Try to consolidate special nested patterns into ad
                # Pattern 1: ('bracket', X, ('bracket', X, Y)) -> ('ad', X, Y, 2)
                if isinstance(R, tuple) and R and R[0] == 'bracket' and nodes_equal(L, R[1]):
                    candidate = ('ad', L, R[2], 2)
                    return normalize(fold_ad(candidate))
                # Pattern 2: ('bracket', X, ('bracket', Y, X)) -> -('ad', X, Y, 2)
                if isinstance(R, tuple) and R and R[0] == 'bracket' and nodes_equal(L, R[2]):
                    candidate = ('scale', -1, ('ad', L, R[1], 2))
                    return normalize(fold_ad(candidate))
                # Otherwise keep bracket
                return normalize(('bracket', L, R))
            if tag == 'ad':
                # Simplify inside and fold if possible
                Xs = simplify_node(n[1])
                Ys = simplify_node(n[2])
                node2 = ('ad', Xs, Ys, int(n[3]))
                return normalize(fold_ad(node2))
            # Fallback
            return n

        simplified = simplify_node(root)
        return abstract_VF(simplified)

    def __call__(self, arg):
        """
        Evaluate the abstract vector field on a zero-form-like argument.

        Rules implemented (recursively on the AST):
          • zero:             0(arg) = 0
          • atom:             delegates to leaf __call__ (coframe_derivative)
          • add:              (X1 + ... + Xk)(f) = Σ Xi(f)
          • scale:            (c X)(f) = c * X(f)          (c is any _is_scalar)
          • bracket:          [X, Y](f) = X(Y(f)) - Y(X(f))

        Notes:
          • If `arg` is a pure number (in _get_expr_num_types()), return 0.
          • If `arg` is abstract_ZF or zeroFormAtom, we rely on their own arithmetic
            (e.g., product rule) via `coframe_derivative` in leaf evaluation.
        """
        # Constant functions are killed by vector fields.
        if isinstance(arg, _get_expr_num_types()):
            return 0

        # Accept abstract_ZF or zeroFormAtom (and potentially SymPy Expr) as inputs.
        if not isinstance(arg, (abstract_ZF, zeroFormAtom)) and not isinstance(arg, _get_expr_num_types()):
            raise TypeError(f"Unsupported argument for vector field action: {type(arg)!r}")

        def eval_node(n):
            tag = n[0]
            if tag == 'zero':
                return 0
            if tag == 'atom':
                leaf = n[1]
                # Delegate to atom's action (uses coframe_derivative)
                return leaf(arg)
            if tag == 'sum':
                acc = 0
                first = True
                for child in n[1:]:
                    val = eval_node(child)
                    if first:
                        acc = val
                        first = False
                    else:
                        acc = acc + val
                return acc if not first else 0
            if tag == 'scale':
                c = n[1]
                v = n[2]
                inner = eval_node(v)
                return c * inner
            if tag == 'bracket':
                Xn, Yn = n[1], n[2]
                # [X,Y](f) = X(Y(f)) - Y(X(f))
                Yf = eval_node(Yn)
                X_of_Yf = apply_node(n[1], Yf)
                Y_of_Xf = apply_node(n[2], eval_node(n[1]))
                return X_of_Yf - Y_of_Xf
            if tag == 'ad':
                # Evaluate via normalized expansion
                expanded = self._normalize(n)
                return eval_node(expanded)
            raise ValueError(f"Unknown abstract_VF node tag during evaluation: {tag!r}")

        def apply_node(vf_node, zero_form):
            t = vf_node[0]
            if t == 'zero':
                return 0
            if t == 'atom':
                return vf_node[1](zero_form)
            if t == 'sum':
                acc2 = 0
                first2 = True
                for ch in vf_node[1:]:
                    val2 = apply_node(ch, zero_form)
                    if first2:
                        acc2 = val2
                        first2 = False
                    else:
                        acc2 = acc2 + val2
                return acc2 if not first2 else 0
            if t == 'scale':
                c2 = vf_node[1]
                return c2 * apply_node(vf_node[2], zero_form)
            if t == 'bracket':
                # [A,B](g) = A(B(g)) - B(A(g))
                return apply_node(vf_node[1], apply_node(vf_node[2], zero_form)) - \
                       apply_node(vf_node[2], apply_node(vf_node[1], zero_form))
            if t == 'ad':
                expanded = self._normalize(vf_node)
                return apply_node(expanded, zero_form)
            raise ValueError(f"Unknown node in apply_node: {t!r}")

        return eval_node(self._node)

def _vf_apply_node_to_zero_form(vf_node, zero_form):
    tag = vf_node[0]
    if tag == 'zero':
        return 0
    if tag == 'atom':
        return vf_node[1](zero_form)
    if tag == 'scale':
        c, v = vf_node[1], vf_node[2]
        return c * _vf_apply_node_to_zero_form(v, zero_form)
    if tag == 'sum':
        it = iter(vf_node[1:])
        try:
            acc = _vf_apply_node_to_zero_form(next(it), zero_form)
        except StopIteration:
            return 0
        for ch in it:
            acc = acc + _vf_apply_node_to_zero_form(ch, zero_form)
        return acc
    if tag == 'bracket':
        # [A,B](g) = A(B(g)) - B(A(g))
        return _vf_apply_node_to_zero_form(vf_node[1], _vf_apply_node_to_zero_form(vf_node[2], zero_form)) - \
               _vf_apply_node_to_zero_form(vf_node[2], _vf_apply_node_to_zero_form(vf_node[1], zero_form))
    raise ValueError(f"Unknown VF node in _vf_apply_node_to_zero_form: {tag!r}")

def VF_expand(vf: abstract_VF,iterations=None) -> abstract_VF:
    """Expand Lie brackets with function scalars bottom-up.

    Steps:
      1) Expand all 'ad' operators into nested 'bracket' nodes.
      2) Normalize once (flatten sums, factor numeric scalars, distribute brackets over sums).
      3) Expand from the lowest bracket depth up to the root using:
            [f*X, R] = f [X, R] - R(f) * X
            [L, g*Y] = g [L, Y] + L(g) * Y
         (f, g are function-scalars: abstract_ZF or zeroFormAtom).
      4) Distribute outer 'scale' over sums for readability as we go.
    """
    if iterations is None:
        iterations=vf.order
    if iterations>1:
        return VF_expand(VF_expand(vf,iterations=1),iterations=iterations-1)
    # --- Phase 1: expand all 'ad' operators into nested brackets ---
    def expand_ad(n):
        tag = n[0]
        if tag in ('zero', 'atom'):
            return n
        if tag == 'sum':
            return ('sum',) + tuple(expand_ad(ch) for ch in n[1:])
        if tag == 'scale':
            return ('scale', n[1], expand_ad(n[2]))
        if tag == 'bracket':
            return ('bracket', expand_ad(n[1]), expand_ad(n[2]))
        if tag == 'ad':
            Xn = expand_ad(n[1])
            Yn = expand_ad(n[2])
            try:
                k = int(n[3])
            except Exception:
                k = 0
            if k <= 0:
                return Yn
            if k == 1:
                return ('bracket', Xn, Yn)
            acc = Yn
            for _ in range(k):
                acc = ('bracket', Xn, acc)
            return acc
        raise ValueError(f"Unknown VF node in expand_ad: {tag!r}")

    root = expand_ad(vf._node)

    # Normalize once to flatten sums and factor numeric scalars (and bracket bilinearity)
    root = abstract_VF(root)._node

    # --- Utilities for Phase 2 ---
    def subtree_order(n):
        tag = n[0]
        if tag in ('zero', 'atom'):
            return 0
        if tag == 'sum':
            if len(n) == 1:
                return 0
            return max(subtree_order(ch) for ch in n[1:])
        if tag == 'scale':
            return subtree_order(n[2])
        if tag == 'bracket':
            return 1 + max(subtree_order(n[1]), subtree_order(n[2]))
        return 0  # no 'ad' remains after Phase 1

    def expand_at_depth(n, target_depth):
        tag = n[0]
        if tag in ('zero', 'atom'):
            return n, 0, False
        if tag == 'sum':
            changed = False
            new_children = []
            max_d = 0
            for ch in n[1:]:
                ch2, d, chg = expand_at_depth(ch, target_depth)
                new_children.append(ch2)
                changed = changed or chg
                if d > max_d:
                    max_d = d
            return ('sum',) + tuple(new_children), max_d, changed
        if tag == 'scale':
            c, v = n[1], n[2]
            v2, d, chg = expand_at_depth(v, target_depth)
            # Distribute scale over sums produced by bracket expansion
            if isinstance(v2, tuple) and v2 and v2[0] == 'sum':
                parts = []
                for ch in v2[1:]:
                    parts.append(('scale', c, ch))
                return (('sum',) + tuple(parts)), d, True
            return ('scale', c, v2), d, chg
        if tag == 'bracket':
            L2, dL, chgL = expand_at_depth(n[1], target_depth)
            R2, dR, chgR = expand_at_depth(n[2], target_depth)
            my_depth = 1 + max(dL, dR)
            if my_depth == target_depth:
                # Both sides function-scaled?
                if (
                    isinstance(L2, tuple) and L2 and L2[0] == 'scale' and isinstance(L2[1], (abstract_ZF, zeroFormAtom)) and
                    isinstance(R2, tuple) and R2 and R2[0] == 'scale' and isinstance(R2[1], (abstract_ZF, zeroFormAtom))
                ):
                    f, X = L2[1], L2[2]
                    g, Y = R2[1], R2[2]
                    Xg = _vf_apply_node_to_zero_form(X, g)
                    Yf = _vf_apply_node_to_zero_form(Y, f)
                    term1 = ('scale', f * g, ('bracket', X, Y))
                    term2 = ('scale', f * Xg, Y)
                    term3 = ('scale', -g * Yf, X)
                    return (('sum', term1, term2, term3)), my_depth, True
                # Left function-scaled?
                if isinstance(L2, tuple) and L2 and L2[0] == 'scale' and isinstance(L2[1], (abstract_ZF, zeroFormAtom)):
                    f, X = L2[1], L2[2]
                    term1 = ('scale', f, ('bracket', X, R2))
                    Yf = _vf_apply_node_to_zero_form(R2, f)
                    term2 = ('scale', -Yf, X)
                    return (('sum', term1, term2)), my_depth, True
                # Right function-scaled?
                if isinstance(R2, tuple) and R2 and R2[0] == 'scale' and isinstance(R2[1], (abstract_ZF, zeroFormAtom)):
                    g, Y = R2[1], R2[2]
                    term1 = ('scale', g, ('bracket', L2, Y))
                    Xg = _vf_apply_node_to_zero_form(L2, g)
                    term2 = ('scale', Xg, Y)
                    return (('sum', term1, term2)), my_depth, True
                # Nothing to expand here
                return ('bracket', L2, R2), my_depth, (chgL or chgR)
            else:
                return ('bracket', L2, R2), my_depth, (chgL or chgR)
        raise ValueError(f"Unknown VF node in expand_at_depth: {tag!r}")

    # --- Phase 2: expand from lowest depth up to the root ---
    current = root
    max_depth = subtree_order(current)
    for depth in range(1, max_depth + 1):
        current, _, _ = expand_at_depth(current, depth)
        # Normalize after each layer to keep the tree shallow and factor numerics
        current = abstract_VF(current)._node

    return abstract_VF(current)

def VF_factor(vf: abstract_VF, aggressive: bool = False) -> abstract_VF:
    """Factor a vector-field expression.

    Behavior:
      A) Factor out a common scalar from a top-level sum of ('scale', c_i, ·) terms.
         - Handles exact-equality common factors across arbitrary scalars.
         - Handles integer numeric GCD across integer coefficients.
      B) Combine compatible bracket/"ad" terms using linearity in the *second* slot
         (for numeric coefficients only), e.g. [X,Y] + [Z,X] -> [X, Y - Z].
      C) If ``aggressive=True``, run ``vf._eval_simplify()`` first to canonicalize
         nested bracket patterns into ``('ad', X, Y, k)`` where possible.

    Notes:
      • The returned object always has ``favor_factoring=False`` so that '_normalize'
        does not distribute brackets over sums again.
      • We **only** combine terms using numeric outer coefficients; function-scalars
        (abstract_ZF/zeroFormAtom) are not pushed inside brackets/ad because
        [X, fY] ≠ f[X, Y] in general.
    """
    # Optional canonicalization pass
    root_vf = vf._eval_simplify() if aggressive else vf

    def norm(n):
        return abstract_VF(n, favor_factoring=False)._node

    node = root_vf._node
    if node[0] != 'sum':
        return abstract_VF(node, favor_factoring=False)

    # ---- Helper predicates ----
    NumT = _get_expr_num_types()
    def is_numeric(x):
        return isinstance(x, NumT)

    # ---- Flatten and drop zeros (safety) ----
    flat_terms = []
    for t in node[1:]:
        tn = norm(t)
        if isinstance(tn, tuple) and tn and tn[0] == 'sum':
            flat_terms.extend(tn[1:])
        elif tn[0] == 'zero':
            continue
        else:
            flat_terms.append(tn)

    # ---- (A) Factor common scalar from top-level scaled terms ----
    all_scale = all(isinstance(t, tuple) and t and t[0] == 'scale' for t in flat_terms)
    factored_prefix = None

    if all_scale and flat_terms:
        coeffs = [t[1] for t in flat_terms]
        # Case A1: exact common scalar (incl. abstract_ZF/zeroFormAtom) across all terms
        all_equal = all(coeffs[0] == c for c in coeffs)
        if all_equal:
            factored_prefix = coeffs[0]
            flat_terms = [norm(t[2]) for t in flat_terms]
        else:
            # Case A2: integer numeric GCD across integer coefficients
            if all(isinstance(c, int) for c in coeffs):
                from math import gcd
                g = 0
                for c in coeffs:
                    g = gcd(g, abs(c))
                if g not in (0, 1):
                    factored_prefix = g
                    # divide coefficients by g
                    new_terms = []
                    for t in flat_terms:
                        c = t[1]
                        v = t[2]
                        new_terms.append(('scale', c // g, v))
                    flat_terms = [norm(nt) for nt in new_terms]

    # ---- (A2.5) Combine like terms with identical vector factor child ----
    # e.g., (c1)*X + (c2)*X + (c3)*X  ->  (c1+c2+c3)*X
    like_groups = {}  # key: repr(child) -> (child_node, coeff_sum)
    non_scale_terms = []
    for t in flat_terms:
        if isinstance(t, tuple) and t and t[0] == 'scale':
            c = t[1]
            child = norm(t[2])
            key = repr(child)
            if key in like_groups:
                # sum coefficients using native addition (supports abstract_ZF/zeroFormAtom/numerics)
                like_groups[key] = (child, like_groups[key][1] + c)
            else:
                like_groups[key] = (child, c)
        else:
            non_scale_terms.append(t)

    # Rebuild flat_terms from combined like terms + others; drop zero coefficients
    combined_like_terms = []
    for child, coeff_sum in like_groups.values():
        try:
            is_zero = (coeff_sum == 0)
        except Exception:
            is_zero = False
        if not is_zero:
            combined_like_terms.append(('scale', coeff_sum, child))
    flat_terms = combined_like_terms + non_scale_terms

    # ---- (B) Combine compatible bracket and ad terms (numeric coefficients only) ----
    combined = []
    leftovers = []

    # Grouping maps: brackets by left operand; ads by (X, k)
    br_groups = {}  # key: left node; val: list of rhs terms already scaled by numeric coeff
    ad_groups = {}  # key: (X, k); val: list of Y terms already scaled by numeric coeff

    for t in flat_terms:
        if t[0] == 'scale' and is_numeric(t[1]):
            c = t[1]
            child = norm(t[2])
            # Brackets
            if child[0] == 'bracket':
                L, R = child[1], child[2]
                # Canonicalize to key on left operand; use antisymmetry
                if isinstance(L, tuple) and L and L[0] == 'atom' and isinstance(R, tuple) and R and R[0] == 'atom':
                    # Ok, but we don't actually require atoms; antisymmetry is general
                    pass
                # If of the form [Z, X], rewrite to -c * [X, Z]
                if isinstance(L, tuple) and isinstance(R, tuple) and L == root_vf._VFify(L)._node and R == root_vf._VFify(R)._node:
                    pass  # nodes are already normalized
                if True:
                    if child[1] != child[2]:
                        if child[1] == child[2]:
                            pass
                    # swap when right equals key target; simple canonical rule: make smaller repr on left? No, just if right is same as some left later.
                # Simple antisymmetry swap:
                # We'll canonicalize by comparing reprs so [A,B] and [B,A] map to the same left via swap if repr(B) < repr(A)
                L_c, R_c, c_c = L, R, c
                try:
                    if repr(R) < repr(L):
                        L_c, R_c = R, L
                        c_c = -c
                except Exception:
                    pass
                rhs_scaled = ('scale', c_c, R_c) if c_c != 1 else R_c
                br_groups.setdefault(repr(L_c), (L_c, []) )[1].append(rhs_scaled)
                continue
            # ad terms
            if child[0] == 'ad':
                X, Y, k = child[1], child[2], child[3]
                rhs_scaled = ('scale', c, Y) if c != 1 else Y
                ad_groups.setdefault((repr(X), k), (X, k, []) )[2].append(rhs_scaled)
                continue
        # Not combinable; keep as is
        leftovers.append(t)

    # Build combined bracket terms
    for _, (L_key, rhs_list) in br_groups.items():
        if len(rhs_list) == 1:
            # Just c*[L, R]
            rhs_sum = rhs_list[0]
        else:
            rhs_sum = ('sum', *rhs_list)
        combined.append(('bracket', L_key, norm(rhs_sum)))

    # Build combined ad terms
    for _, (X, k, rhs_list) in ad_groups.items():
        if len(rhs_list) == 1:
            rhs_sum = rhs_list[0]
        else:
            rhs_sum = ('sum', *rhs_list)
        combined.append(('ad', X, norm(rhs_sum), k))

    # Reassemble all pieces
    new_sum_terms = combined + leftovers
    if not new_sum_terms:
        out_node = ('zero',)
    elif len(new_sum_terms) == 1:
        out_node = new_sum_terms[0]
    else:
        out_node = ('sum', *new_sum_terms)

    # Attach factored prefix if any
    if factored_prefix is not None:
        out_node = ('scale', factored_prefix, out_node)

    return abstract_VF(out_node, favor_factoring=False)

def abst_lie_bracket(X: "abstract_VF | abstract_VF_atom", Y: "abstract_VF | abstract_VF_atom") -> abstract_VF:
    """Return the abstract Lie bracket [X, Y] with normalization."""
    Xv = abstract_VF._VFify(X)
    Yv = abstract_VF._VFify(Y)
    return abstract_VF(('bracket', Xv, Yv))
