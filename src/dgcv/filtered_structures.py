
import pandas as pd
import sympy as sp

from ._config import get_dgcv_settings_registry
from ._safeguards import _cached_caller_globals, create_key, retrieve_passkey
from .dgcv_core import clearVar, listVar, variableProcedure
from .dgcv_formatter import get_style
from .finite_dim_algebras import AlgebraElement, FAClass
from .tensors import tensorProduct


class Tanaka_symbol(sp.Basic):
    """
    dgcv class representing a symbol-like object for Tanaka prolongation.

    Parameters
    ----------
    GLA : FAClass, (with negatively graded Lie algebra structure)
        ambiant Lie algebra's negative part. The first entry in GLS.gading must be a list of negative weights, which will be used for the prolongation degrees

    Methods
    -------
    prolong

    Examples
    --------
    """
    def __new__(cls, GLA, nonnegParts = [], assume_FGLA = False, subspace = None, index_threshold = None, _validated = False):
        if _validated is False:

            if not isinstance(GLA, FAClass):
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` to be a graded Lie algebra (`FAClass` in particular) with negative weights in the first element of `GLA.grading`."
                )
            elif isinstance(GLA.grading[0],(list,tuple)):
                if not all(j<0 for j in GLA.grading[0]):
                    raise TypeError(
                        "`Tanaka_symbol` expects `GLA` to be a graded Lie algebra (`FAClass` in particular) with negative weights in the first element of `GLA.grading`."
                    )
            elif not all(j<0 for j in GLA.grading):
                print(GLA.grading)
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` to be a graded Lie algebra (`FAClass` in particular) with negative weights in the first element of `GLA.grading`."
                )

            if subspace:
                if not isinstance(subspace,(list,tuple)):
                    raise TypeError(
                        "`Tanaka_symbol` expects `subpsace` to be a list of AlgebraElement instances belonging to the `FAClass` `GLA`."
                    )
                if not all(isinstance(j,AlgebraElement) for j in subspace) or not all(j.algebra==GLA for j in subspace):
                    raise TypeError(
                        "`Tanaka_symbol` expects `subpsace` to be a list of AlgebraElement instances belonging to the `FAClass` given for `GLA`."
                    )
            else:
                subspace = GLA

            if isinstance(nonnegParts,dict):
                NNPList = list(nonnegParts.values())
            elif isinstance(nonnegParts,(list,tuple)):
                NNPList = [nonnegParts]
            else:
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` built from the `FAClass` given for `GLA` with `valence` of the form (1,0,...0). Or it can be a dictionariy whose keys are non-negative weights, and whose key-values are such lists."
                )
            for NNP in NNPList:
                if not all(isinstance(j,tensorProduct) for j in NNP) or not all(j.vector_space==GLA for j in NNP):
                    print('NNP')
                    print(NNP)
                    print('NNP')
                    raise TypeError(
                        "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `FAClass` given for `GLA` with `valence` of the form (1,0,...0).Or it can be a dictionariy whose keys are non-negative weights, and whose key-values are such lists."
                    )

            def valence_check(tp):
                for j in tp.coeff_dict:
                    valence = j[len(j)//2:]
                    if valence[0] != 1:
                        return False
                    if not all(j in {0} for j in valence[1:]):
                        return False
                return True

            if not all(valence_check(j) for j in nonnegParts):
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `FAClass` given for `GLA` with `valence` of the form (1,0,...0)."
                )

        obj = sp.Basic.__new__(cls, GLA, nonnegParts)

        obj._subspace = subspace

        return obj

    def __init__(self, GLA, nonnegParts = [], assume_FGLA = False, subspace = None, index_threshold = None, _validated = False):
        self.ambiantGLA = GLA
        self.negativePart = self._subspace
        self.assume_FGLA = assume_FGLA
        self.nonnegParts = nonnegParts
        negWeights = sorted(tuple(set(GLA.grading[0]))) if isinstance(GLA.grading,(list,tuple)) else sorted(tuple(set(GLA.grading)))
        if negWeights[-1]!=-1:
            raise AttributeError('`Tanaka_symbol` expects negatively graded LA to have a weight -1 component.')
        self.negWeights = negWeights
        if isinstance(nonnegParts,dict):
            nonNegWeights = sorted([k for k,v in nonnegParts.items() if len(v)!=0])
        else:
            nonNegWeights = sorted(tuple(set([j.compute_weight()[0] for j in nonnegParts])))
        if len(nonNegWeights)==0:
            self.height = -1
        else:
            self.height = nonNegWeights[-1]
        self.depth = negWeights[0]
        self.weights = negWeights+nonNegWeights
        GLA_levels = dict()
        for weight in negWeights:
            level = [j for j in GLA.basis if j.check_element_weight()[0]==weight]
            GLA_levels[weight]=level
        self.GLA_levels = GLA_levels

        if isinstance(nonnegParts,dict):
            self.nonneg_levels = nonnegParts
        else:
            nonneg_levels = dict()
            for weight in nonNegWeights:
                level = [j for j in nonnegParts if j.compute_weight()[0]==weight]
                nonneg_levels[weight]=level
            self.nonneg_levels = nonneg_levels
        levels = self.GLA_levels | self.nonneg_levels

        class dynamic_dict(dict):                   # special dict structure for the graded decomp.
            def __init__(self, dict_data, initial_index = None):
                super().__init__(dict_data)
                self.index_threshold = initial_index

            def __getitem__(self, key):
                # If index_threshold is None, behave like a regular dictionary
                if self.index_threshold is None:
                    return super().get(key, None)

                # Otherwise, apply the threshold logic
                if isinstance(key, int) and key >= self.index_threshold:
                    return []  # Return an empty list for keys > index_threshold
                return super().get(key, None)  # Default behavior otherwise

            def _set_index_thr(self, new_threshold):
                # Allow None or an integer as valid values
                if not (isinstance(new_threshold, (int,float,sp.Expr)) or new_threshold is None):
                    raise TypeError("index_threshold must be an integer or None.")
                self.index_threshold = new_threshold
        self._GLA_structure = dynamic_dict
        self.levels = dynamic_dict(levels, initial_index = index_threshold)
        self._test_commutators = None

    @property
    def test_commutators(self):
        if self._test_commutators:
            return self._test_commutators
        if self.assume_FGLA:
            deeper_levels = sum([self.GLA_levels[j] for j in self.negWeights[:-1]],[])
            f_level = self.GLA_levels[-1]
            first_commutators = [(f_level[j],f_level[k],f_level[j]*f_level[k]) for j in range(len(f_level)) for k in range(j+1,len(f_level))]
            remaining_comm = [(j,k,j*k) for j in f_level for k in deeper_levels]
            self._test_commutators = first_commutators+remaining_comm
            return first_commutators+remaining_comm
        else:
            neg_levels = sum([list(j) for j in (self.GLA_levels).values()],[])
            return [(neg_levels[j],neg_levels[k],neg_levels[j]*neg_levels[k]) for j in range(len(neg_levels)) for k in range(j+1,len(neg_levels))]

    def _prolong_by_1(self, levels, height): # start with self.levels, self.height (height must match levels structure)
        if self.assume_FGLA and len(levels[height])==0:
            new_levels = levels
            print(f'height: {height}, type: {type(height)}')
            new_levels._set_index_thr(height)
            stable = True
        else:
            ambiant_basis = []
            for weight in self.negWeights:
                ambiant_basis += [k@(j.dual()) for j in self.GLA_levels[weight] for k in levels[height+1+weight]]

            varLabel=create_key(prefix="center_var")   # label for temparary variables
            variableProcedure(
                varLabel,
                len(ambiant_basis),
                _tempVar=retrieve_passkey()
            )
            tVars = _cached_caller_globals[varLabel]    # pointer to tuple of coef vars

            general_elem = sum([tVars[j]*ambiant_basis[j] for j in range(1, len(tVars))],tVars[0]*ambiant_basis[0])
            eqns = []
            for triple in self.test_commutators:
                derivation_rule = (general_elem*triple[0])*triple[1]+triple[0]*(general_elem*triple[1])-general_elem*triple[2]
                if isinstance(derivation_rule,tensorProduct):
                    eqns += list(derivation_rule.coeff_dict.values())
                elif isinstance(derivation_rule,AlgebraElement):
                    eqns += derivation_rule.coeffs
            eqns = list(set(eqns))
            if eqns == [0]:
                solution = [{}]
            else:
                solution = sp.solve(eqns,tVars,dict=True)
            if len(solution)==0:
                raise RuntimeError(f'`Tanaka_symbol.prolongation` failed at a step where sympy.solve was being applied. The equation system was {eqns} w.r.t. {tVars}')
            el_sol = general_elem.subs(solution[0])

            free_variables = tuple(set.union(*[set(sp.sympify(j).free_symbols) for j in el_sol.coeff_dict.values()]))

            return_list = []
            for var in free_variables:
                basis_element = el_sol.subs({var: 1}).subs(
                    [(other_var, 0) for other_var in free_variables if other_var != var]
                )
                return_list.append(basis_element)

            clearVar(*listVar(temporary_only=True), report=False)

            new_levels =  self._GLA_structure(levels | {height+1:return_list}, levels.index_threshold)
            stable = False
        return new_levels, stable

    def prolong(self, iterations, return_symbol=False, report_progress=False):
        if not isinstance(iterations, int) or iterations < 1:
            raise TypeError('`prolong` expects `iterations` to be a positive int.')
        levels = self.levels
        height = self.height
        stable = False
        if report_progress:
            prol_counter = 1
            def count_to_str(count):
                return f"{count}{'st' if count == 1 else 'nd' if count == 2 else 'rd' if count == 3 else 'th'}"

        for j in range(iterations):
            if stable:
                break
            levels, stable = self._prolong_by_1(levels, height)

            if report_progress:
                max_len = max(
                    max(len(str(weight)) for weight in levels.keys()),
                    max(len(str(len(basis))) for basis in levels.values())
                )

                weights = " │ ".join([str(weight).ljust(max_len) for weight in levels.keys()])
                dimensions = " │ ".join([str(len(basis)).ljust(max_len) for basis in levels.values()])
                weights = f"Weights    │ {weights}"
                dimensions = f"Dimensions │ {dimensions}"
                line_length = max(len(weights), len(dimensions)) + 1

                header_length = len("Weights    │ ")
                top_border = f"┌{'─' * (header_length - 1)}┬{'─' * (1+line_length - header_length)}┐"
                middle_border = f"├{'─' * (header_length - 1)}┼{'─' * (1+line_length - header_length)}┤"
                bottom_border = f"└{'─' * (header_length - 1)}┴{'─' * (1+line_length - header_length)}┘"

                print(f"After {count_to_str(prol_counter)} iteration:")
                print(top_border)
                print(f"│ {weights} │")
                print(middle_border)
                print(f"│ {dimensions} │")
                print(bottom_border)
                prol_counter += 1

            height += 1
        if return_symbol:
            new_nonneg_parts = []
            for key, value in levels.items():
                if key >= 0:
                    new_nonneg_parts += value
            return Tanaka_symbol(self.ambiantGLA, new_nonneg_parts, assume_FGLA=self.assume_FGLA, index_threshold=levels.index_threshold, _validated=True)
        else:
            return levels

    def summary(self, style=None, use_latex=None, display_length=500):
        """
        Generates a pandas DataFrame summarizing the Tanaka_symbol data, with optional styling and LaTeX rendering.

        Parameters
        ----------
        style : str, optional
            A string key to retrieve a custom pandas style from the style_guide.
        use_latex : bool, optional
            If True, formats the table with rendered LaTeX in the Jupyter notebook.
            Defaults to False.

        Returns
        -------
        pandas.DataFrame or pandas.io.formats.style.Styler
            A styled DataFrame summarizing the Tanaka_symbol data, optionally with LaTeX rendering.
        """

        # Determine use_latex only once
        if use_latex is None:
            use_latex = get_dgcv_settings_registry()['use_latex']

        # Prepare data for the DataFrame
        data = {
            "Weight": [],
            "Dimension": [],
            "Basis": [],
        }
        for weight, basis in self.levels.items():
            data["Weight"].append(weight)
            data["Dimension"].append(len(basis))
            if use_latex:
                latex_basis = ", ".join(
                    f"${sp.latex(b)}$" if not (sp.latex(b).startswith('$') and sp.latex(b).endswith('$')) else sp.latex(b)
                    for b in basis
                )
                data["Basis"].append(latex_basis)
            else:
                data["Basis"].append(", ".join(map(str, basis)))

        df = pd.DataFrame(data)
        if display_length is not None:
            def _cap_text(s):
                return s if len(s) <= display_length else "output too long to display; raise `display_length` to a higher bound if needed."
            df["Basis"] = df["Basis"].apply(_cap_text)
        df = df.sort_values(by="Weight").reset_index(drop=True)

        if style is None:
            dgcvSR=get_dgcv_settings_registry()
            style = dgcvSR['theme']
        pandas_style = get_style(style)

        # Extract styles
        border_style = "1px solid #ccc"  # Default border style
        hover_background = None
        hover_color = None

        # Utility to grab a style property value
        def extract_property(props, property_name):
            for name, value in props:
                if property_name in name:
                    return value
            return None

        # Extract border style
        for style_dict in pandas_style:
            if style_dict.get("selector") == "table":
                border_style = extract_property(style_dict.get("props", []), "border") or border_style
                break

        # Extract hover styles if defined
        if "hover" in pandas_style and "props" in pandas_style["hover"]:
            hover_background = extract_property(pandas_style["hover"]["props"], "background-color")
            hover_color = extract_property(pandas_style["hover"]["props"], "color")

        # Define styles: outer border, header bottom and vertical separators only
        additional_styles = [
            {"selector": "",     "props": [("border-collapse", "collapse"), ("border", border_style)]},
            {"selector": "th",   "props": [("border-bottom", border_style), ("border-right", border_style), ("text-align", "center")]},
            {"selector": "td",   "props": [("border-right", border_style), ("text-align", "center")]},
        ]

        # Apply hover styles to data cells if defined
        if hover_background or hover_color:
            additional_styles.append({
                "selector": "td:hover",
                "props": [
                    ("background-color", hover_background or "inherit"),
                    ("color", hover_color or "inherit"),
                ],
            })

        # Merge custom style list with additional styles
        table_styles = pandas_style + additional_styles

        # Apply styles
        if use_latex:
            # Convert the DataFrame to HTML with LaTeX
            styled_df = (
                df.style
                .hide(axis="index")  # Suppress the index column first
                .format({"Basis": lambda x: f"<div style='text-align: center;'>{x}</div>"})
                .set_caption("Summary of Tanaka Symbol (with prolongations)")
                .set_table_attributes('style="max-width:900px; table-layout:fixed; overflow-x:auto;"')
                .set_table_styles(table_styles)
            )
            return styled_df

        # Apply styles for the non-LaTeX version
        styled_df = (
            df.style
            .hide(axis="index")  # Suppress the index column first
            .set_caption("Summary of Tanaka Symbol (with prolongations)")
            .set_table_attributes('style="max-width:900px; table-layout:fixed; overflow-x:auto;"')
            .set_table_styles(table_styles)
        )

        return styled_df

    def __str__(self):
        result = ["Tanaka Symbol:"]
        result.append("Graded vector space with weights and dimensions:")
        for weight, basis in self.levels.items():
            dim = len(basis)
            basis_str = ", ".join(map(str, basis))
            result.append(f"  Weight {weight}: Dimension {dim}")
            result.append(f"    Basis: [{basis_str}]")
        return "\n".join(result)

    def _repr_latex_(self):
        lines = [
            r"\text{Tanaka Symbol:}\\",
            r"\text{Graded vector space with weights and dimensions:}\\",
            r"\begin{alignedat}{2}"
        ]
        for weight, basis in self.levels.items():
            dim = len(basis)
            basis_latex = ", ".join(map(lambda b: sp.latex(b), basis))
            lines.append(
                rf"&\text{{Weight {weight}: }} &\text{{Dimension {dim}, Basis: }} [{basis_latex}] \\"
            )
        lines.append(r"\end{alignedat}")
        return "$\n" + "\n".join(lines) + "\n$"

    def _sympystr(self, printer):
        result = ["Tanaka Symbol:"]
        result.append("Weights and Dimensions:")
        for weight, basis in self.levels.items():
            dim = len(basis)
            basis_str = ", ".join(printer.doprint(b) for b in basis)
            result.append(f"  {weight}: Dimension {dim}, Basis: [{basis_str}]")
        return "\n".join(result)

    def __repr__(self):
        return f"Tanaka_symbol(ambiantGLA={repr(self.ambiantGLA)}, levels={len(self.levels)} levels)"
