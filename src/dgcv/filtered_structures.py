import warnings

import pandas as pd
import sympy as sp

from ._config import get_dgcv_settings_registry
from ._safeguards import _cached_caller_globals, create_key, retrieve_passkey
from .algebras.algebras_core import (
    algebra_class,
    algebra_element_class,
    algebra_subspace_class,
)
from .algebras.algebras_secondary import createAlgebra, subalgebra_class
from .dgcv_core import (
    DFClass,
    VFClass,
    allToHol,
    allToReal,
    variableProcedure,
)
from .solvers import solve_dgcv
from .styles import get_style
from .tensors import tensorProduct
from .vector_fields_and_differential_forms import LieDerivative, annihilator, decompose
from .vmf import clearVar, listVar


class Tanaka_symbol(sp.Basic):
    """
    dgcv class representing a symbol-like object for Tanaka prolongation.

    Parameters
    ----------
    GLA : algebra, (with negatively graded Lie algebra structure)
        ambiant Lie algebra's negative part. The first entry in GLS.gading must be a list of negative weights, which will be used for the prolongation degrees

    Methods
    -------
    prolong

    Examples
    --------
    """
    def __new__(cls, GLA, nonnegParts = [], assume_FGLA = False, subspace = None, distinguished_subspaces = None, index_threshold = None, _validated = None):
        if _validated != retrieve_passkey():

            if not isinstance(GLA, (algebra_class,algebra_subspace_class,subalgebra_class)):
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` (which represents a generalized graded Lie algebra) to be an `algebra`, `sualgebra`, or `algebra_subspace_class`, and the first element of `GLA.grading` must contain negative weights (-depth,...,-1)."
                )
            elif len(GLA.grading)==0:
                    raise TypeError(
                        "`Tanaka_symbol` expects `GLA` to be a graded Lie algebra, but the supplied `GLA` has no grading assigned."
                    )
            elif isinstance(GLA.grading[0],(list,tuple)):
                if not all(j<=0 for j in GLA.grading[0]):
                    raise TypeError(
                        f"`Tanaka_symbol` expects `GLA` to be a graded Lie algebra (`algebra`, `algebra_subspace_class`, or `sualgebra` in particular) with non-positive weights in the first element of `GLA.grading`. Recieved grading data: {GLA.grading}"
                    )
            elif not all(j<=0 for j in GLA.grading):
                raise TypeError(
                    f"`Tanaka_symbol` expects `GLA` to be a graded Lie algebra (`algebra`, `algebra_subspace_class`, or `sualgebra` in particular) with non-positive weights in the first element of `GLA.grading`. Recieved grading data: {GLA.grading}."
                )

            if isinstance(nonnegParts,dict):
                NNPList = list(nonnegParts.values())
            elif isinstance(nonnegParts,(list,tuple)):
                NNPList = [nonnegParts]
            else:
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` built from the `algebra` given for `GLA` with `valence` of the form (1,0,...0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                )
            for NNP in NNPList:
                if not all(isinstance(j,tensorProduct) for j in NNP) or not all(j.vector_space==GLA for j in NNP):
                    raise TypeError(
                        "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra` given for `GLA` with `valence` of the form (1,0,...0).Or it can be a dictionariy whose keys are non-negative weights, and whose key-values are such lists."
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
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra` given for `GLA` with `valence` of the form (1,0,...0)."
                )

        obj = sp.Basic.__new__(cls, GLA, nonnegParts)
        return obj

    def __init__(self, GLA, nonnegParts = [], assume_FGLA = False, subspace = None, distinguished_subspaces = None, index_threshold = None, _validated = None):
        if subspace is None:
            noSubSSet = True
            subspace = GLA
        else:
            noSubSSet = False
            if not isinstance(subspace,(list,tuple)):
                raise TypeError(
                    "`Tanaka_symbol` expects `subpsace` to be a list of algebra_element_class instances belonging to the `algebra` `GLA`."
                )
            if not all(isinstance(j,algebra_element_class) for j in subspace) or not all(j.algebra==GLA for j in subspace):
                raise TypeError(
                    "`Tanaka_symbol` expects `subpsace` to be a list of algebra_element_class instances belonging to the `algebra` given for `GLA`."
                )
        DSProcessed = False
        self._default_to_characteristic_space_reductions=False
        if max(GLA.grading[0])==0:
            if noSubSSet is False:
                raise TypeError('`Tanaka_symbol` does not support setting the optional parameter `subspace` while also supplying `GLA` in the optional formatting with non-negative components. To initialize a symbol with specified subspace parameter then give only negative components in `GLA` and specify additional non-negative parts in the `nonnegParts` parameter.')
            DSProcessed = True
            GLABasis = GLA.basis
            if distinguished_subspaces and _validated!=retrieve_passkey():
                indexedDS = []
                if not isinstance(distinguished_subspaces,(list,tuple)):
                    raise TypeError(
                        "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol. General linear combinations of such are not supported."
                    )
                else:
                    for subS in distinguished_subspaces:
                        if not isinstance(subS,(list,tuple)):
                            raise TypeError(
                                "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol. General linear combinations of such are not supported."
                            )
                        indexedDS.append([])
                        for elem in subS:
                            try:
                                indexedDS[-1].append(GLABasis.index(elem))
                            except Exception:
                                raise TypeError(
                                    "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol. General linear combinations of such are not supported."
                                )
            else:
                indexedDS = [[]]

            def formatDegZero(elem):
                terms = []
                for elem1 in subspace:
                    if elem1.check_element_weight()[0]<0:
                        terms.append((elem*elem1)@elem1.dual())
                if len(terms)>0:
                    return sum(terms[1:],terms[0])
                else:
                    return 0*subspace[0]*subspace[0].dual()
            zeroPart = []
            zero_indices = {}
            nonzero_indices = {}
            localCount=0
            for loopCount,elem in enumerate(subspace):
                if GLA.grading[0][loopCount]==0:
                    zeroPart.append(formatDegZero(elem))
                    zero_indices[loopCount]=localCount
                    localCount+=1
                else:
                    nonzero_indices[loopCount]=loopCount-localCount
            subIndices = []
            filtered_grading = []
            truncateIndices = {}
            for count, weight in enumerate(GLA.grading[0]):
                if weight<0:
                    truncateIndices[count]=len(subIndices)
                    subIndices.append(count)
                    filtered_grading.append(weight)
            def truncateBySubInd(li):
                return [li[j] for j in subIndices]
            structureData = truncateBySubInd(GLA._structureData)
            structureData = [truncateBySubInd(plane) for plane in structureData]
            structureData = [[truncateBySubInd(li) for li in plane] for plane in structureData]
            # GLA = algebra(structureData,grading=filtered_grading,_exclude_from_VMF=retrieve_passkey())
            GLA = subalgebra_class(truncateBySubInd(GLA.basis),GLA,grading=[filtered_grading],_compressed_structure_data=structureData,_internal_lock=retrieve_passkey())
            def converCoeffDict(cd):
                newdict = {}
                for key,val in cd.items():
                    newdict[(truncateIndices[key[0]],truncateIndices[key[1]],1,0)]=val  # (...,1,0) for tensor valence
                return newdict
            zeroPart = [tensorProduct(GLA,converCoeffDict(elem.coeff_dict)) for elem in zeroPart]         

            subspace = GLA
            if len(nonnegParts)>0:
                warnings.warn('The GLA parameter provided to `Tanaka_symbol` has nonnegatively weighted components. If providing such `GLA` data then the optional `nonnegParts` cannot be manually set. So the provided manual setting for `nonnegParts` is being ignored.')
            nonnegParts = {0:zeroPart}
            distinguished_subspaces = []
            def populateDS(idx):
                if idx in zero_indices:
                    self._default_to_characteristic_space_reductions = True
                    return zeroPart[zero_indices[idx]]
                else:
                    return GLA.basis[nonzero_indices[idx]]
            for subS in indexedDS:
                distinguished_subspaces.append([])
                for idx in subS:
                    distinguished_subspaces[-1].append(populateDS(idx))

        self.negativePart = subspace
        self.ambiantGLA = GLA
        self.assume_FGLA = assume_FGLA
        self.nonnegParts = nonnegParts
        negWeights = sorted(tuple(set(GLA.grading[0]))) if isinstance(GLA.grading,(list,tuple)) else sorted(tuple(set(GLA.grading)))
        if negWeights and negWeights[-1] == 0:
            negWeights = negWeights[:-1]
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
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='Tanaka_symbol'

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
        if DSProcessed is not True:
            if distinguished_subspaces and _validated!=retrieve_passkey():
                if not isinstance(distinguished_subspaces,(list,tuple)):
                    raise TypeError(
                        "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol. General linear combinations of such are not supported."
                    )
                else:
                    testList = sum(list(self.levels.values()),[])
                    for subS in distinguished_subspaces:
                        if not isinstance(subS,(list,tuple)) or not all(elem in testList for elem in subS):
                            raise TypeError(
                                "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol. General linear combinations of such are not supported."
                            )
            else:
                distinguished_subspaces = [[]]
        self.distinguished_subspaces = distinguished_subspaces

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

    @property
    def basis(self):
        return sum(list(self.levels.values()),[])

    def __iter__(self):
        return iter(self.basis) 

    def _prolong_by_1(self, levels, height, distinguished_s_weight_bound = -1, with_characteristic_space_reductions=False): # height must match levels structure
        if self.assume_FGLA and len(levels[height])==0: # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        elif min(j for j in levels)>=-1-height and all(len(levels[height-j])==0 for j in range(-min(j for j in levels))):   # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        else:
            def validate_for_DS(tp,basisVec):
                for subS in self.distinguished_subspaces:
                    if basisVec in subS:
                        if tp not in subS:
                            return False
                return True
            ambiant_basis = []
            for weight in self.negWeights:
                ambiant_basis += [k@(j.dual()) for j in self.GLA_levels[weight] for k in levels[height+1+weight] if height+1+weight>distinguished_s_weight_bound or validate_for_DS(k,j)]
            if len(ambiant_basis)==0:
                ambiant_basis = [0*self.basis[0]]

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
                elif isinstance(derivation_rule,algebra_element_class):
                    eqns += derivation_rule.coeffs
            eqns = list(set(eqns))
            if eqns == [0]:
                solution = [{}]
            else:
                solution = solve_dgcv(eqns,tVars)
            if len(solution)==0:
                raise RuntimeError(f'`Tanaka_symbol.prolongation` failed at a step where sympy.solve was being applied. The equation system was {eqns} w.r.t. {tVars}')
            el_sol = general_elem.subs(solution[0])
            if hasattr(el_sol,'_convert_to_tp'):
                el_sol = el_sol._convert_to_tp()

            free_variables = tuple(set.union(*[set(sp.sympify(j).free_symbols) for j in el_sol.coeff_dict.values()]))

            new_level = []
            for var in free_variables:
                basis_element = el_sol.subs({var: 1}).subs(
                    [(other_var, 0) for other_var in free_variables if other_var != var]
                )
                new_level.append(basis_element)
            clearVar(*listVar(temporary_only=True), report=False)

            if len(new_level)>0 and with_characteristic_space_reductions is True and len(levels[0])>0:
                stabilized = False
                while stabilized is False:
                    ambiant_basis = new_level
                    varLabel=create_key(prefix="_cv")   # label for temparary variables
                    variableProcedure(varLabel,len(ambiant_basis),_tempVar=retrieve_passkey())
                    tVars = _cached_caller_globals[varLabel]    # pointers to tuple of coef vars
                    solVars = tVars
                    general_elem = sum([tVars[j]*ambiant_basis[j] for j in range(1, len(tVars))],tVars[0]*ambiant_basis[0])
                    eqns = []
                    for idx,dzElem in enumerate(levels[0]):
                        varLabel2=varLabel+f'{idx}_'
                        variableProcedure(varLabel2,len(ambiant_basis),_tempVar=retrieve_passkey())
                        solVars += _cached_caller_globals[varLabel2]
                        general_elem2 = sum([_cached_caller_globals[varLabel2][j]*ambiant_basis[j] for j in range(1, len(tVars))],_cached_caller_globals[varLabel2][0]*ambiant_basis[0])

                        commutator = general_elem*dzElem-general_elem2
                        if isinstance(commutator,tensorProduct):
                            eqns += list(commutator.coeff_dict.values())
                        elif isinstance(commutator,algebra_element_class):
                            eqns += commutator.coeffs
                    eqns = list(set(eqns))
                    solution = solve_dgcv(eqns,solVars)
                    if len(solution)==0:
                        raise RuntimeError(f'`Tanaka_symbol.prolongation` failed at a step where sympy.linsolve was being applied. The equation system was {eqns} w.r.t. {solVars}')
                    solCoeffs = [j.subs(solution[0]) for j in tVars]

                    free_variables = tuple(set.union(*[set(sp.sympify(j).free_symbols) for j in solCoeffs]))
                    new_vectors = []
                    for var in free_variables:
                        basis_element = [j.subs({var: 1}).subs([(other_var, 0) for other_var in free_variables if other_var != var]) for j in solCoeffs]
                        new_vectors.append(basis_element)
                    columns = (sp.Matrix(new_vectors).T).columnspace()
                    filtered_vectors = [list(sp.nsimplify(col, rational=True)) for col in columns]
                    def reScale(vec):
                        denom=1
                        for t in vec:
                            if hasattr(t,'denominator') and denom<t.denominator:
                                denom=t.denominator
                        if denom==1:
                            return list(vec)
                        else:
                            return [t*denom for t in vec]
                    filtered_vectors = [reScale(vec) for vec in filtered_vectors]

                    new_basis=[]
                    for coeffs in filtered_vectors:
                        new_basis.append(sum([coeffs[j]*ambiant_basis[j] for j in range(1,len(ambiant_basis))],coeffs[0]*ambiant_basis[0]))
                    clearVar(*listVar(temporary_only=True), report=False)
                    if len(new_basis)==0:
                        new_level=[]
                        stabilized = True
                    elif len(new_basis)<len(new_level):
                        new_level=new_basis
                    else:
                        new_level=new_basis
                        stabilized = True

            new_levels =  self._GLA_structure(levels | {height+1:new_level}, levels.index_threshold)
            stable = False
        return new_levels, stable

    def prolong(self, iterations, return_symbol=False, report_progress=False, report_progress_and_return_nothing=False, with_characteristic_space_reductions=None):
        if with_characteristic_space_reductions is None:
            with_characteristic_space_reductions=self._default_to_characteristic_space_reductions
        if report_progress_and_return_nothing is True:
            report_progress = True
        if not isinstance(iterations, int) or iterations < 1:
            raise TypeError('`prolong` expects `iterations` to be a positive int.')
        levels = self.levels
        height = self.height
        distinguished_s_weight_bound = self.height
        stable = False
        if report_progress:
            prol_counter = 1
            def count_to_str(count):
                return f"{count}{'st' if count == 1 else 'nd' if count == 2 else 'rd' if count == 3 else 'th'}"

        for j in range(iterations):
            if stable:
                break
            levels, stable = self._prolong_by_1(levels, height, distinguished_s_weight_bound=distinguished_s_weight_bound,with_characteristic_space_reductions=with_characteristic_space_reductions)

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
        if report_progress_and_return_nothing is not True:
            if return_symbol:
                new_nonneg_parts = []
                for key, value in levels.items():
                    if key >= 0:
                        new_nonneg_parts += value
                return Tanaka_symbol(self.ambiantGLA, new_nonneg_parts, assume_FGLA=self.assume_FGLA,distinguished_subspaces=self.distinguished_subspaces, index_threshold=levels.index_threshold, _validated=retrieve_passkey())
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

        if use_latex is None:
            use_latex = get_dgcv_settings_registry()['use_latex']
        def _to_string(element, ul=False):
            if ul:
                latex_str = element._repr_latex_(verbose=False)
                if latex_str.startswith('$') and latex_str.endswith('$'):
                    latex_str = latex_str[1:-1]
                latex_str = latex_str.replace(r'\\displaystyle', '').replace(r'\displaystyle', '').strip()
                return f'${latex_str}$'
            else:
                return str(element)
        # Prepare data for the DataFrame
        data = {
            "Weight": [],
            "Dimension": [],
            "Basis": [],
        }
        for weight, basis in self.levels.items():
            data["Weight"].append(weight)
            data["Dimension"].append(len(basis))
            data["Basis"].append(", ".join(map(lambda target: _to_string(target,ul=use_latex), basis)))

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

        # border style
        for style_dict in pandas_style:
            if style_dict.get("selector") == "table":
                border_style = extract_property(style_dict.get("props", []), "border") or border_style
                break

        # hover styles
        if "hover" in pandas_style and "props" in pandas_style["hover"]:
            hover_background = extract_property(pandas_style["hover"]["props"], "background-color")
            hover_color = extract_property(pandas_style["hover"]["props"], "color")

        # Define styles: outer border, header bottom and vertical separators only
        additional_styles = [
            {"selector": "",     "props": [("border-collapse", "collapse"), ("border", border_style)]},
            {"selector": "th",   "props": [("border-bottom", border_style), ("border-right", border_style), ("text-align", "center")]},
            {"selector": "td",   "props": [("border-right", border_style), ("text-align", "center")]},
        ]

        # Apply hover styles to data cells
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
        styled_df = (
            df.style
            .hide(axis="index")  # Suppress the index column first
            .set_caption("Summary of Tanaka Symbol (with prolongations)")
            .set_table_attributes('style="max-width:900px; table-layout:fixed; overflow-x:auto;"')
            .set_table_styles(table_styles)
        )

        return styled_df

    def __str__(self):
        levels = self.levels
        total_dim = sum(len(basis) for basis in levels.values())

        all_weights = list(levels.keys()) + ['total']
        all_dims = [len(basis) for basis in levels.values()] + [total_dim]

        max_len = max(
            max(len(str(w)) for w in all_weights),
            max(len(str(d)) for d in all_dims),
        )

        weights_row = " │ ".join(str(w).ljust(max_len) for w in all_weights)
        dims_row = " │ ".join(str(d).ljust(max_len) for d in all_dims)

        weights_line = f"Weights    │ {weights_row}"
        dims_line = f"Dimensions │ {dims_row}"
        line_len = max(len(weights_line), len(dims_line)) + 1
        header_len = len("Weights    │ ")

        top =    f"┌{'─' * (header_len - 1)}┬{'─' * (1+line_len - header_len)}┐"
        middle = f"├{'─' * (header_len - 1)}┼{'─' * (1+line_len - header_len)}┤"
        bottom = f"└{'─' * (header_len - 1)}┴{'─' * (1+line_len - header_len)}┘"

        result = ["Tanaka Symbol:", top, f"│ {weights_line} │", middle, f"│ {dims_line} │", bottom]
        return "\n".join(result)

    def _repr_latex_(self):
        levels = self.levels
        weights = list(levels.keys())
        dims = [len(basis) for basis in levels.values()]
        total_dim = sum(dims)

        # Format weights and dims as strings
        weights_row = " & ".join(map(str, weights)) + r" & \text{total} \\"
        dims_row = " & ".join(map(str, dims)) + rf" & {total_dim} \\"

        lines = [
            r"\textbf{Tanaka Symbol}\\[0.5em]",
            r"\begin{array}{|c||" + "c" * (len(weights)) + r"|c|}",
            r"\hline",
            r"\text{Weights} & " + weights_row,
            r"\hline",
            r"\text{Dimensions} & " + dims_row,
            r"\hline",
            r"\end{array}"
        ]
        return "$" + "\n".join(lines) + "$"

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

    def export_algebra_data(self,_internal_call_lock=None):
        grading_vec = []
        indexThresholds = [0]
        levelLengths = []
        for weight,level in self.levels.items():
            lLength = len(level)
            grading_vec += [weight]*lLength
            indexThresholds.append(indexThresholds[-1]+lLength)
            levelLengths.append(lLength)
        dimen = len(grading_vec)
        def flatToLayered(idx):
            for count, iT in enumerate(indexThresholds):
                if idx<iT:
                    return count+self.depth-1,idx-indexThresholds[count-1]
        def bracket_decomp(idx1,idx2):
            w1,sId1=flatToLayered(idx1)
            w2,sId2=flatToLayered(idx2)
            newElem = self.levels[w1][sId1]*self.levels[w2][sId2]
            newWeight = w1+w2
            ambiant_basis = self.levels[newWeight]
            nLDim= 0 if ambiant_basis is None else len(ambiant_basis)
            if nLDim==0:
                if newElem.is_zero():
                    return [0]*dimen
                else:
                    return 'NoSol'
            varLabel=create_key(prefix="_cv")   # label for temparary variables
            variableProcedure(varLabel,nLDim,_tempVar=retrieve_passkey())
            tVars = _cached_caller_globals[varLabel]    # pointers to tuple of coef vars
            general_elem = sum([tVars[j]*ambiant_basis[j] for j in range(1, len(tVars))],tVars[0]*ambiant_basis[0])
            sol=solve_dgcv([newElem-general_elem],tVars)
            if len(sol)==0:
                return 'NoSol'
            coeffVec = [var.subs(sol[0]) for var in tVars]
            clearVar(*listVar(temporary_only=True),report=False)
            if 0<=newWeight-self.depth and newWeight-self.depth<len(indexThresholds)-1: 
                start = [0]*indexThresholds[newWeight-self.depth]
                end = [0]*(dimen-indexThresholds[newWeight-self.depth]-nLDim)
                coeffVec = start+coeffVec+end
            else:
                end = [0]*(dimen-nLDim)
                coeffVec = coeffVec+end
            return coeffVec
        zeroVec = [0]*dimen
        str_data = [[bracket_decomp(k,j) if j<k else zeroVec for j in range(dimen)] for k in range(dimen)]
        for j in range(dimen):
            for k in range(j+1,dimen):
                skew_data=str_data[k][j]
                if skew_data=='NoSol':
                    warningStr = f'due to failure to confirm if the symbol data is closed under brackets between basis elements {j} and {k}.'
                    if _internal_call_lock!=retrieve_passkey():
                        warnings.warn('Unable to extract algebra structure, '+warningStr+' So `None` was returned by `export_algebra_data`.')
                        return None
                    return 'Unable to extract algebra structure from `Tanaka_symbol` object, '+warningStr
                str_data[j][k]=[-entry for entry in skew_data]
        return {'structure_data':str_data,'grading':[grading_vec]}



class distribution(sp.Basic):
    def __new__(cls, spanning_vf_set=None, spanning_df_set=None, assume_compatibility = False, check_compatibility_aggressively = False, _assume_minimal_Data = None):
        if spanning_vf_set is not None:
            if isinstance(spanning_vf_set,(list,tuple)):
                spanning_vf_set = tuple(spanning_vf_set)
                if not all(isinstance(vf,VFClass) for vf in spanning_vf_set):
                    raise TypeError('The `spanning_vf_set` keyword in `distribution` can only be assigned a list or tuple of `VFClass` instances')
            else:
                raise TypeError('The `spanning_vf_set` keyword in `distribution` can only be assigned a list or tuple of `VFClass` instances')
        if spanning_df_set is not None:
            if isinstance(spanning_df_set,(list,tuple)):
                spanning_df_set = tuple(spanning_df_set)
                if not all(isinstance(df,DFClass) and df.degree == 1 for df in spanning_df_set):
                    raise TypeError('The `spanning_df_set` keyword in `distribution` can only be assigned a list or tuple of `DFClass` instances. And they must all be degree 1.')
            else:
                raise TypeError('The `spanning_df_set` keyword in `distribution` can only be assigned a list or tuple of `DFClass` instances')
        if spanning_vf_set is not None and spanning_df_set is not None and assume_compatibility is False:
            for df in spanning_df_set:
                for vf in spanning_vf_set:
                    if check_compatibility_aggressively is True:
                        val = sp.simplify(df(vf))
                        if val != 0:
                            raise TypeError(f'Unnable to verify if the provided vector fields and differential forms annihilate each other. This may be due a failure in the program to recognize if {val} is zero. If that value in nonzero then the two sets do not define a comonn distribution. Set `assume_compatibility = True` to force initialization despite this compatibility check failing.')
                    else:
                        if df(vf) != 0:
                            raise TypeError('Unnable to verify if the provided vector fields and differential forms annihilate each other. This may be due a failure in the program to recognize if complex expression is zero. Set `check_compatibility_aggressively = True` to implement more expensive simplify methods in this step. Or set `assume_compatibility = True` to force initialization despite this compatibility check failing.')
        obj = sp.Basic.__new__(cls, spanning_vf_set, spanning_df_set, _assume_minimal_Data)
        return obj

    def __init__(self,spanning_vf_set=None,spanning_df_set=None, assume_compatibility = False, check_compatibility_aggressively = False, _assume_minimal_Data=None):
        self._prefered_data_type = 1 if spanning_df_set is None else 0
        if spanning_vf_set is None and spanning_df_set is None:
            self._spanning_vf_set = tuple()
            self._spanning_df_set = tuple()
        if spanning_vf_set is not None:
            self._spanning_vf_set, varSpace1, self._varSpace_type = self._validate_spanning_sets(spanning_vf_set)
            if spanning_df_set is not None:
                self._spanning_df_set, varSpace2, _ = self._validate_spanning_sets(spanning_df_set, target_type=self._varSpace_type)
                varSpace1.extend(varSpace2)
                self.varSpace = tuple(dict.fromkeys(varSpace1))
            self._spanning_df_set = None
            self.varSpace = tuple(varSpace1)
        else:
            self._spanning_vf_set = None
            self._spanning_df_set, self.varSpace, self._varSpace_type = self._validate_spanning_sets(spanning_df_set)
        if _assume_minimal_Data == retrieve_passkey():
            self._vf_basis = self._spanning_vf_set
            self._df_basis = self._spanning_df_set
        else:
            self._vf_basis = None
            self._df_basis = None
        self._derived_flag = None


    def _validate_spanning_sets(self,spanning_set,target_type=None):
        standardList = []
        realList = []
        complexList = []
        if target_type=='real' or target_type=='complex':
            primaryType = target_type
        else:
            primaryType = 'standard'
        for elem in spanning_set:
            if elem._varSpace_type=='standard':
                standardList.append(elem)
            if elem._varSpace_type=='real':
                realList.append(elem)
                if primaryType=='standard':
                    primaryType='real'
            if elem._varSpace_type=='complex':
                complexList.append(elem)
                if primaryType=='standard':
                    primaryType='complex'
        if primaryType=='complex':
            formattedList = tuple(standardList + complexList + [allToHol(elem) for elem in realList])
        elif primaryType=='real':
            formattedList = tuple(standardList + realList + [allToReal(elem) for elem in complexList])
        else:
            formattedList = tuple(standardList)

        varSpaceLoc = []
        for j in formattedList:
            varSpaceLoc.extend(j.varSpace)
        varSpaceList = list(dict.fromkeys(varSpaceLoc))
        return formattedList, varSpaceList, primaryType


    @property
    def spanning_vf_set(self):
        if self._spanning_vf_set is None:
            self._spanning_vf_set = annihilator(self.df_basis,self.varSpace)
            self._vf_basis = self._spanning_vf_set
        return self._spanning_vf_set

    @property
    def vf_basis(self):
        if self._vf_basis is None:
            if self._spanning_vf_set is None:
                return self.spanning_vf_set
            vfBasis = []
            for vf in self._spanning_vf_set:
                if decompose(vf,vfBasis,only_check_decomposability=True) is False:
                    vfBasis.append(vf)
            self._vf_basis = vfBasis
        return self._vf_basis

    @property
    def spanning_df_set(self):
        if self._spanning_df_set is None:
            self._spanning_df_set = annihilator(self.vf_basis,self.varSpace)
            self._df_basis = annihilator(self._spanning_df_set,self.varSpace)
        return self._spanning_df_set

    @property
    def df_basis(self):
        if self._df_basis is None:
            if self._spanning_df_set is None:
                return self.spanning_df_set
            dfBasis = []
            for df in self._spanning_df_set:
                if decompose(df,dfBasis,only_check_decomposability=True) is False:
                    dfBasis.append(df)
            self._df_basis = dfBasis
        return self._df_basis

    def derived_flag(self, max_iterations = 10):
        if self._derived_flag is None:
            tiered_list = [list(self.vf_basis)]
            if self._prefered_data_type==0:
                pass                
            def derive_extension(tieredList):
                baseL = tieredList[0]
                flattenedTL = sum(tieredList,[])
                newTeir = []
                topLevel = tieredList[-1]
                for vf1 in baseL:
                    for vf2 in topLevel:
                        bracket = LieDerivative(vf1,vf2)
                        if decompose(bracket,flattenedTL,only_check_decomposability=True) is False:
                            flattenedTL.append(bracket)
                            newTeir.append(bracket)
                return list(tieredList)+[newTeir]
            for _ in range(max_iterations):
                newTL = derive_extension(tiered_list)
                if len(newTL[-1])==0:
                    break
                else:
                    tiered_list = newTL
            self._derived_flag = tiered_list
        return self._derived_flag

    def nilpotent_approximation(self,expansion_point=None, label=None, basis_labels=None, exclude_from_VMF=False):
        """ expansion point should be a dictionary assigning numeric (float, int) values to the variables `distribution.varSpace`"""
        if expansion_point is None:
            expansion_point = {var:0 for var in self.varSpace}
        derFlag = self.derived_flag()
        depth = len(derFlag)
        basisVF = sum(derFlag,[])
        if len(basisVF)<len(self.varSpace):
            warnings.warn(f'The distribution is not bracket generating. A compliment to its bracket-generated envelope has been assigned weight {-depth} and added to the nilpotent approximation as a component commuting with everything.')
        elif len(basisVF)>len(self.varSpace):
            raise TypeError(f'The distribution is singular at the point {expansion_point}. Nilpotent approximations are not yet supported for singular distributions.')
        VFCoeffs = [[tuple((vf.subs(expansion_point)).coeffs) for vf in level] for level in derFlag]
        VFCFlattened = sum(VFCoeffs,[])
        level_dimensions = [len(level) for level in derFlag]
        def i_to_w_rule(idx):
            cap = 0
            for level, ld in enumerate(level_dimensions):
                cap+=ld
                if idx<cap:
                    return -1-level
            return -depth
        idx_to_weight_assignment = {j:i_to_w_rule(j) for j in range(len(VFCFlattened))}
        weight_to_level_assignment = {w:-1-w for w in idx_to_weight_assignment.values()}
        grading_vec = [idx_to_weight_assignment[idx] for idx in range(len(self.varSpace))]
        rank = sp.Matrix(VFCFlattened).rank()
        if rank < len(VFCFlattened):
            raise TypeError(f'The distribution is singular at the point {expansion_point}. Nilpotent approximations are not yet supported for singular distributions.')
        algebra_data = dict()
        VFC_enum = list(enumerate(basisVF))
        varLabel=create_key(prefix="temp_var_")   # label for temparary variables
        variableProcedure(varLabel,len(VFCFlattened),_tempVar=retrieve_passkey())
        tVars = _cached_caller_globals[varLabel]    # pointer to tuple of coef vars
        for count1,elem1 in VFC_enum:
            for count2,elem2 in VFC_enum[count1+1:]:
                newLevelWeight = idx_to_weight_assignment[count1]+idx_to_weight_assignment[count2]
                if newLevelWeight<min(weight_to_level_assignment.keys()):
                    resulting_coeffs = [0]*len(self.varSpace)
                else:
                    newCoeffs = (LieDerivative(elem1,elem2).subs(expansion_point)).coeffs
                    if len(self.varSpace)!= len(newCoeffs):
                        raise SystemError('DEBUG: distribution VF var spaces need alignment (add it to the distribution initializer). Please report this bug to `dgcv` maintainers.')
                    index_cap = sum(level_dimensions[:weight_to_level_assignment[newLevelWeight]+1])
                    spanning_set = VFCFlattened[:index_cap]
                    general_elem = [sum([tVars[count]*coeff[idx] for count,coeff in enumerate(spanning_set)]) for idx in range(len(self.varSpace))]
                    eqns = [j-k for j,k in zip(newCoeffs,general_elem)]
                    sol = solve_dgcv(eqns,tVars[:len(spanning_set)])
                    if len(sol)==0:
                        raise RuntimeError(f'Unable to compute the nilpotent approximation due to failure by internal solvers processing the equations {eqns}.')
                    resulting_coeffs = [0 if idx_to_weight_assignment[count]!=newLevelWeight else sol[0][var] for count, var in enumerate(tVars)]
                    algebra_data[(count1,count2)]=resulting_coeffs
        clearVar(*listVar(temporary_only=True),report=False)
        if label is None:
            printWarning = "This algebra was initialized via the `distribution.nilpotent_approximation` method, but no labeling was assigned. Intentionally obscur labels were therefore assignemed automatically. Use optional keywords `distribution.nilpotent_approximation(label='custom_label',basis_labels='list_of_labels')` for better labels. If such non-labeled initialization is really wanted, then use `distribution.nilpotent_approximation(exclude_from_VMF=True)` instead to suppress warnings such as this."
            childPrintWarning = "This algebraElement\'s parent algebra was initialized via the `distribution.nilpotent_approximation` method, but no labeling was assigned. Intentionally obscur labels were therefore assigned automatically. Use optional keywords `distribution.nilpotent_approximation(label='custom_label',basis_labels='list_of_labels')` for better labels. If such non-labeled initialization is really wanted, then use `distribution.nilpotent_approximation(exclude_from_VMF=True)` instead to suppress warnings such as this."
            exclusionPolicy = retrieve_passkey() if exclude_from_VMF is True else None
            return algebra_class(algebra_data,grading=[grading_vec],assume_skew=True,_callLock=retrieve_passkey(),_print_warning=printWarning,_child_print_warning=childPrintWarning,_exclude_from_VMF=exclusionPolicy)
        else:
            createAlgebra(algebra_data,label, basis_labels=basis_labels,grading=[grading_vec],assume_skew=True)




