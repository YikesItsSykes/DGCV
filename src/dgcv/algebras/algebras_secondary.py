import warnings

import sympy as sp

from .._config import (
    dgcv_exception_note,
    get_dgcv_settings_registry,
    get_variable_registry,
)
from .._safeguards import (
    _cached_caller_globals,
    get_dgcv_category,
    retrieve_passkey,
    retrieve_public_key,
    validate_label,
    validate_label_list,
)
from ..combinatorics import carProd
from ..dgcv_core import variableProcedure
from ..solvers import solve_dgcv
from ..tensors import tensorProduct
from ..vmf import clearVar, listVar
from .algebras_aux import _validate_structure_data
from .algebras_core import (
    algebra_class,
    algebra_element_class,
    algebra_subspace_class,
)


class subalgebra_class(algebra_subspace_class):
    def __new__(cls, basis, alg, grading=None, _compressed_structure_data=None, _internal_lock=None):
        return super().__new__(cls, basis, alg, test_weights=None, _grading=grading, _internal_lock=_internal_lock)
    def __init__(self, basis, alg, grading=None, _compressed_structure_data=None, _internal_lock=None):
        super().__init__(basis, alg, test_weights=None, _grading=grading, _internal_lock=_internal_lock)
        basis = self.filtered_basis
        self.structureData = None
        if _internal_lock==retrieve_passkey():
            if _compressed_structure_data is not None:
                self.structureData=_compressed_structure_data
        if self.structureData is None:
            self.structureData = self.is_subalgebra(return_structure_data=True)['structure_data']
        self._structureData=tuple(map(tuple, self.structureData))
        self.subindices_to_ambiant_dict = {count:elem for count,elem in enumerate(basis)}
        self.basis_in_ambiant_alg = tuple(basis)
        self.basis = [subalgebra_element(self,[1 if j==count else 0 for j in range(self.dimension)],elem.valence) for count,elem in enumerate(basis)]
        if all(elem in self.ambiant for elem in basis):
            self.basis_labels = [elem.__str__() for elem in self.basis]
        else:
            self.basis_labels = [f'_e_{j+1}' for j in range(self.dimension)]
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='subalgebra'

# (self,alg,coeffs,valence,ambiant_rep=None,_internalLock=None)
    def _class_builder(self,coeffs,valence):
        return subalgebra_element(self,coeffs,valence)

    def __eq__(self, other):
        if not isinstance(other, subalgebra_class):
            return NotImplemented
        return (self.basis_in_ambiant_alg == other.basis_in_ambiant_alg)

    def __hash__(self):
        return hash(self.basis_in_ambiant_alg)


    def multiplication_table(self, elements=None, restrict_to_subspace=False, style=None, use_latex=None):
        if elements is None:
            newElements = [elem.ambiant_rep for elem in self.basis]
        elif isinstance(elements,(list,tuple)):
            warningMessage = ''
            newElements=[]
            for elem in elements:
                elemTest = elem.ambiant_rep if isinstance(elem,subalgebra_element) else elem
                if self.contains(elemTest) is False:
                    if warningMessage=='':
                        warningMessage+='Some elements in the `elements` list were not in the span of the subalgebra\'s basis, so they were omitted from the multiplication table.'
                else:
                    newElements.append(elem)
            if warningMessage!=0 and len(newElements)>0:
                warnings.warn(warningMessage)
            else:
                raise TypeError('No elements from the provided `elements` list belong to the subalgebra, so a multiplication table will not be produced.') from None
        else:
            raise TypeError('If provided, the `elements` parameter in `subalgebra_class.multiplication_table` must be a list.') from None

        return self.ambiant.multiplication_table(elements=newElements, restrict_to_subspace=restrict_to_subspace, style=style, use_latex=use_latex,_called_from_subalgebra={'internalLock':retrieve_passkey(),'basis':self.basis})

    def subalgebra(self,basis,grading=None):
        return self.ambiant.subalegra(basis,grading=grading)
    def subspace(self,basis,grading=None):
        return self.ambiant.subspace(basis,grading=grading)
    def contains(self, items, return_basis_coeffs = False):
        if not isinstance(items,(list,tuple)):
            items = [items]
        for item in items:
            if get_dgcv_category(item)=='subalgebra_element':
                if item.algebra==self:
                    bas=self.basis
                elif item.algebra.ambiant==self.ambiant:
                    item=item.ambiant_rep
                    bas=self.basis_in_ambiant_alg
                else:
                    return False
            elif get_dgcv_category(item)=='algebra_element' and item.algebra.ambiant==self.ambiant:
                item=item.ambiant_rep
                bas=self.basis_in_ambiant_alg
            else:
                return False
            if item not in bas:
                tempVarLabel = "T" + retrieve_public_key()
                variableProcedure(tempVarLabel, len(bas), _tempVar=retrieve_passkey())
                genElement = sum([_cached_caller_globals[tempVarLabel][j+1] * elem for j,elem in enumerate(bas[1:])],_cached_caller_globals[tempVarLabel][0]*(bas[0]))
                sol = solve_dgcv(item-genElement,_cached_caller_globals[tempVarLabel])
                if len(sol)==0:
                    clearVar(*listVar(temporary_only=True))
                    return False
            else:
                if return_basis_coeffs is True:
                    idx = bas.index(item)
                    return [1 if _==idx else 0 for _ in range(len(bas))]
        if return_basis_coeffs is True:
            vec=[var.subs(sol[0]) for var in _cached_caller_globals[tempVarLabel]]
            clearVar(*listVar(temporary_only=True))
            return vec
        clearVar(*listVar(temporary_only=True))
        return True

class subalgebra_element():
    def __init__(self,alg,coeffs,valence,ambiant_rep=None,_internalLock=None):
        self.algebra = alg
        self.vectorSpace = alg
        self.valence = valence
        if isinstance(coeffs, (list, tuple)):  
            self.coeffs = tuple(coeffs)
        else:
            raise TypeError("subalgebra_element expects coeffs to be a list or tuple.") from None
        if _internalLock==retrieve_passkey():
            self._ambiant_rep = ambiant_rep
        else:
            self._ambiant_rep = None
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='subalgebra_element'

    @property
    def ambiant_rep(self):
        if self._ambiant_rep is None:
            self._ambiant_rep = sum([coeff*self.algebra.subindices_to_ambiant_dict[j+1] for j,coeff in enumerate(self.coeffs[1:])],self.coeffs[0]*self.algebra.subindices_to_ambiant_dict[0])
        return self._ambiant_rep

    def __eq__(self, other):
        if not isinstance(other, subalgebra_element):
            return NotImplemented
        return (self.algebra == other.algebra and self.coeffs == other.coeffs and self.valence == other.valence)
    def __hash__(self):
        return hash((self.algebra, self.coeffs, self.valence))

    def __contains__(self, item):
        return item in self.basis

    def __iter__(self):
        return iter(self.basis) 

    def __getitem__(self, indices):
        if isinstance(indices,int):
            return self.basis[indices]
        elif isinstance(indices,list): 
            if len(indices)==1:
                return self.basis[indices[0]]
            elif isinstance(indices,list) and len(indices)==2:
                return self.structureData[indices[0]][indices[1]]
            elif isinstance(indices,list) and len(indices)==3:
                return self.structureData[indices[0]][indices[1]][indices[2]]
        else:
            raise TypeError(f'To access a subalgebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}') from None



    def __str__(self):
        return self.ambiant_rep.__str__()
    def _repr_latex_(self,verbose=False):
        return self.ambiant_rep._repr_latex_(verbose=verbose)
    def _latex(self,printer=None):
        return self._repr_latex_()
    def _sympystr(self):
        return self.ambiant_rep._sympystr()
    def _latex_verbose(self, printer=None):
        return self.ambiant_rep._latex_verbose(printer=printer)
    def __repr__(self):
        return self.ambiant_rep.__repr__()

    def is_zero(self):
        for j in self.coeffs:
            if sp.simplify(j) != 0:
                return False
        else:
            return True

    def subs(self, subsData):
        newCoeffs = [sp.sympify(j).subs(subsData) for j in self.coeffs]
        return subalgebra_element(self.algebra, newCoeffs, self.valence)

    def dual(self):
        return subalgebra_element(self.algebra, self.coeffs, (self.valence+1)%2)

    def _convert_to_tp(self):
        return tensorProduct(self.algebra,{(j,self.valence):self.coeffs[j] for j in range(self.algebra.dimension)})

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def __add__(self, other):
        if hasattr(other,"is_zero") and other.is_zero():
            return self
        if isinstance(other, subalgebra_element):
            if self.algebra == other.algebra and self.valence==other.valence:
                return subalgebra_element(
                    self.algebra,
                    [self.coeffs[j] + other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence,
                )
            else:
                raise TypeError(
                    "subalgebra_element operands for + must belong to the same subalgebra."
                ) from None
        return self.ambiant_rep.__add__(other)

    def __sub__(self, other):
        if hasattr(other,"is_zero") and other.is_zero():
            return self
        if isinstance(other, subalgebra_element):
            if self.algebra == other.algebra and self.valence==other.valence:
                return subalgebra_element(
                    self.algebra,
                    [self.coeffs[j] - other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence,
                )
            else:
                raise TypeError(
                    "subalgebra_element operands for - must belong to the same algebra."
                ) from None
        return self.ambiant_rep.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, subalgebra_element):
            if self.algebra == other.algebra and self.valence==other.valence:
                result_coeffs = [0] * self.algebra.dimension
                for i in range(self.algebra.dimension):
                    for j in range(self.algebra.dimension):
                        scalar_product = self.coeffs[i] * other.coeffs[j]
                        structure_vector_product = [
                            scalar_product * element
                            for element in self.algebra.structureData[i][j]
                        ]
                        result_coeffs = [
                            sp.sympify(result_coeffs[k] + structure_vector_product[k])
                            for k in range(len(result_coeffs))
                        ]
                return subalgebra_element(self.algebra, result_coeffs, self.valence)
            else:
                raise TypeError(
                    "Both operands for * must be subalgebra_element instances from the same algebra."
                ) from None
        elif isinstance(other, (int, float, sp.Expr)):
            new_coeffs = [coeff * other for coeff in self.coeffs]
            return subalgebra_element(self.algebra, new_coeffs, self.valence)
        elif isinstance(other,tensorProduct):
            return other.__rmul__(self)
        return self.ambiant_rep.__mul__(other)

    def __rmul__(self, other):
        if isinstance(
            other, (int, float, sp.Expr)
        ):
            return self * other
        elif isinstance(other, subalgebra_element):
            other * self
        elif isinstance(other,tensorProduct):
            return other.__mul__(self)
        return self.ambiant_rep.__rmul__(other)

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if isinstance(other, subalgebra_element) and other.algebra==self.algebra:
            new_dict = {(j,k,self.valence,other.valence):self.coeffs[j]*other.coeffs[k] for j in range(self.algebra.dimension) for k in range(self.algebra.dimension)}
            return tensorProduct(self.algebra, new_dict)
        return self.ambiant_rep.__matmul__(other)

    def __xor__(self, other):
        if other == '':
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    def __neg__(self):
        return -1*self

    def check_element_weight(self):
        """
        Determines the weight vector of this subalgebra_element with respect to its ambiant algebra's grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parent algebra' check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        return self.algebra.check_element_weight(self)

class simpleLieAlgebra(algebra_class):
    def __init__(self, structure_data, grading=None, format_sparse=False, process_matrix_rep=False, preferred_representation=None, _label=None, _basis_labels=None, _calledFromCreator=None, _callLock=None, _print_warning=None, _child_print_warning=None, _exclude_from_VMF=None,_simple_data=None):
        if _calledFromCreator!=retrieve_passkey():
            raise RuntimeError('`simpleLieAlgebra` class instances can only be initialized by internal `dgcv` functions indirectly.' ) from None
        super().__init__(structure_data, grading, format_sparse, process_matrix_rep, preferred_representation, _label, _basis_labels, _calledFromCreator, _callLock, _print_warning, _child_print_warning, _exclude_from_VMF)

        # assuming grading vectors are complete and given relative to simple roots
        self.roots=[]
        self.simpleRoots=[]
        self.rootSpaces={(0,)*len(self.grading):[]}
        def isSimpleRoot(vec):
            if vec.count(0)==len(vec)-1 and vec.count(1)==1:
                return True
            else:
                return False
        for elem in self.basis:
            root=tuple(elem.check_element_weight())
            if root in self.rootSpaces:
                self.rootSpaces[root].append(elem)
            else:
                self.rootSpaces[root]=[elem]
                self.roots.append(root)
                if isSimpleRoot(root):
                    self.simpleRoots.append(root)
        self.simpleRootSpaces = {root:self.rootSpaces[root] for root in self.simpleRoots}
        seriesLabel,rank = _simple_data['type']
        self.rank = rank
        self.Cartan_subalgebra = self.basis[0:rank]
        self.simpleLieType = f'{seriesLabel}{rank}' # example: "A3", "D4", ... etch

    def root_space_summary(self):
        def pluralize(idx):
            if idx!=1:
                return 's'
            else:
                return ''
        def rootString(idx):
            if idx==1: 
                return '(r_1)'
            if idx==2:
                return '(r_1, r_2)'
            if idx==3:
                return '(r_1, r_2, r_3)'
            else:
                return f'(r_1, ..., r_{idx})'
        print(f'This simple algebra {self.simpleLieType} has {self.rank} root{pluralize(self.rank)} {rootString(self.rank)}, which are dual to the Cartan subalgebra basis {self.Cartan_subalgebra}. These roots correspond to vertices in the Dynkin diagram as follows:\n')

        if self.simpleLieType[0] == 'D':
            n = self.rank
            if n == 2:
                print("Dynkin diagram for D2 is just two disconnected vertices corresponding to a direct sum of two u(2) copies.")
            else:
                lines = []
                horiz = "   "
                if n > 7:
                    mid_nodes = ["r_1 r_2", f"r_{n-4}", f"r_{n-3}", f"r_{n-2}"]
                    latter_rules = [" ◯───◯─"+" ┅ ─","─"*(len(mid_nodes[1])),"─"*(len(mid_nodes[2])),"─"*(len(mid_nodes[3])),""]
                    horiz += "◯".join(latter_rules)
                    top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
                    fork_pos = 16+len(latter_rules[1])+len(latter_rules[2])
                elif n>1:
                    horiz += "───".join("◯" for _ in range(n - 2))
                    top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n-2)])
                    horiz += "───◯"
                    fork_pos = 4*(n-2)-1
                else:
                    horiz += "───".join("◯" for _ in range(n - 2))
                    top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n-2)])
                    horiz += "───◯"
                    fork_pos = 4*(n-2)-1

                top_labels += " " + f"r_{n-1}"

                # Final node
                final_line = " " * fork_pos + "│"
                final_node = " " * fork_pos + f"◯ r_{n}"

                # bounding box
                width_bound= len(top_labels)
                title = "│" + self.simpleLieType.center(width_bound) + " │"
                border_top = "┌"+"─"*width_bound+"─┐"
                head_sep = "╞"+"═"*width_bound+"═╡"
                top_labels = "│" + top_labels + " │"
                horiz = "│" + horiz.ljust(width_bound) + " │"
                final_line = "│" + final_line.ljust(width_bound) + " │"
                final_node = "│" + final_node.ljust(width_bound) + " │"
                border_bottom= "└"+"─"*width_bound+"─┘"


                lines.append(border_top)
                lines.append(title)
                lines.append(head_sep)
                lines.append(top_labels)
                lines.append(horiz)
                lines.append(final_line)
                lines.append(final_node)
                lines.append(border_bottom)

                print("\n".join(lines))
        elif self.simpleLieType[0]=='B':
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n-3}", f"r_{n-2}", f"r_{n-1}"]
                latter_rules = [" ◯───◯─"+" ⋯ ─","─"*(len(mid_nodes[1])),"─"*(len(mid_nodes[2])),"═"*(len(mid_nodes[3])-2)+"═>",""]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n-1))
                top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n-1)])
                horiz += "══>◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound= len(top_labels)+1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌"+"─"*width_bound+"─┐"
            head_sep = "╞"+"═"*width_bound+"═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom= "└"+"─"*width_bound+"─┘"


            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))
        elif self.simpleLieType[0]=='A':
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n-3}", f"r_{n-2}", f"r_{n-1}"]
                latter_rules = [" ◯───◯─"+" ⋯ ─","─"*(len(mid_nodes[1])),"─"*(len(mid_nodes[2])),"─"*(len(mid_nodes[3])),""]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n-1))
                top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n-1)])
                horiz += "───◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound= len(top_labels)+1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌"+"─"*width_bound+"─┐"
            head_sep = "╞"+"═"*width_bound+"═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom= "└"+"─"*width_bound+"─┘"


            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))

    def parabolic_grading(self, roots=None):
        if roots is None:
            roots=[]
        if isinstance(roots,int):
            roots=[roots]
        elif not isinstance(roots,(list,tuple)):
            raise TypeError(f'The `roots` parameter in `simpleLieAlgebra.parabolic_grading(roots)` should be either `None`, an `int`, or a list of integers in the range (1,...,{self.rank}) representing indices of simple roots as enumerated in the algebras Dynkin diagram (see `simpleLieAlgebra.root_space_summary()` for a summary of this indexing).') from None
        gradingVector = [sum([self.grading[idx-1][j] for idx in roots]) for j in range(self.dimension)]
        denom = 1
        for weight in gradingVector:
            if isinstance(weight,sp.Rational):
                if denom<weight.denominator:
                    denom=weight.denominator
        if denom>1:
            gradingVector=[denom*weight for weight in gradingVector]
        return gradingVector

    def parabolic_subalgebra(self,roots=None,label=None,basis_labels=None,register_in_vmf=None, return_Alg=False,use_non_positive_weights=False, format_as_subalgebra_class=False):
        if roots is None:
            roots=[]
        if isinstance(roots,int):
            roots=[roots]
        elif not isinstance(roots,(list,tuple)):
            raise TypeError(f'The `roots` parameter in `simpleLieAlgebra.parabolic_subalgebra(roots)` should be either `None`, an `int`, or a list of integers in the range (1,...,{self.rank}) representing indices of simple roots as enumerated in the algebras Dynkin diagram (see `simpleLieAlgebra.root_space_summary()` for a summary of this indexing).') from None
        newGrading = [sum([self.grading[idx-1][j] for idx in roots]) for j in range(self.dimension)]
        if format_as_subalgebra_class is True:
            parabolic = []
        subIndices = []
        filtered_grading = []
        if not isinstance(use_non_positive_weights,bool):
            use_non_positive_weights = False
        for count, weight in enumerate(newGrading):
            if (weight>=0 and use_non_positive_weights is False) or (weight<=0 and use_non_positive_weights is True):
                if format_as_subalgebra_class is True:
                    parabolic.append(self.basis[count])
                subIndices.append(count)
                filtered_grading.append(weight)
        denom = 1
        for weight in filtered_grading:
            if isinstance(weight,sp.Rational):
                if denom<weight.denominator:
                    denom=weight.denominator
        if denom>1:
            filtered_grading=[denom*weight for weight in filtered_grading]

        def truncateBySubInd(li):
            return [li[j] for j in subIndices]
        structureData = truncateBySubInd(self._structureData)
        structureData = [truncateBySubInd(plane) for plane in structureData]
        structureData = [[truncateBySubInd(li) for li in plane] for plane in structureData]
        if format_as_subalgebra_class is True:
            ignoredList = []
            if label is not None:
                ignoredList.append('label')
            if basis_labels is not None:
                ignoredList.append('basis_labels')
            if register_in_vmf is True:
                ignoredList.append('register_in_vmf')
            if len(ignoredList)==1:
                warnings.warn(f'A parameter value was supplied for `{ignoredList[0]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so this parameter value was ignored. A subalgebra_class instance was returned instead.')
            elif len(ignoredList)==2:
                warnings.warn(f'Parameter values were supplied for `{ignoredList[0]}` and `{ignoredList[1]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so these parameter values were ignored. A subalgebra_class instance was returned instead.')
            elif len(ignoredList)==3:
                warnings.warn(f'Parameter values were supplied for `{ignoredList[0]}`, `{ignoredList[1]}`, and `{ignoredList[2]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so these parameter values were ignored. `A subalgebra_class instance was returned instead.`')
            return subalgebra_class(parabolic, self, grading=[filtered_grading], _compressed_structure_data=structureData, _internal_lock=retrieve_passkey())
        if isinstance(label,str) or isinstance(basis_labels,(list,tuple,str)):
            register_in_vmf=True
        if register_in_vmf is True:
            if label is None:
                label = self.label+'_parabolic'
            if basis_labels is None:
                basis_labels = label
            elif (isinstance(basis_labels,(list,tuple)) and not all(isinstance(elem,str) for elem in basis_labels)) or not isinstance(basis_labels,str):
                raise TypeError('If supplying the optional parameter `basis_labels` to `simpleLieAlgebra.parabolic_subalgebra` then it should be either a string or list of strings') from None
            createAlgebra(structureData,label=label,basis_labels=basis_labels,grading=filtered_grading)
            if return_Alg is True:
                return _cached_caller_globals[label]
        if return_Alg is True:
            return algebra_class(structureData,grading=[filtered_grading])
        elif register_in_vmf is not True:
            warnings.warn('Optional keywords for the `parabolic_subalgebra` method indicate that nothing should be return returned or registered in the vmf. Probably that is not intended, in which case at least one keyword `label`, `basis_labels`, `register_in_vmf`, `return_Alg`, or `format_as_subalgebra_class` should be set differently.')


def createSimpleLieAlgebra(
    series: str,
    label: str = None,
    basis_labels: list = None,
    build_standard_mat_rep = False
):
    """
    Creates a simple (with 2 exceptions) complex Lie algebra specified from the classical
    series
        - A_n = sl(n+1)     for n>0
        - B_n = so(2n+1)    for n>0
        - C_n = sp(2n)      for n>0
        - D_n = so(2n)      for n>0 (not simple for n=1,2)


    Parameters
    ----------
    series : str
        The type and rank of the Lie algebra, e.g., "A1", "A2", ..., "Dn".
    label : str, optional
        Custom label for the Lie algebra. If not provided, defaults to a standard notation,
        like sl2 for A2 etc.
    basis_labels : list, optional
        Custom labels for the basis elements. If not provided, default labels will be generated.

    Returns
    -------
    algebra
        The resulting Lie algebra as an algebra instance.

    Raises
    ------
    ValueError
        If the series label is not recognized or not implemented.

    Notes
    -----
    - Currently supports only the A,B, and D series (special linear Lie algebras: A_n = sl(n+1), etc.).
    """
    # Extract series type and rank
    try:
        series_type, rank = series[0], int(series[1:])
        series_type = ''.join(c.upper() if c.islower() else c for c in series_type)
    except (IndexError, ValueError):
        raise ValueError(f"Invalid series format: {series}. Expected a letter 'A', 'B', 'C', 'D', 'E', 'F', or 'G' followed by a positive integer, like 'A1', 'B5', etc.") from None
    if rank <= 0:
            raise ValueError(f"Sequence index must be a positive integer, but got: {rank}.") from None

    def _generate_A_series_structure_data(n):
        matrix_dim = n+1

        # Basis elements
        hBasis = {'elems':dict(),'grading':dict()}
        offDiag = {'elems':dict(),'grading':dict()}

        repMatrix = [[0] * matrix_dim for _ in range(matrix_dim)]
        def elemWeights(idx1,idx2):
            wVec = []
            for idx in range(n):
                if idx1==idx:
                    if idx2==idx+1:
                        wVec.append(2)
                    else:
                        wVec.append(1)
                elif idx1==idx+1:
                    if idx2==idx:
                        wVec.append(-2)
                    else:
                        wVec.append(-1)
                elif idx2==idx:
                    wVec.append(-1)
                elif idx2==idx+1:
                    wVec.append(1)
                else:
                    wVec.append(0)
            return wVec

        for j in range(n+1):
            for k in range(j,n+1):
                # Diagonal (Cartan) element
                if j == k and j<n:
                    M = [row[:] for row in repMatrix]
                    for idx in range(n+1):
                        if idx==j+1:
                            M[idx][idx] = -1
                        elif idx==j:
                            M[idx][idx] = 1
                    hBasis['elems'][(j,k,0)] = M
                    hBasis['grading'][(j,k,0)] = [0]*n
                elif j!=k:
                    # off diagonal generators
                    MPlus = [row[:] for row in repMatrix]
                    MMinus = [row[:] for row in repMatrix]
                    MPlus[j][k]=1
                    MMinus[k][j]=1
                    offDiag['elems'][(j,k,1)] = MPlus
                    offDiag['grading'][(j,k,1)] = elemWeights(j,k)
                    offDiag['elems'][(k,j,1)] = MMinus
                    offDiag['grading'][(k,j,1)] = elemWeights(k,j)

        indexingKey = dict(enumerate(list(hBasis['grading'].keys())+list(offDiag['grading'].keys())))
        indexingKeyRev = {j:k for k,j in indexingKey.items()}
        LADimension = len(indexingKey)
        def _structureCoeffs(idx1,idx2):
            coeffs = [0]*LADimension
            if idx2==idx1:
                return coeffs
            if idx2<idx1:
                reSign = -1  
                idx2,idx1 = idx1,idx2
            else: 
                reSign = 1
            p10,p11,p12=indexingKey[idx1]
            p20,p21,p22=indexingKey[idx2]
            if p12==0:
                if p22==1:
                    coeffs[idx2]+=reSign*(int(p10==p20)-int(p10==p21)+int(p10+1==p21)-int(p10+1==p20))
            elif p12==1:
                if p22==1:
                    if p11==p20:
                        if p10==p21:
                            if p10<p11:
                                for idx in range(p10,p11):
                                    coeffs[indexingKeyRev[(idx,idx,0)]]=reSign
                            else:
                                for idx in range(p11,p10):
                                    coeffs[indexingKeyRev[(idx,idx,0)]]=-reSign
                        else:
                            coeffs[indexingKeyRev[(p10,p21,1)]]=reSign
                    elif p10==p21:
                        coeffs[indexingKeyRev[(p20,p11,1)]]=-reSign
            return coeffs


        _structure_data = [[_structureCoeffs(k,j) for j in range(LADimension)] for k in range(LADimension)]
        CartanSubalg=list(hBasis['elems'].values())
        matrixBasis=CartanSubalg+list(offDiag['elems'].values())
        gradingVecs=list(hBasis['grading'].values())+list(offDiag['grading'].values())
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_B_series_structure_data(n):
        matrix_dim = 2*n+1

        # Basis elements
        hBasis = {'elems':dict(),'grading':dict()}
        GPlus = {'elems':dict(),'grading':dict()}
        GMinus = {'elems':dict(),'grading':dict()}
        DPlus = {'elems':dict(),'grading':dict()}
        DMinus = {'elems':dict(),'grading':dict()}

        skew_symmetric = [[0] * matrix_dim for _ in range(matrix_dim)]
        def gPlusWeights(idx1,idx2):
            wVec = []
            for idx in range(n-1):
                if (idx1<=idx and idx2<=idx) or (idx1>idx and idx2>idx):
                    wVec.append(0)
                elif idx1<=idx:
                    wVec.append(1)
                else:
                    wVec.append(-1)
            wVec.append(0)
            return wVec
        def gMinusWeights(idx1,idx2):
            wVec = []
            sign = 1 if idx2<idx1 else -1
            for idx in range(n-1):
                if (idx1<=idx and idx2<=idx):
                    wVec.append(sign)
                elif (idx1>idx and idx2>idx):
                    wVec.append(-sign)
                else:
                    wVec.append(0)
            wVec.append(sign)
            return wVec
        def DWeights(idx1,sign):
            wVec = []
            for idx in range(n-1):
                if (idx1<=idx):
                    wVec.append(-sign*sp.Rational(1,2))
                else:
                    wVec.append(-sign*sp.Rational(-1,2))
            wVec.append(-sign*sp.Rational(1,2))
            return wVec

        for j, k in carProd(range(n), range(n)):
            # Diagonal (Cartan) element
            if j == k and j<n-1:
                M = [row[:] for row in skew_symmetric]
                for idx in range(n):
                    if idx>j:
                        M[2*idx][2*idx+1] = -sp.I/2
                        M[2*idx+1][2*idx] = sp.I/2
                    else:
                        M[2*idx][2*idx+1] = sp.I/2
                        M[2*idx+1][2*idx] = -sp.I/2
                hBasis['elems'][(j,k,0)] = M
                hBasis['grading'][(j,k,0)] = [0]*n
                if j+2==n:
                    M = [row[:] for row in skew_symmetric]
                    for idx in range(n):
                        M[2*idx][2*idx+1] = sp.I/2
                        M[2*idx+1][2*idx] = -sp.I/2
                    hBasis['elems'][(j+1,k+1,0)] = M
                    hBasis['grading'][(j+1,k+1,0)] = [0]*n
            elif j!=k:
                # “+” generator
                MPlus = [row[:] for row in skew_symmetric]
                MPlus[2*j][2*k]     =  1
                MPlus[2*k][2*j]     = -1
                MPlus[2*j+1][2*k+1] =  1
                MPlus[2*k+1][2*j+1] = -1
                MPlus[2*j][2*k+1]   =  sp.I
                MPlus[2*k+1][2*j]   = -sp.I
                MPlus[2*j+1][2*k]   = -sp.I
                MPlus[2*k][2*j+1]   =  sp.I
                GPlus['elems'][(j,k,1)] = MPlus
                GPlus['grading'][(j,k,1)] = gPlusWeights(j,k)

                # “–” generator
                if j<k:
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2*j][2*k]     =  1
                    MMinus[2*k][2*j]     = -1
                    MMinus[2*j+1][2*k+1] = -1
                    MMinus[2*k+1][2*j+1] =  1
                    MMinus[2*j][2*k+1]   =  sp.I
                    MMinus[2*k+1][2*j]   = -sp.I
                    MMinus[2*j+1][2*k]   =  sp.I
                    MMinus[2*k][2*j+1]   = -sp.I
                    GMinus['elems'][(j,k,-1)] = MMinus
                    GMinus['grading'][(j,k,-1)] = gMinusWeights(j,k)
                else: # k<j
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2*k][2*j]     =  1
                    MMinus[2*j][2*k]     = -1
                    MMinus[2*k+1][2*j+1] = -1
                    MMinus[2*j+1][2*k+1] =  1
                    MMinus[2*k][2*j+1]   = -sp.I
                    MMinus[2*j+1][2*k]   = sp.I
                    MMinus[2*k+1][2*j]   = -sp.I
                    MMinus[2*j][2*k+1]   = sp.I
                    GMinus['elems'][(j,k,-1)] = MMinus
                    GMinus['grading'][(j,k,-1)] = gMinusWeights(j,k)
        for j in range(n):
            MPlus = [row[:] for row in skew_symmetric]
            MMinus = [row[:] for row in skew_symmetric]
            MPlus[2*j][2*n] = 1
            MPlus[2*n][2*j] = -1
            MPlus[2*j+1][2*n] = sp.I
            MPlus[2*n][2*j+1] = -sp.I
            MMinus[2*j][2*n] = 1
            MMinus[2*n][2*j] = -1
            MMinus[2*j+1][2*n] = -sp.I
            MMinus[2*n][2*j+1] = sp.I
            DPlus['elems'][(j,2*n,2)] = MPlus
            DPlus['grading'][(j,2*n,2)] = DWeights(j,1)
            DMinus['elems'][(j,2*n,-2)] = MMinus
            DMinus['grading'][(j,2*n,-2)] = DWeights(j,-1)

        indexingKey = dict(enumerate(list(hBasis['grading'].keys())+list(GPlus['grading'].keys())+list(GMinus['grading'].keys())+list(DPlus['grading'].keys())+list(DMinus['grading'].keys())))
        indexingKeyRev = {j:k for k,j in indexingKey.items()}
        LADimension = len(indexingKey)
        CSDict = {idx:{0:1,n-1:1} if idx==0 else {idx:1,idx-1:-1} for idx in range(n)} # Cartan subalgebra basis transform indexing
        CSDictInv = {idx:{j:-1 if j>idx else 1 for j in range(n)} for idx in range(n-1)}|{n-1:{j:1 for j in range(n)}}
        # _cached_caller_globals['debug']={'indexingKey':indexingKey}
        def _structureCoeffs(idx1,idx2):
            coeffs = [0]*LADimension
            if idx2==idx1:
                return coeffs
            if idx2<idx1:
                reSign = -1  
                idx2,idx1 = idx1,idx2
            else: 
                reSign = 1
            idx1,idx2 = sorted([idx1,idx2])
            p10,p11,p12=indexingKey[idx1]
            p20,p21,p22=indexingKey[idx2]
            if p12==0:
                for term, scale in CSDictInv[p10].items():
                    if p22==1:
                        coeffs[idx2]+=scale*reSign*(int(term==p20)-int(term==p21))*sp.Rational(1,2)
                    elif p22==-1:
                        sign = -reSign if p20<p21 else reSign
                        coeffs[idx2] += scale*sign*(int(term==p20)+int(term==p21))*sp.Rational(1,2)
                    elif p22==2:
                        coeffs[idx2] += -scale*(int(term==p20))*sp.Rational(1,2)*reSign
                    elif p22==-2:
                        coeffs[idx2] += scale*(int(term==p20))*sp.Rational(1,2)*reSign
            elif p12==1:
                if p22==1:
                    if p11==p20:
                        if p10==p21:
                            #l(p10)-l(p11)
                            for t, s in CSDict[p10].items():
                                coeffs[t]+=reSign*4*s
                            for t, s in CSDict[p11].items():
                                coeffs[t]+=-reSign*4*s
                        else:
                            coeffs[indexingKeyRev[(p10,p21,1)]]+=2*reSign
                    elif p10==p21:
                        coeffs[indexingKeyRev[(p20,p11,1)]]+=-2*reSign
                elif p22==-1:
                    slope1 = 1 if p10<p11 else -1
                    slope2 = 1 if p20<p21 else -1
                    if p10==p20:
                        if not(slope1==-1 and slope2==-1):
                            if p11<p21:
                                coeffs[indexingKeyRev[(p11,p21,-1)]]+=-2*reSign
                            elif p21<p11:
                                if not(slope1==1 and slope2==-1):
                                    coeffs[indexingKeyRev[(p21,p11,-1)]]+=2*reSign
                    elif p11==p21:
                        if not(slope1==1 and slope2==1):
                            if p10<p20:
                                coeffs[indexingKeyRev[(p20,p10,-1)]]+=2*reSign
                            elif p20<p10:
                                if not(slope1==-1 and slope2==1):
                                    coeffs[indexingKeyRev[(p10,p20,-1)]]+=-2*reSign
                    elif p11==p20:
                        if not(slope1==1 and slope2==1) and not(slope1==-1 and slope2==1):
                            if p10<p21:
                                coeffs[indexingKeyRev[(p21,p10,-1)]]=-2*reSign
                            elif p21<p10:
                                coeffs[indexingKeyRev[(p10,p21,-1)]]=2*reSign
                    elif p10==p21:
                        if not(slope1==-1 and slope2==-1) and not(slope1==1 and slope2==-1):
                            if p11<p20:
                                coeffs[indexingKeyRev[(p11,p20,-1)]]=2*reSign
                            elif p20<p11:
                                coeffs[indexingKeyRev[(p20,p11,-1)]]=-2*reSign
                elif p22==2:
                    if p10==p20:
                        coeffs[indexingKeyRev[(p11,p21,2)]]=-2*reSign
                elif p22==-2:
                    if p11==p20:
                        coeffs[indexingKeyRev[(p10,p21,-2)]]=2*reSign
            elif p12==-1:
                slope1 = 1 if p10<p11 else -1
                slope2 = 1 if p20<p21 else -1
                if p22==-1:
                    sign2 = 1 if p10<p11 else -1
                    if (p10<p11 and p20<p21) or (p10>p11 and p20>p21):
                        pass
                    elif p11==p20:
                        if p10==p21:
                            # plus/minus (l(p10)+l(p11))
                            for t, s in CSDict[p10].items():
                                coeffs[t]+=sign2*reSign*4*s
                            for t, s in CSDict[p11].items():
                                coeffs[t]+=sign2*reSign*4*s
                        else:
                            if sign2==1:
                                coeffs[indexingKeyRev[(p21,p10,1)]]+=2*reSign*sign2
                            else:
                                coeffs[indexingKeyRev[(p10,p21,1)]]+=2*reSign*sign2
                    elif p10==p21:
                        if sign2==1:
                            coeffs[indexingKeyRev[(p20,p11,1)]]+=2*reSign*sign2
                        else:
                            coeffs[indexingKeyRev[(p11,p20,1)]]+=2*reSign*sign2
                    elif p10==p20 and p21!=p11:
                        if sign2==1:
                            coeffs[indexingKeyRev[(p21,p11,1)]]+=-2*reSign*sign2
                        else:
                            coeffs[indexingKeyRev[(p11,p21,1)]]+=-2*reSign*sign2
                    elif p11==p21 and p10!=p20:
                        if sign2==1:
                            coeffs[indexingKeyRev[(p20,p10,1)]]+=-2*reSign*sign2
                        else:
                            coeffs[indexingKeyRev[(p10,p20,1)]]+=-2*reSign*sign2
                elif p22==2:
                    if slope1==-1:
                        if p11==p20:
                            coeffs[indexingKeyRev[(p10,p21,-2)]]=-2*reSign
                        elif p10==p20:
                            coeffs[indexingKeyRev[(p11,p21,-2)]]=2*reSign
                elif p22==-2:
                    if p10==p20 and slope1==1:
                        coeffs[indexingKeyRev[(p11,p21,2)]]=-2*reSign
                    if p11==p20:
                        if slope1==1:
                            coeffs[indexingKeyRev[(p10,p21,2)]]=2*reSign
            elif p12==2:
                if p22==2:
                    if p10<p20:
                        coeffs[indexingKeyRev[(p10,p20,-1)]]=-reSign
                    elif p20<p10:
                        coeffs[indexingKeyRev[(p20,p10,-1)]]=reSign
                if p22==-2:
                    if p10==p20:
                        for term, scale in CSDict[p10].items():
                            coeffs[term]=2*scale*reSign
                    elif p10<p20:
                        coeffs[indexingKeyRev[(p20,p10,1)]]=reSign
                    else:
                        coeffs[indexingKeyRev[(p20,p10,1)]]=reSign
            elif p12==-2:
                if p22==-2:
                    if p10<p20:
                        coeffs[indexingKeyRev[(p20,p10,-1)]]=-reSign
                    elif p20<p10:
                        coeffs[indexingKeyRev[(p10,p20,-1)]]=reSign
            return coeffs


        _structure_data = [[_structureCoeffs(k,j) for j in range(LADimension)] for k in range(LADimension)]
        CartanSubalg=list(hBasis['elems'].values())
        matrixBasis=CartanSubalg+list(GPlus['elems'].values())+list(GMinus['elems'].values())+list(DPlus['elems'].values())+list(DMinus['elems'].values())
        gradingVecs=list(hBasis['grading'].values())+list(GPlus['grading'].values())+list(GMinus['grading'].values())+list(DPlus['grading'].values())+list(DMinus['grading'].values())
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def generate_C_series_structure_data(n):
        """
        Generates the structure data and weight vectors for the C_n series 
        (symplectic Lie algebra: sp(2n)).

        Parameters
        ----------
        n : int
            The rank of the C-series Lie algebra (2n is the matrix dimension).

        Returns
        -------
        tuple
            - basis (list): A 3-dimensional list representing the structure data for sp(2n).
                            Each element is a 2D list of lists representing a matrix.
            - weight_vectors (list): A single weight vector for the basis elements, 
                                    representing a grading of the algebra.

        Notes
        -----
        - Basis matrices are partitioned into nxn blocks and constructed in three groups:
        1. Lower block triangular: [[0, 0], [S_{j,k}, 0]] (weight = -1).
        2. Block diagonal: [[E_{j,k}, 0], [0, -E_{k,j}]] (weight = 0).
        3. Upper block triangular: [[0, S_{j,k}], [0, 0]] (weight = 1).
        """
        # Dimension of the full matrices
        matrix_dim = 2 * n

        # Basis elements and weight vector
        basis = []
        weight_vector = []

        # Step 1: Create symmetric matrices S_{j,k} = E_{j,k} + E_{k,j} for j ≤ k
        symmetric_matrices = []
        for j in range(n):
            for k in range(j, n):
                S = [[0] * n for _ in range(n)]
                S[j][k] = 1
                if j != k:
                    S[k][j] = 1
                symmetric_matrices.append(S)

        # Step 2: Create pairs P_{j,k} = (E_{j,k}, -E_{k,j})
        matrix_pairs = []
        for j in range(n):
            for k in range(n):
                P1 = [[0] * n for _ in range(n)]
                P2 = [[0] * n for _ in range(n)]
                P1[j][k] = 1
                P2[k][j] = -1
                matrix_pairs.append((P1, P2))

        # Step 3: Create basis matrices in three groups
        # Group 1: Lower block triangular [[0, 0], [S_{j,k}, 0]] (weight = -1)
        for S in symmetric_matrices:
            lower_triangular = [[0] * matrix_dim for _ in range(matrix_dim)]
            # Insert S into the lower-left block
            for i in range(n):
                for j in range(n):
                    lower_triangular[n + i][j] = S[i][j]
            basis.append(lower_triangular)
            weight_vector.append(-1)

        # Group 2: Block diagonal [[E_{j,k}, 0], [0, -E_{k,j}]] (weight = 0)
        for P1, P2 in matrix_pairs:
            block_diagonal = [[0] * matrix_dim for _ in range(matrix_dim)]
            # Insert P1 into the top-left block and P2 into the bottom-right block
            for i in range(n):
                for j in range(n):
                    block_diagonal[i][j] = P1[i][j]
                    block_diagonal[n + i][n + j] = P2[i][j]
            basis.append(block_diagonal)
            weight_vector.append(0)

        # Group 3: Upper block triangular [[0, S_{j,k}], [0, 0]] (weight = 1)
        for S in symmetric_matrices:
            upper_triangular = [[0] * matrix_dim for _ in range(matrix_dim)]
            # Insert S into the upper-right block
            for i in range(n):
                for j in range(n):
                    upper_triangular[i][n + j] = S[i][j]
            basis.append(upper_triangular)
            weight_vector.append(1)

        # Return basis and weight vector wrapped in a list (to allow multiple gradings)
        return basis, [weight_vector]

    def _generate_D_series_structure_data(n):
        matrix_dim = 2 * n

        # Basis elements
        hBasis = {'elems':dict(),'grading':dict()}
        GPlus = {'elems':dict(),'grading':dict()}
        GMinus = {'elems':dict(),'grading':dict()}

        skew_symmetric = [[0] * matrix_dim for _ in range(matrix_dim)]
        def gPlusWeights(idx1,idx2):
            wVec = []
            for idx in range(n-1):
                if (idx1<=idx and idx2<=idx) or (idx1>idx and idx2>idx):
                    wVec.append(0)
                elif idx1<=idx:
                    wVec.append(-1)
                else:
                    wVec.append(1)
            wVec.append(0)
            return wVec
        def gMinusWeights(idx1,idx2):
            wVec = []
            sign = -1 if idx2<idx1 else 1
            for idx in range(n-1):
                if (idx1<=idx and idx2<=idx):
                    wVec.append(sign)
                elif (idx1>idx and idx2>idx):
                    wVec.append(-sign)
                else:
                    wVec.append(0)
            wVec.append(sign)
            return wVec

        for j, k in carProd(range(n), range(n)):
            # Diagonal (Cartan) element
            if j == k and j<n-1:
                M = [row[:] for row in skew_symmetric]
                for idx in range(n):
                    if idx>j:
                        M[2*idx][2*idx+1] = -sp.I/2
                        M[2*idx+1][2*idx] = sp.I/2
                    else:
                        M[2*idx][2*idx+1] = sp.I/2
                        M[2*idx+1][2*idx] = -sp.I/2
                hBasis['elems'][(j,k,0)] = M
                hBasis['grading'][(j,k,0)] = [0]*n
                if j+2==n:
                    M = [row[:] for row in skew_symmetric]
                    for idx in range(n):
                        M[2*idx][2*idx+1] = sp.I/2
                        M[2*idx+1][2*idx] = -sp.I/2
                    hBasis['elems'][(j+1,k+1,0)] = M
                    hBasis['grading'][(j+1,k+1,0)] = [0]*n
            elif j!=k:
                # “+” generator
                MPlus = [row[:] for row in skew_symmetric]
                MPlus[2*j][2*k]     =  1
                MPlus[2*k][2*j]     = -1
                MPlus[2*j+1][2*k+1] =  1
                MPlus[2*k+1][2*j+1] = -1
                MPlus[2*j][2*k+1]   =  sp.I
                MPlus[2*k+1][2*j]   = -sp.I
                MPlus[2*j+1][2*k]   = -sp.I
                MPlus[2*k][2*j+1]   =  sp.I
                GPlus['elems'][(j,k,1)] = MPlus
                GPlus['grading'][(j,k,1)] = gPlusWeights(j,k)

                # “–” generator
                if j<k:
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2*j][2*k]     =  1
                    MMinus[2*k][2*j]     = -1
                    MMinus[2*j+1][2*k+1] = -1
                    MMinus[2*k+1][2*j+1] =  1
                    MMinus[2*j][2*k+1]   =  sp.I
                    MMinus[2*k+1][2*j]   = -sp.I
                    MMinus[2*j+1][2*k]   =  sp.I
                    MMinus[2*k][2*j+1]   = -sp.I
                    GMinus['elems'][(j,k,-1)] = MMinus
                    GMinus['grading'][(j,k,-1)] = gMinusWeights(j,k)
                else: # k<j
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2*k][2*j]     =  1
                    MMinus[2*j][2*k]     = -1
                    MMinus[2*k+1][2*j+1] = -1
                    MMinus[2*j+1][2*k+1] =  1
                    MMinus[2*k][2*j+1]   = -sp.I
                    MMinus[2*j+1][2*k]   = sp.I
                    MMinus[2*k+1][2*j]   = -sp.I
                    MMinus[2*j][2*k+1]   = sp.I
                    GMinus['elems'][(j,k,-1)] = MMinus
                    GMinus['grading'][(j,k,-1)] = gMinusWeights(j,k)

        indexingKey = dict(enumerate(list(hBasis['grading'].keys())+list(GPlus['grading'].keys())+list(GMinus['grading'].keys())))
        indexingKeyRev = {j:k for k,j in indexingKey.items()}
        LADimension = len(indexingKey)
        CSDict = {idx:{0:1,n-1:1} if idx==0 else {idx:1,idx-1:-1} for idx in range(n)} # Cartan subalgebra basis transform indexing
        CSDictInv = {idx:{j:-1 if j>idx else 1 for j in range(n)} for idx in range(n-1)}|{n-1:{j:1 for j in range(n)}}
        def _structureCoeffs(idx1,idx2):
            coeffs = [0]*LADimension
            if idx2==idx1:
                return coeffs
            if idx2<idx1:
                reSign = -1  
                idx2,idx1 = idx1,idx2
            else: 
                reSign = 1
            idx1,idx2 = sorted([idx1,idx2])
            p10,p11,p12=indexingKey[idx1]
            p20,p21,p22=indexingKey[idx2]
            if p12==0:
                for term, scale in CSDictInv[p10].items():
                    if p22==1:
                        coeffs[idx2]+=scale*reSign*(int(term==p20)-int(term==p21))*sp.Rational(1,2)
                    elif p22==-1:
                        sign = -reSign if p20<p21 else reSign
                        coeffs[idx2] += scale*sign*(int(term==p20)+int(term==p21))*sp.Rational(1,2)
            elif p12==1:
                if p22==1:
                    if p11==p20:
                        if p10==p21:
                            #l(p10)-l(p11)
                            for t, s in CSDict[p10].items():
                                coeffs[t]+=reSign*4*s
                            for t, s in CSDict[p11].items():
                                coeffs[t]+=-reSign*4*s
                        else:
                            coeffs[indexingKeyRev[(p10,p21,1)]]+=2*reSign
                    elif p10==p21:
                        coeffs[indexingKeyRev[(p20,p11,1)]]+=-2*reSign
                else:
                    slope1 = 1 if p10<p11 else -1
                    slope2 = 1 if p20<p21 else -1
                    if p10==p20:
                        if not(slope1==-1 and slope2==-1):
                            if p11<p21:
                                coeffs[indexingKeyRev[(p11,p21,-1)]]+=-2*reSign
                            elif p21<p11:
                                if not(slope1==1 and slope2==-1):
                                    coeffs[indexingKeyRev[(p21,p11,-1)]]+=2*reSign
                    elif p11==p21:
                        if not(slope1==1 and slope2==1):
                            if p10<p20:
                                coeffs[indexingKeyRev[(p20,p10,-1)]]+=2*reSign
                            elif p20<p10:
                                if not(slope1==-1 and slope2==1):
                                    coeffs[indexingKeyRev[(p10,p20,-1)]]+=-2*reSign
                    elif p11==p20:
                        if not(slope1==1 and slope2==1) and not(slope1==-1 and slope2==1):
                            if p10<p21:
                                coeffs[indexingKeyRev[(p21,p10,-1)]]=-2*reSign
                            elif p21<p10:
                                coeffs[indexingKeyRev[(p10,p21,-1)]]=2*reSign
                    elif p10==p21:
                        if not(slope1==-1 and slope2==-1) and not(slope1==1 and slope2==-1):
                            if p11<p20:
                                coeffs[indexingKeyRev[(p11,p20,-1)]]=2*reSign
                            elif p20<p11:
                                coeffs[indexingKeyRev[(p20,p11,-1)]]=-2*reSign
            else:
                sign2 = 1 if p10<p11 else -1
                if (p10<p11 and p20<p21) or (p10>p11 and p20>p21):
                    pass
                elif p11==p20:
                    if p10==p21:
                        # plus/minus (l(p10)+l(p11))
                        for t, s in CSDict[p10].items():
                            coeffs[t]+=sign2*reSign*4*s
                        for t, s in CSDict[p11].items():
                            coeffs[t]+=sign2*reSign*4*s
                    else:
                        if sign2==1:
                            coeffs[indexingKeyRev[(p21,p10,1)]]+=2*reSign*sign2
                        else:
                            coeffs[indexingKeyRev[(p10,p21,1)]]+=2*reSign*sign2
                elif p10==p21:
                    if sign2==1:
                        coeffs[indexingKeyRev[(p20,p11,1)]]+=2*reSign*sign2
                    else:
                        coeffs[indexingKeyRev[(p11,p20,1)]]+=2*reSign*sign2
                elif p10==p20 and p21!=p11:
                    if sign2==1:
                        coeffs[indexingKeyRev[(p21,p11,1)]]+=-2*reSign*sign2
                    else:
                        coeffs[indexingKeyRev[(p11,p21,1)]]+=-2*reSign*sign2
                elif p11==p21 and p10!=p20:
                    if sign2==1:
                        coeffs[indexingKeyRev[(p20,p10,1)]]+=-2*reSign*sign2
                    else:
                        coeffs[indexingKeyRev[(p10,p20,1)]]+=-2*reSign*sign2
            return coeffs


        _structure_data = [[_structureCoeffs(k,j) for j in range(LADimension)] for k in range(LADimension)]
        CartanSubalg=list(hBasis['elems'].values())
        matrixBasis=CartanSubalg+list(GPlus['elems'].values())+list(GMinus['elems'].values())
        gradingVecs=list(hBasis['grading'].values())+list(GPlus['grading'].values())+list(GMinus['grading'].values())
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    if series_type == "A":
        default_label = f"sl{rank + 1}" if label is None else label
        # Compute structure data for sl(n+1)
        structure_data, grading,CartanSubalgebra, matrixBasis = _generate_A_series_structure_data(rank)
        # Create and return the Lie algebra
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={'lockKey':passkey,'CartanSubalgebra':CartanSubalgebra,'type':[series_type,rank]}
            )
        else:
            return createAlgebra(structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={'lockKey':passkey,'CartanSubalgebra':CartanSubalgebra,'type':[series_type,rank]}
            )

    elif series_type == "B":
        default_label = f"so{2*rank + 1}" if label is None else label
        # Compute structure data for so(2n+1)
        structure_data, grading,CartanSubalgebra, matrixBasis = _generate_B_series_structure_data(rank)
        # Create and return the Lie algebra
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={'lockKey':passkey,'CartanSubalgebra':CartanSubalgebra,'type':[series_type,rank]}
            )
        else:
            return createAlgebra(structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={'lockKey':passkey,'CartanSubalgebra':CartanSubalgebra,'type':[series_type,rank]}
            )

    elif series_type == "C":
        raise ValueError("The C series (i.e., C_1=sp(2), C2=sp(4), ...) is not yet supported by `createSimpleLieAlgebra`. A future update will include support the C series.") from None

    elif series_type == "D":
        default_label = f"so{2*rank}" if label is None else label
        # Compute structure data for so(2n)
        structure_data, grading,CartanSubalgebra, matrixBasis = _generate_D_series_structure_data(rank)
        # Create and return the Lie algebra
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={'lockKey':passkey,'CartanSubalgebra':CartanSubalgebra,'type':[series_type,rank]}
            )
        else:
            return createAlgebra(structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={'lockKey':passkey,'CartanSubalgebra':CartanSubalgebra,'type':[series_type,rank]}
            )

    elif series_type+str(rank) in {'G2', 'F4', 'E6', 'E7','E8'}:
        raise ValueError("Exceptional Lie algebras are not yet supported by `createSimpleLieAlgebra`.") from None

    else:
        raise ValueError(f"Invalid series parameter format: {series}. Expected a letter 'A', 'B', 'C', 'D', 'E', 'F', or 'G' followed by a positive integer, like 'A1', 'B5', etc. For the exceptional LA labels 'E', 'F', and 'G' the integer must be among the classified types (i.e., only 'G2', 'F4', 'E6', 'E7', and 'E8' are admissible).") from None

def createFiniteAlg(
        obj,
        label,
        basis_labels=None,
        grading=None,
        format_sparse=False,
        process_matrix_rep=False,
        preferred_representation=None,
        verbose=False,
        assume_skew=False, 
        assume_Lie_alg=False,
        basis_order_for_supplied_str_eqns=None,
        _simple=None
    ):
    warnings.warn(
        "`createFiniteAlg` has been deprecated as it is being replaced with a more genera function. "
        "It will be removed in 2026. Use `createAlgebra` instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return createAlgebra(obj, label, basis_labels=basis_labels, grading=grading, format_sparse=format_sparse, process_matrix_rep=process_matrix_rep, preferred_representation=preferred_representation, verbose=verbose, assume_skew=assume_skew, assume_Lie_alg=assume_Lie_alg, basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns, _simple=_simple)
def createAlgebra(
    obj,
    label,
    basis_labels=None,
    grading=None,
    format_sparse=False,
    process_matrix_rep=False,
    preferred_representation=None,
    verbose=False,
    assume_skew=False, 
    assume_Lie_alg=False,
    basis_order_for_supplied_str_eqns=None,
    _simple=None
):
    """
    Registers an algebra object and its basis elements in the caller's global namespace,
    and adds them to the variable_registry for tracking in the Variable Management Framework.

    Parameters
    ----------
    obj : algebra, structure data, or list of algebra_element_class
        The algebra object (an instance of algebra), the structure data used to create one,
        or a list of algebra_element_class instances with the same parent algebra.
    label : str
        The label used to reference the algebra object in the global namespace.
    basis_labels : list, optional
        A list of custom labels for the basis elements of the algebra.
        If not provided, default labels will be generated.
    grading : list of lists or list, optional
        A list specifying the grading(s) of the algebra.
    format_sparse : bool, optional
        Whether to use sparse arrays when creating the algebra object.
    process_matrix_rep : bool, optional
        Whether to compute and store the matrix representation of the algebra.
    verbose : bool, optional
        If True, provides detailed feedback during the creation process.
    """

    if get_dgcv_category(obj)=='Tanaka_symbol':
        if grading is not None:
            warnings.warn('When processing a `Tanaka_symbol` object, `createAlgebra` uses the symbol\'s pre-defined grading rather than a manually supplied grading. You are getting this warning because an additional grading was manually supplied. To apply the custom grading instead, extract the symbol object\'s structure data using `Tanaka_symbol.export_algebra_data()`, and then pass that to `createAlgebra`.')
        symbolData = obj.export_algebra_data(_internal_call_lock=retrieve_passkey())
        if isinstance(symbolData,str):
            raise TypeError(symbolData+' So no `createAlgebra` did not instantiate a new algebra.') from None
        obj = symbolData['structure_data']
        grading = symbolData['grading']

    passkey = retrieve_passkey()
    if label in listVar(algebras_only=True) and get_dgcv_settings_registry()['forgo_warnings'] is not True:
        if isinstance(_simple,dict) and _simple.get('lockKey',None)==passkey:
            callFunction = 'createSimpleLieAlgebra'
        else:
            callFunction = 'createAlgebra'
        warnings.warn(f'`{callFunction}` was called with a `label` parameter already assigned to another algebra, so `{callFunction}` will overwrite the other algebra in the VMF and global namespace.')
        clearVar(label)

    def extract_structure_from_elements(elements):
        """
        Computes structure constants and validates linear independence from a list of algebra_element_class.

        Parameters
        ----------
        elements : list of algebra_element_class
            A list of algebra_element_class instances.

        Returns
        -------
        structure_data : list of lists of lists
            The structure constants for the subalgebra spanned by the elements.

        Raises
        ------
        ValueError
            If the elements are not linearly independent or not closed under the algebra product.
        """
        if not elements or not all(isinstance(el, algebra_element_class) for el in elements):
            raise ValueError(
                "Invalid input: All elements must be instances of algebra_element_class."
            ) from None

        # Check that all elements have the same parent algebra
        parent_algebra = elements[0].algebra
        if not all(el.algebra == parent_algebra for el in elements):
            raise ValueError(
                "All algebra_element_class instances must share the same parent algebra."
            ) from None

        try:
            # Use the parent algebra's is_subspace_subalgebra method
            result = parent_algebra.is_subspace_subalgebra(
                elements, return_structure_data=True
            )
        except ValueError as e:
            raise ValueError(
                "Error during subalgebra validation. "
                "The input list of algebra_element_class instances must be linearly independent and closed under the algebra product. "
                f"Original error: {e}"
            ) from e

        if not result["linearly_independent"]:
            raise ValueError(
                "The input elements are not linearly independent. "
            ) from None

        if not result["closed_under_product"]:
            raise ValueError(
                "The input elements are not closed under the algebra product. "
            ) from None

        # Return structure data
        return result["structure_data"]

    # Validate or create the algebra object
    if isinstance(obj, algebra_class):
        if verbose:
            print(f"Using existing algebra instance: {label}")
        structure_data = obj.structureData
        dimension = obj.dimension
    elif isinstance(obj, (list,tuple)) and all(isinstance(el, algebra_element_class) for el in obj):
        if verbose:
            print("Creating algebra from list of algebra_element_class instances.")
        structure_data = extract_structure_from_elements(obj)
        dimension = len(obj)
    else:
        if verbose:
            print("processing structure data...")
        try:
            structure_data = _validate_structure_data(
                obj, process_matrix_rep=process_matrix_rep, assume_skew=assume_skew, assume_Lie_alg=assume_Lie_alg, basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns
            )
        except dgcv_exception_note as e:
            raise SystemExit(e)
        dimension = len(structure_data)

    # Create or validate basis labels
    if basis_labels is None:
        basis_labels = [validate_label(f"{label}_{i+1}") for i in range(dimension)]
    elif isinstance(basis_labels,str):
        basis_labels = [validate_label(f"{basis_labels}_{i+1}") for i in range(dimension)]
    else:
        validate_label_list(basis_labels)

    # Process grading
    if grading is None:
        grading = [tuple([0] * dimension)]
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(w, (int, sp.Expr)) for w in grading
    ):
        # Single grading vector
        if len(grading) != dimension:
            raise ValueError(
                f"Grading vector length ({len(grading)}) must match the algebra dimension ({dimension})."
            ) from None
        grading = [tuple(grading)]
    elif isinstance(grading, list) and all(
        isinstance(vec, (list, tuple)) for vec in grading
    ):
        # List of grading vectors
        for vec in grading:
            if len(vec) != dimension:
                raise ValueError(
                    f"Grading vector length ({len(vec)}) must match the algebra dimension ({dimension})."
                ) from None
        grading = [tuple(vec) for vec in grading]
    else:
        raise ValueError("Grading must be a single vector or a list of vectors.") from None

    if isinstance(_simple,dict) and _simple.get('lockKey',None)==passkey:
        algebra_obj = simpleLieAlgebra(
            structure_data=structure_data,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey,
            _simple_data = _simple
        )
    else:
        algebra_obj = algebra_class(
            structure_data=structure_data,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey,
        )

    assert (
        algebra_obj.basis is not None
    ), "Algebra object basis elements must be initialized."

    _cached_caller_globals.update({label: algebra_obj})
    _cached_caller_globals.update(zip(basis_labels, algebra_obj.basis))

    variable_registry = get_variable_registry()
    variable_registry["finite_algebra_systems"][label] = {
        "family_type": "algebra",
        "family_names": tuple(basis_labels),
        "family_values": tuple(algebra_obj.basis),
        "dimension": dimension,
        "grading": grading,
        "basis_labels": basis_labels,
        "structure_data": structure_data,
    }
    variable_registry["_labels"][label] = {
        "path": ("finite_algebra_systems", label),
        "children": set(basis_labels)
    }

    if verbose:
        print(f"Algebra '{label}' registered successfully.")
        print(
            f"Dimension: {dimension}, Grading: {grading}, Basis Labels: {basis_labels}"
        )
