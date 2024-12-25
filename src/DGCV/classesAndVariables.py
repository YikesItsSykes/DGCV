"""
DGCV: Differential Geometry with Complex Variables

This module defines the core classes and functions for the DGCV package, handling the creation 
and manipulation of vector fields, differential forms, tensor fields, algebras, and more. It includes 
tools for managing relationships between real and complex coordinate systems, object creation function,
and basic operations for its classes.

Key Classes:
    - DFClass: Represents differential forms.
    - VFClass: Represents vector fields.
    - STFClass: Represents symmetric tensor fields.
    - TFClass: Represents general tensor fields.
    - DGCVPolyClass: Hooks DGCV's complex variable handling into SymPy's Poly tools.
    - FAClass: Represents finite dimensional algebras.
    - AlgebraElement: Represents individual elements of algebras.

Key Functions:

Object Creation:
    - createVariables(): Initializes and labels coordinate systems of various types and 
    registers them within DGCV's Variable Management Framework.
    - createFiniteAlg(): Initializes and labels finite dimensional algebras and their elements.

Coordinate Conversion:
    - holToReal(): Converts holomorphic coordinates to their real counterparts.
    - realToSym(): Converts real coordinates to holomorphic with symbolic conjugate representations.
    - symToHol(): Converts symbolic conjugate coordinate representations to standard holomorphic coordinates.
    - holToSym(): Converts holomorphic coordinates to holomorphic with symbolic conjugate representations.
    - realToHol(): Converts real coordinates to holomorphic coordinates.
    - symToReal(): Converts symbolic conjugate coordinate representations to real coordinates.
    - allToReal(): Converts all types of coordinates to real coordinates.
    - allToHol(): Converts all types of coordinates to holomorphic coordinates.
    - allToSym(): Converts all types of coordinates to holomorphic with symbolic conjugate representations.

Structural Operations:
    - complex_struct_op(): Applies the complex structure operator to a vector field.
    - changeVFBasis(): Changes the coordinates space of a vector field.
    - changeDFBasis(): Changes the coordinates space of a differential form.
    - changeTFBasis(): Changes the coordinates space of a tensor field.
    - changeSTFBasis(): Changes the coordinates space of a symmetric tensor field.
    - VF_bracket(): Computes the Lie bracket of two vector fields.
    - exteriorProduct(): Computes the exterior product of differential forms.
    - tensorProduct(): Computes the tensor product of tensor fields.
    - scaleDF(): Scales a differential form.
    - addDF(): Adds two differential forms.
    - addVF(): Adds two vector fields.
    - scaleVF(): Scales a vector field.
    - addSTF(): Adds two symmetric tensor fields.
    - addTF(): Adds two tensor fields.
    - scaleTF(): Scales a tensor field.

Coefficients and Decompositions:
    - VF_coeffs(): Returns the coefficients of a vector field w.r.t. given coordinate vector 
    fields.
    - holVF_coeffs(): Returns the holomorphic coefficients of a vector field w.r.t. given 
    coordinate vector fields.
    - antiholVF_coeffs(): Returns the antiholomorphic coefficients of a vector field w.r.t. 
    given coordinate vector fields.
    - complexVFC(): Returns the holomorphic and antiholomorphic coefficients of a vector 
    field w.r.t. given coordinate vector fields.
    - realPartOfVF(): Extracts the real part of a vector field.
    - killingForm(): Computes the Killing form for a Lie algebra.
    - adjointRepresentation(): Returns the adjoint representation of a Lie algebra basis.

Variable Management:
    - listVar(): Lists the "parent names" of objects currently tracked by the DGCV VMF.
    - clearVar(): Clears the variables from the DGCV registry and deletes them from caller's globals().
    - DGCV_snapshot(): Takes a snapshot of the current DGCV VMF and reports a summary in a
    Pandas table.

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy
    - pandas

License:
    MIT License
"""


############## dependencies
import warnings
import re
import sympy
from sympy import Add, Array, Basic, conjugate, diff, I, ImmutableDenseNDimArray, ImmutableSparseNDimArray, latex, linsolve, Matrix, Mul, Poly, pretty, Rational, symbols, sympify, simplify, SparseMatrix, total_degree, Tuple, zeros

from random import randint
from pandas import DataFrame, MultiIndex
# from IPython.display import HTML

from .config import _cached_caller_globals, get_variable_registry, greek_letters
from ._safeguards import retrieve_passkey, retrieve_public_key, protected_caller_globals, validate_label, validate_label_list
from .combinatorics import chooseOp, permSign, carProd_with_weights_without_R, carProd
from .styles import get_style

############## classes

# differential form class
class DFClass(Basic):
    """
    A class representing symbolic differential forms in the DGCV package, with support for standard and complex differential forms.

    The DFClass represents differential forms where each component is a symbolic expression (using SymPy expressions) 
    corresponding to a differential form in the given variable space. It supports standard differential forms, 
    as well as complex differential forms with automatic handling of holomorphic and anti-holomorphic variables.

    Parameters:
    -----------
    varSpace : list
        The variables (coordinates) with respect to which the differential form is defined.
    
    coeffs : dict
        The coefficients for each variable combination in the differential form. The keys of the dictionary are tuples
        representing the indices of the variables, and the values are the coefficients.

    degree : int
        The degree of the differential form (number of variables in each wedge product).

    DGCVType : str, optional, default='standard'
        The type of differential form: 'standard' for real differential forms or 'complex' for complex differential forms.

    holVarSpace : list, optional
        For complex differential forms, specifies the holomorphic variable space. If not provided, it will be inferred
        from `varSpace` using the `variable_registry`.

    compVarSpace : list, optional
        For complex differential forms, specifies the full variable space including both holomorphic and conjugate variables.
        If not provided, it will be inferred from `varSpace`.

    Examples
    --------
    >>> from DGCV import DFClass, variableProcedure, DGCV_init_printing()
    >>> variableProcedure(['x', 'y'])
    >>> df = DFClass([x, y], {(0,): 1, (1,): -1}, 1)
    >>> df
    DFClass([x, y], {(0,): 1, (1,): -1}, 1)
    
    >>> df_latex = df._repr_latex_()  # Custom LaTeX formatting
    >>> print(df_latex) # display copy-able LaTex
    >>> DGCV_init_printing() # Nicely formats all DGCV and sympy objects displayed in Jupyter (or an iPython shell) 
    >>> display(df) # renders nicely format differential form display in Jupyter (or an iPython shell) 

    Attributes:
    -----------
    varSpace : list
        The variables with respect to which the differential form is defined.

    coeffs : dict
        The coefficients for each variable combination in the differential form.

    degree : int
        The degree of the differential form.

    DGCVType : str
        Indicates whether the differential form is standard ('standard') or complex ('complex').

    holVarSpace : tuple
        The holomorphic variable space for complex differential forms.

    compVarSpace : tuple
        The complete variable space (holomorphic and conjugate variables) for complex differential forms.

    Methods
    -------
    _repr_latex_ :
        Returns a custom LaTeX representation of the differential form, useful for rendering in Jupyter notebooks.
    
    __add__ :
        Custom addition method to allow for differential form addition.
    
    __sub__ :
        Custom subtraction method to allow for differential form subtraction.
    
    __mul__ and __rmul__ :
        Custom multiplication methods to scale differential forms by a scalar.
    
    __call__ :
        Apply the differential form to a set of vector fields, with optional handling for complex variables.

    Raises
    ------
    TypeError :
        Raised if operations (e.g., addition, subtraction) are attempted between incompatible types.
    
    ValueError :
        Raised if the differential form is applied to an unsupported argument type (e.g., incorrect number of arguments).
    """

    def __new__(cls, varSpace, data_dict, degree, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        # Call Basic.__new__ with only the positional arguments
        obj = Basic.__new__(cls, varSpace, data_dict, degree)
        return obj

    def __init__(self, varSpace, data_dict, degree, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        if len(varSpace)!=len(set(varSpace)):
            raise TypeError('`DFClass` expects the `varSpace` tuple not to have repeated variables.')
        self.varSpace = varSpace
        self.degree = degree
        self.simplifyKW = simplifyKW
        self.DGCVType = DGCVType
        variable_registry=get_variable_registry()
        if self.DGCVType == 'complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in self.varSpace):
                self._varSpace_type= 'real'
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in self.varSpace):
                self._varSpace_type = 'complex'
            else:
                raise KeyError('To initialize a DFClass instance with DGCVType=\'complex\', `varSpace` must contain only variables from DGCV\'s complex variables systems, and all variables in `varSpace` must be simulataneously among the real and imaginary types, or simulateneously among the holomorphic and antiholomorphic types. Use `complexVarProc` to easily create DGCV complex variable systems.')
        else:
            self._varSpace_type = 'standard'

        self.DFClassDataDict = DFClass.validateDFClassData(data_dict,self.degree)
        if self.DFClassDataDict == dict():
            self.DFClassDataDict = {(0,)*self.degree: 0}

        self.DFClassDataMinimal = [[list(a),b] for a,b in self.DFClassDataDict.items()]
        if self.DFClassDataMinimal == []:
            self.DFClassDataMinimal = [([0,] * degree, 0)]
        
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._coeff_dicts = None
        if self.degree == 0:
            self.coeffsInKFormBasis = [j[1] for j in self.DFClassDataMinimal]
            self.kFormBasisGenerators = [[1]]
        else:
            oneFormsLabelsLoc = ['d_' + str(j) for j in self.varSpace]
            self.coeffsInKFormBasis = [j[1] for j in self.DFClassDataMinimal]
            self.kFormBasisGenerators = [[oneFormsLabelsLoc[k] for k in j[0]] for j in self.DFClassDataMinimal]

    @staticmethod
    def validateDFClassData(dataDict,degr):
        if isinstance(dataDict,dict):
            newDict=dict()
            for a,b in dataDict.items():
                permS,orderedList=permSign(a,returnSorted=True)
                orderedTuple=tuple(orderedList)
                if orderedTuple in newDict:
                    if newDict[orderedTuple]!=permS*b:
                        raise TypeError('The `DFClass` initializers was given structure data that is does not appear skew-symmetric. Tip: try simplifying data furhter before initializing if you believe it is skew-symmetric.')
                else:
                    newDict[orderedTuple]=permS*b
            return newDict


    @property
    def realVarSpace(self):
        if self.DGCVType=='standard':
            return self._realVarSpace
        if self._realVarSpace == None or self._imVarSpace == None:
            self.coeff_dicts
            return self._realVarSpace+self._imVarSpace
        return self._realVarSpace+self._imVarSpace
    
    @property
    def holVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace
        if self._holVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace
        return self._holVarSpace
    
    @property
    def antiholVarSpace(self):
        if self.DGCVType=='standard':
            return self._antiholVarSpace
        if self._antiholVarSpace == None:
            self.coeff_dicts
            return self._antiholVarSpace
        return self._antiholVarSpace    
    
    @property
    def compVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace == None or self._antiholVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace + self._antiholVarSpace
        return self._holVarSpace + self._antiholVarSpace

    @property
    def coeff_dicts(self): # Retrieves coeffs in different variable formats and updates *VarSpace and _coeff_dicts caches if needed
        if self.DGCVType == 'standard' or all(j!=None for j in [self._realVarSpace,self._holVarSpace,self._antiholVarSpace,self._imVarSpace,self._coeff_dicts]):
            return self._coeff_dicts
        variable_registry=get_variable_registry()
        CVS=variable_registry['complex_variable_systems']
        if self._coeff_dicts == None:
            exhaust1 = list(self.varSpace)
            populate = {'compCoeffDataDict':dict(),'realCoeffDataDict':dict(),'holVarDict':dict(),'antiholVarDict':dict(),'realVarDict':dict(),'imVarDict':dict(),'preProcessMinDataToHol':dict(),'preProcessMinDataToReal':dict()}
            if self._varSpace_type == 'real':
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust1:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][2:])-{var}).pop()
                                if cousin in exhaust1:
                                    exhaust1.remove(cousin)
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'real':
                                    realVar = var
                                    exhaust1.remove(var)
                                    imVar = cousin
                                else:
                                    realVar = cousin
                                    exhaust1.remove(var)
                                    imVar = var
                                holVar = parent['variable_relatives'][varStr]['complex_family'][0]
                                antiholVar = parent['variable_relatives'][varStr]['complex_family'][1]
                                populate['holVarDict'][holVar]=[realVar,imVar]
                                populate['antiholVarDict'][antiholVar]=[realVar,imVar]
                                populate['realVarDict'][realVar]=[holVar,antiholVar]
                                populate['imVarDict'][imVar]=[holVar,antiholVar]
            else: # self._varSpace_type == 'complex'
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust1:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][:2])-{var}).pop()
                                if cousin in exhaust1:
                                    exhaust1.remove(cousin)
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'holomorphic':
                                    holVar = var
                                    exhaust1.remove(var)
                                    antiholVar = cousin
                                else:
                                    holVar = cousin
                                    exhaust1.remove(var)
                                    antiholVar = var
                                realVar = parent['variable_relatives'][varStr]['complex_family'][2]
                                imVar = parent['variable_relatives'][varStr]['complex_family'][3]
                                populate['holVarDict'][holVar]=[realVar,imVar]
                                populate['antiholVarDict'][antiholVar]=[realVar,imVar]
                                populate['realVarDict'][realVar]=[holVar,antiholVar]
                                populate['imVarDict'][imVar]=[holVar,antiholVar]
            self._realVarSpace = tuple(populate['realVarDict'].keys())
            self._holVarSpace = tuple(populate['holVarDict'].keys())
            self._antiholVarSpace = tuple(populate['antiholVarDict'].keys())
            self._imVarSpace = tuple(populate['imVarDict'].keys())
            
            if self.degree==0:
                if self._varSpace_type=='real':
                    populate['realCoeffDataDict'] = [self.varSpace,self.DFClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,{(0,)*self.degree:self.DFClassDataDict[(0,)*self.degree]}]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.DFClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,{(0,)*self.degree:self.DFClassDataDict[(0,)*self.degree]}]
            else:
                def _retrieve_indices(term,typeSet=None):
                    if typeSet == 'symb':
                        dictLoc= populate['realVarDict'] | populate['imVarDict']
                        refTuple = self._holVarSpace + self._antiholVarSpace
                        termList = dictLoc[term]
                    elif typeSet == 'real':
                        dictLoc= populate['holVarDict'] | populate['antiholVarDict']
                        refTuple = self._realVarSpace + self._imVarSpace
                        termList = dictLoc[term]
                    index_a = refTuple.index(termList[0])
                    index_b = refTuple.index(termList[1], index_a + 1)
                    return [index_a, index_b]

                if self._varSpace_type == 'real':
                    populate['preProcessMinDataToHol'] = {j:_retrieve_indices(self.varSpace[j],'symb') for j in range(len(self.varSpace))}
                else: # if self._varSpace_type == 'complex'
                    populate['preProcessMinDataToReal'] = {j:_retrieve_indices(self.varSpace[j],'real') for j in range(len(self.varSpace))}

                def decorateWithWeights(index,target='symb'):
                    if target=='symb':
                        if 2*index<len(self.varSpace):
                            holScale=Rational(1,2) # d_z coeff of d_x
                            antiholScale=Rational(1,2) # d_BARz coeff of d_x
                        else:
                            holScale=-I/2 # d_z coeff of d_y
                            antiholScale=I/2 # d_BARz coeff of d_y
                        return [[populate['preProcessMinDataToHol'][index][0],holScale],[populate['preProcessMinDataToHol'][index][1],antiholScale]]
                    else:# converting from hol to real
                        if 2*index<len(self.varSpace):
                            realScale=1 # d_x coeff in d_z
                            imScale=I # d_y coeff in d_z
                        else:
                            realScale=1 # d_x coeff of d_BARz
                            imScale=-I # d_y coeff of d_BARz
                        return [[populate['preProcessMinDataToReal'][index][0],realScale],[populate['preProcessMinDataToReal'][index][1],imScale]]

                def weightedOrdering(arg):
                    permS,orderedList=permSign(arg[0],returnSorted=True)
                    orderedTuple=tuple(orderedList)
                    return [orderedTuple,permS*arg[1]]

                otherDict=dict()
                for j in self.DFClassDataMinimal:
                    if self._varSpace_type=='real':
                        reformatTarget='symb'
                    else:
                        reformatTarget='real'
                    termIndices=[decorateWithWeights(k,target=reformatTarget) for k in j[0]]
                    prodWithWeights=carProd_with_weights_without_R(*termIndices)
                    prodWWRescaled=[[k[0],j[1]*k[1]] for k in prodWithWeights]
                    prodWithWeightedOrder=[weightedOrdering(j) for j in prodWWRescaled]
                    for term in prodWithWeightedOrder:
                        if term[0] in otherDict:
                            oldVal=otherDict[term[0]]
                            otherDict[term[0]]=(allToSym(oldVal+term[1]))
                        else:
                            otherDict[term[0]]=(allToSym(term[1]))

                if self._varSpace_type=='real':
                    populate['realCoeffDataDict'] = [self.varSpace,self.DFClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,otherDict]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.DFClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,otherDict]

            self._coeff_dicts = populate

            return populate
        else:
            return self._coeff_dicts


    def __repr__(self):
        return self._generate_representation()

    def __str__(self):
        return self._generate_representation()

    def _sympystr(self, printer):
        """
        Overrides SymPy's default string formatting method.
        This ensures that custom __repr__ is used when printing VFClass objects.
        """
        return self.__repr__()

    def _latex(self, printer=None):
        """
        Overrides SymPy's default LaTeX formatting method.
        This ensures that custom _repr_latex_ is used when calling sympy.latex().
        """
        return self._repr_latex_()
    
    def _generate_representation(self):
        if self.degree == 0:
            return str(self.coeffsInKFormBasis[0])

        basisLoc = ['*'.join(j) for j in self.kFormBasisGenerators]

        termsLoc = ''.join(
            [self._labelerLoc(self.coeffsInKFormBasis[j]) + basisLoc[j]
             for j in range(len(basisLoc)) if self.coeffsInKFormBasis[j] != 0]
        )
        if not termsLoc:
            termsLoc = '0*' + basisLoc[0]
        elif termsLoc[0] == '+':
            termsLoc = termsLoc[1:]

        return termsLoc

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the differential dorm for custom simplification. 

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.
        
        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.
        
        Returns
        -------
        DFClass
            A new DFClass instance with updated simplification settings.
        """
        if format_type not in {None, 'holomorphic','real', 'symbolic_conjugate'}:
            warnings.warn('simplify_format() recieved an unsupported first argument. Try None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\' instead.')
        return DFClass(self.varSpace,self.DFClassDataDict,self.degree,DGCVType=self.DGCVType, simplifyKW={'simplify_rule': format_type, 'simplify_ignore_list': skipVar})

    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self.simplifyKW attribute.

        Returns
        -------
        DFClass
            A simplified DFClass object.
        """
        if self.simplifyKW['simplify_rule']==None: 
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.DFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='holomorphic':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToHol(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.DFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='real':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToReal(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.DFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='symbolic_conjugate':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToSym(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.DFClassDataDict.items()}
        else:
            warnings.warn('_eval_simplify recieved an unsupported DFClass.simplifyKW[\'simplify_rule\']. It is recommend to only set the simplifyKW[\'simplify_rule\'] attribute to None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\'.')
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.DFClassDataDict.items()}

        # Return a new instance of DFClass with simplified coeffs

        # Return a new instance of DFClass with simplified coeffs
        return DFClass(self.varSpace,simplified_coeffs,self.degree,DGCVType=self.DGCVType,simplifyKW=self.simplifyKW)

    def subs(self,subsData):
        newDFData={a:sympify(b).subs(subsData) for a,b in self.DFClassDataDict.items()}
        return DFClass(self.varSpace, newDFData, self.degree, DGCVType=self.DGCVType, simplifyKW=self.simplifyKW)

    def _labelerLoc(self, coeff):
        if str(coeff)[0] == '-':
            return f'{coeff}*' if '+' not in str(coeff)[1:] and '-' not in str(coeff)[1:] else f'+({coeff})*'
        elif coeff == 1:
            return '+'
        elif coeff == -1:
            return '-'
        elif '+' in str(coeff) or '-' in str(coeff):
            return f'+({coeff})*'
        else:
            return f'+{coeff}*'

    def _convert_to_greek(self,var_name):
        for name, greek in greek_letters.items():
            if var_name.startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def _process_var_label(self,var):
        var_str = str(var)
        if var_str.startswith("BAR"):
            # Remove "BAR" prefix
            var_str = var_str[3:]
            
            # Regular expression to match label part and trailing number part
            match = re.match(r"([a-zA-Z_]+)(\d*)$", var_str)  # Match label followed by optional digits
            
            if match:
                label_part = match.group(1)  # The label part
                number_part = match.group(2)  # The number part (if any)

                # Remove trailing underscores in label part
                label_part = label_part.rstrip("_")

                # Convert label part to Greek if applicable
                label_part = self._convert_to_greek(label_part)

                # Return LaTeX formatted string
                return f"\\overline{{{label_part}_{{{number_part}}}}}" if number_part else f"\\overline{{{label_part}}}"

                
        else:
            # Return LaTeX for non-BAR variables
            return latex(var)

    def _repr_latex_(self):
        if self.degree == 0:
            return f"${self._process_var_label(self.coeffsInKFormBasis[0])}$"
        
        terms = []
        varLabels = [[self.varSpace[k] for k in j[0]] for j in self.DFClassDataMinimal]
        for coeff, var_list in zip(self.coeffsInKFormBasis, varLabels):
            if coeff == 0:
                continue
            latex_vars = [f"d {self._process_var_label(var)}" for var in var_list]
            differential_form = " \\wedge ".join(latex_vars)
            latex_coeff = self._format_coeff(coeff)
            term = f"{latex_coeff} {differential_form}"
            terms.append(term)
        
        if not terms:
            for coeff, var_list in zip(self.coeffsInKFormBasis, varLabels):
                latex_vars = [f"d {self._process_var_label(var)}" for var in var_list]
                differential_form = " \\wedge ".join(latex_vars)
                latex_coeff = self._format_coeff(coeff)
                term = f"{latex_coeff }{differential_form}"
                terms.append(term)
        
        latex_str = terms[0]
        for term in terms[1:]:
            latex_str += f" {term}" if term.startswith('-') else f" + {term}"

        return f"${latex_str}$"

    def _format_coeff(self, coeff):
        if coeff == 1:
            return ""
        elif coeff == -1:
            return "-"
        elif sympify(coeff).is_Atom or len(coeff.as_ordered_terms()) == 1:
            return latex(coeff)
        else:
            return f"\\left({latex(coeff)}\\right)"

    def is_zero(self):
        return all([simplify(allToReal(self.DFClassDataDict[a]))==0 for  a in self.DFClassDataDict])
         
    def __add__(self, other):
        """
        Adds two differential forms (i.e., DFClass instances). The addition is performed 
        by combining the coefficients of both forms. If the variable spaces of the forms 
        are not the same, the resulting form will have a variable space that is the union 
        of both input spaces.
        
        Parameters
        ----------
        other : DFClass
            Another differential form to add. The variable spaces of the two forms need not 
            be identical. The result will use the union of the variable spaces of the two 
            forms.
        
        Returns
        -------
        DFClass
            A new differential form representing the sum of the two input forms.
        
        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `DFClass`.
        
        Examples
        --------
        >>> from DGCV import DFClass, creatVariables
        >>> creatVariables('x1','x2')
        >>> d_x1 = DFClass([x1], {(0,): 1}, 1)
        >>> d_x2 = DFClass([x2], {(0,): 1}, 1)
        >>> d_sum = d_x1 + d_x2
        >>> print(d_sum)
        d_x1 + d_x2
        """
        if isinstance(other, DFClass):
            return addDF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with DFClass")

    def __sub__(self, other):
        """
        Subtracts one differential form from another (i.e., DFClass instances). The subtraction 
        is performed by negating the second form and then adding it to the first form. If the 
        variable spaces of the forms are not the same, the resulting form will have a variable 
        space that is the sum of both input spaces.
        
        Parameters
        ----------
        other : DFClass
            Another differential form to subtract. The variable spaces of the two forms need not 
            be identical. The result will use the sum of the variable spaces of the two forms.
        
        Returns
        -------
        DFClass
            A new differential form representing the difference between the two input forms.
        
        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `DFClass`.
        
        Examples
        --------
        >>> from DGCV import DFClass, creatVariables
        >>> creatVariables('x1','x2')
        >>> d_x1 = DFClass([x1], {(0,): 1}, 1)
        >>> d_x2 = DFClass([x2], {(0,): 1}, 1)
        >>> d_diff = d_x1 - d_x2
        >>> print(d_diff)
        d_x1 - d_x2
        """
        if isinstance(other, DFClass):
            return addDF(self, scaleDF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with DFClass")

    def __mul__(self, arg1):
        """
        Multiplication for DFClass objects. If the argument is a scalar (SymPy expression or numeric),
        the form is scaled by the scalar. If the argument is another DFClass object,
        the exterior (wedge) product is computed.
        
        Parameters
        ----------
        arg1 : sympy expression, numeric, or DFClass
            The object to multiply by. Can be a scalar function (SymPy expression or numeric) or
            another differential form (DFClass instance).

        Returns
        -------
        DFClass
            The result of scaling the form by the scalar or the exterior product of the
            two differential forms.

        Raises
        ------
        TypeError
            If the argument is neither a scalar (SymPy expression, numeric) nor a DFClass instance.
        """
        if isinstance(arg1, DFClass):
            # If the argument is another differential form, compute the exterior product
            return exteriorProduct(self, arg1)
        elif isinstance(arg1, (Basic, int, float, Rational)):  # Handle SymPy expressions and numeric types
            # If the argument is a scalar (SymPy expression or numeric), scale the form
            return scaleDF(arg1, self)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'DFClass' and '{type(arg1).__name__}'")

    def __rmul__(self, arg1):
        """
        Right multiplication for DFClass objects. If the argument is a scalar (SymPy expression or numeric),
        the form is scaled by the scalar. If the argument is another DFClass object,
        the exterior (wedge) product is computed.
        
        Parameters
        ----------
        arg1 : sympy expression, numeric, or DFClass
            The object to multiply by. Can be a scalar function (SymPy expression or numeric) or
            another differential form (DFClass instance).

        Returns
        -------
        DFClass
            The result of scaling the form by the scalar or the exterior product of the
            two differential forms.

        Raises
        ------
        TypeError
            If the argument is neither a scalar (SymPy expression, numeric) nor a DFClass instance.
        """
        if isinstance(arg1, DFClass):
            # If the argument is another differential form, compute the exterior product
            return exteriorProduct(arg1, self)
        elif isinstance(arg1, (Basic, int, float, Rational)):  # Handle SymPy expressions and numeric types
            # If the argument is a scalar (SymPy expression or numeric), scale the form
            return scaleDF(arg1, self)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'DFClass' and '{type(arg1).__name__}'")
    
    def __call__(self, *VFArgs):
        if self.degree == len(VFArgs):
            if self.degree == 0:
                return self.coeffsInKFormBasis[0]
            else:
                if all(isinstance(var,VFClass) for var in VFArgs):
                    if self.DGCVType == 'complex':
                        if self._varSpace_type == 'complex':
                            VFArgs=[allToSym(vf) if vf._varSpace_type!=self._varSpace_type else vf for vf in VFArgs]
                        else:
                            VFArgs=[allToReal(vf) if vf._varSpace_type!=self._varSpace_type else vf for vf in VFArgs]
                    if self.DGCVType == 'standard':
                        VFArgs=[_remove_complex_handling(vf) for vf in VFArgs]
                    inputCoeffs = contravariantVFTensorCoeffs(self.varSpace, *VFArgs)
                    # Access the elements of coeffArray directly
                    result = 0
                    for index, scalar in inputCoeffs:
                        permS,orderedList=permSign(index,returnSorted=True)
                        index=tuple(orderedList)
                        scalar = permS*scalar 
                        if index in self.DFClassDataDict:
                            result += self.DFClassDataDict[index] * scalar
                    return result
                else:
                    raise ValueError('`DFClass` can only operator on `VFClass` objects.')
        else:
            raise ValueError('A differential form received a number of arguments different from its degree.')

# vector field class
class VFClass(Basic):
    """
    A class representing symbolic vector fields in the DGCV package, with support for standard and complex vector fields.

    The VFClass represents vector fields where each component is a symbolic expression (using SymPy expressions)
    corresponding to a directional derivative in the given variable space. It supports standard vector fields,
    as well as complex vector fields with automatic handling for interaction with holomorphic, anti-holomorphic, real,
    and imaginary parts of DGCV complex variable systems.

    The class offers various methods for simplification, display, and symbolic manipulation, and integrates 
    with SymPy's simplify() function, allowing custom simplification rules for real, holomorphic, and symbolic conjugate 
    expressions within DGCV's complex variable systems. Aaddition, subtraction, and scaling of vector fields is supported

    Parameters
    ----------
    varSpace : list or tuple
        The variables (coordinates) with respect to which the vector field is defined.
    
    coeffs : list
        The coefficients for each variable in the vector field. Must be the same length as `varSpace`.

    DGCVType : str, optional
        The type of vector field: 'standard' for real vector fields or 'complex' for complex vector fields. Default is 'standard'.

    _holVarSpace : list or tuple, optional
        For complex vector fields, specifies the holomorphic variable space. If not provided, it will be inferred
        from `varSpace` using the `variable_registry`. This is intended for internal DGCV use to bypass the computation in populating `varSpace`.

    _compVarSpace : list or tuple, optional
        For complex vector fields, specifies the full variable space including both holomorphic and conjugate variables.
        If not provided, it will be inferred from `varSpace`. If not provided, it will be inferred
        from `varSpace` using the `variable_registry`. This is intended for internal DGCV use to bypass the computation in populating `varSpace`.

    simplifyKW : dict, optional
        A dictionary storing simplification options. Keys include:
        - 'simplify_rule': The rule for simplification ('real', 'holomorphic', or 'symbolic_conjugate').
        - 'simplify_ignore_list': A list of variables to exclude from the simplification process.

    Methods
    -------
    simplify_format(self, arg, skipVar=None)
        Prepares the vector field for custom simplification by setting the simplification rule and variables to ignore.
    
    _eval_simplify(self, **kwargs)
        Applies the simplification according to the current simplification settings.

    __repr__(self)
        Returns a string representation of the vector field, displaying the coefficients and the associated variables.
    
    _repr_latex_(self)
        Returns a custom LaTeX representation of the vector field for display in Jupyter notebooks.
    
    __add__(self, other)
        Adds another vector field (VFClass) to the current vector field.
    
    __sub__(self, other)
        Subtracts another vector field (VFClass) from the current vector field.

    __mul__(self, scalar)
        Multiplies the current vector field by a scalar (SymPy expression).
    
    __rmul__(self, scalar)
        Multiplies the current vector field by a scalar (SymPy expression).

    __call__(self, *args, ignore_complex_handling=None)
        Applies the vector field to a scalar function. Handles both real and complex variables depending on DGCVType.

    Examples
    --------
    Standard Variables:
    >>> variableProcedure(['a', 'b'])
    >>> vf1 = VFClass([a, b], [(a**2 + 2*a*b + b**2)/(a+b), a - b], 'standard')
    >>> display(vf1)  # Not simplified
    >>> display(simplify(vf1))  # Simplified using sympy.simplify()

    Complex Variables:
    >>> complexVarProc('z', 'x', 'y', 2)
    >>> complexVarProc('w', 'u', 'v')
    >>> vf2 = z1 * D_z2 + BARz1 * D_BARz2
    >>> vf3 = (x2 + I*y2) * u * D_x1 - I*(x2 + I*y2) * u * D_y1

    Simplification in Real Coordinates:
    >>> display(vf2)  # Not simplified
    >>> display(simplify(vf2.simplify_format('real')))  # Coefficients converted to real expressions

    Simplification in Holomorphic Coordinates with Variables to Ignore:
    >>> display(vf3)  # Not simplified
    >>> display(simplify(vf3.simplify_format('holomorphic', skipVar=['w'])))  # Coefficients converted to holomorphic expressions
    """

    def __new__(cls, varSpace, coeffs, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        if DGCVType == 'standard':
            assert len(varSpace) == len(coeffs), "varLabels and coeffs must have the same length"
        # Call Basic.__new__ with only the positional argument
        obj = Basic.__new__(cls, varSpace, coeffs)
        return obj

    def __init__(self, varSpace, coeffs, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        if len(varSpace)!=len(set(varSpace)):
            raise TypeError('`VFClass` expects the `varSpace` tuple not to have repeated variables.')
        self.varSpace = varSpace
        self.coeffs = coeffs
        self.DGCVType = DGCVType
        self.simplifyKW=simplifyKW
        variable_registry=get_variable_registry()
        if self.DGCVType == 'complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in self.varSpace):
                self._varSpace_type= 'real'
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in self.varSpace):
                self._varSpace_type = 'complex'
            else:
                raise KeyError('To initialize a VFClass instance with DGCVType=\'complex\', `varSpace` must contain only variables from DGCV\'s complex variables systems, and all variables in `varSpace` must be simulataneously among the real and imaginary types, or simulateneously among the holomorphic and antiholomorphic types. Use `complexVarProc` to create DGCV complex variable systems.')
        else:
            self._varSpace_type = 'standard'
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._coeff_dicts = None


    @property
    def realVarSpace(self):
        if self.DGCVType=='standard':
            return self._realVarSpace
        if self._realVarSpace == None or self._imVarSpace == None:
            self.coeff_dicts
            return self._realVarSpace+self._imVarSpace
        return self._realVarSpace+self._imVarSpace
    
    @property
    def holVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace
        if self._holVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace
        return self._holVarSpace
    
    @property
    def antiholVarSpace(self):
        if self.DGCVType=='standard':
            return self._antiholVarSpace
        if self._antiholVarSpace == None:
            self.coeff_dicts
            return self._antiholVarSpace
        return self._antiholVarSpace    
    
    @property
    def compVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace == None or self._antiholVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace + self._antiholVarSpace
        return self._holVarSpace + self._antiholVarSpace

    @property
    def coeff_dicts(self): # Retrieves coeffs in different variable formats and updates *VarSpace and _coeff_dicts caches if needed
        if self.DGCVType == 'standard':
            return self._coeff_dicts
        variable_registry=get_variable_registry()
        CVS=variable_registry['complex_variable_systems']
        if self._coeff_dicts == None:
            exhaust = dict(zip(self.varSpace,self.coeffs))
            populate = {'holCoeffs':{},'antiholCoeffs':{},'realCoeffs':{},'imCoeffs':{}}
            if self._varSpace_type == 'real':
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][2:])-{var}).pop()
                                if cousin in exhaust:
                                    cousinVal = exhaust[cousin]
                                    del exhaust[cousin]
                                else:
                                    cousinVal = 0
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'real':
                                    realVar = var
                                    realVal = exhaust[var]
                                    del exhaust[var]
                                    imVar = cousin
                                    imVal = cousinVal
                                else:
                                    realVar = cousinVal
                                    realVal = cousinVal
                                    del exhaust[var]
                                    imVar = var
                                    imVal = exhaust[var]
                                holVar = parent['variable_relatives'][varStr]['complex_family'][0]
                                holVal = realVal+I*imVal
                                antiholVar = parent['variable_relatives'][varStr]['complex_family'][1]
                                antiholVal = realVal-I*imVal
                                populate['holCoeffs'][holVar] = holVal
                                populate['antiholCoeffs'][antiholVar] = antiholVal
                                populate['realCoeffs'][realVar] = realVal
                                populate['imCoeffs'][imVar] = imVal
            else: # self._varSpace_type == 'complex'
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][:2])-{var}).pop()
                                if cousin in exhaust:
                                    cousinVal = exhaust[cousin]
                                    del exhaust[cousin]
                                else:
                                    cousinVal = 0
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'holomorphic':
                                    holVar = var
                                    holVal = exhaust[var]
                                    del exhaust[var]
                                    antiholVar = cousin
                                    antiholVal = cousinVal
                                else:
                                    antiholVar = var
                                    antiholVal = exhaust[var]
                                    del exhaust[var]
                                    holVar = cousin
                                    holVal = cousinVal
                                realVar = parent['variable_relatives'][varStr]['complex_family'][2]
                                realVal = Rational(1,2)*(holVal+antiholVal)
                                imVar = parent['variable_relatives'][varStr]['complex_family'][3]
                                imVal = -(I/2)*holVal+(I/2)*antiholVal
                                populate['holCoeffs'][holVar] = holVal
                                populate['antiholCoeffs'][antiholVar] = antiholVal
                                populate['realCoeffs'][realVar] = realVal
                                populate['imCoeffs'][imVar] = imVal
            self._coeff_dicts = populate
            self._realVarSpace = tuple([a for a in self._coeff_dicts['realCoeffs']])
            self._holVarSpace = tuple([a for a in self._coeff_dicts['holCoeffs']])
            self._antiholVarSpace = tuple([a for a in self._coeff_dicts['antiholCoeffs']])
            self._imVarSpace = tuple([a for a in self._coeff_dicts['imCoeffs']])
            return populate
        else:
            return self._coeff_dicts

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the vector field for custom simplification. 

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.
        
        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.
        
        Returns
        -------
        VFClass
            A new VFClass instance with updated simplification settings.
        """
        if format_type not in {None, 'holomorphic','real', 'symbolic_conjugate'}:
            warnings.warn('simplify_format() recieved an unsupported first argument. Try None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\' instead.')
        return VFClass(self.varSpace,self.coeffs,DGCVType=self.DGCVType, simplifyKW={'simplify_rule': format_type, 'simplify_ignore_list': skipVar})

    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self.simplifyKW attribute.

        Returns
        -------
        VFClass
            A simplified VFClass object.
        """
        if self.simplifyKW['simplify_rule'] is None: 
            simplified_coeffs = [simplify(j, **kwargs) for j in self.coeffs]
        elif self.simplifyKW['simplify_rule'] == 'holomorphic':
            simplified_coeffs = [simplify(allToHol(j, skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for j in self.coeffs]
        elif self.simplifyKW['simplify_rule'] == 'real':
            simplified_coeffs = [simplify(allToReal(j, skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for j in self.coeffs]
        elif self.simplifyKW['simplify_rule'] == 'symbolic_conjugate':
            simplified_coeffs = [simplify(allToSym(j, skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for j in self.coeffs]
        else:
            warnings.warn('_eval_simplify recieved an unsupported VFClass.simplifyKW[\'simplify_rule\']. It is recommend to only set the simplifyKW[\'simplify_rule\'] attribute to None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\'.')
            simplified_coeffs = [simplify(j, **kwargs) for j in self.coeffs]
        
        return VFClass(self.varSpace, simplified_coeffs, DGCVType=self.DGCVType, simplifyKW=self.simplifyKW)

    def subs(self,subsData):
        newCoeffs=[sympify(j).subs(subsData) for j in self.coeffs]
        return VFClass(self.varSpace, newCoeffs, DGCVType=self.DGCVType, simplifyKW=self.simplifyKW)

    def __repr__(self):
        basisLoc = ['D_' + str(j) for j in self.varSpace]
        termsLoc = ''.join([self._labelerLoc(self.coeffs[j]) + basisLoc[j] for j in range(len(basisLoc)) if self.coeffs[j] != 0])
        if not termsLoc:
            termsLoc = '0*' + basisLoc[0]
        elif termsLoc[0] == '+':
            termsLoc = termsLoc[1:]
        return termsLoc

    def _labelerLoc(self, coeff):
        """Helper function to format coefficients."""
        if str(coeff)[0] == '-':
            return f'{coeff}*' if '+' not in str(coeff)[1:] and '-' not in str(coeff)[1:] else f'+({coeff})*'
        elif coeff == 1:
            return '+'
        elif coeff == -1:
            return '-'
        elif '+' in str(coeff) or '-' in str(coeff):
            return f'+({coeff})*'
        else:
            return f'+{coeff}*'

    def _convert_to_greek(self,var_name):
        for name, greek in greek_letters.items():
            if var_name.startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def _process_var_label(self,var):
        var_str = str(var)
        if var_str.startswith("BAR"):
            # Remove "BAR" prefix
            var_str = var_str[3:]
            
            # Regular expression to match label part and trailing number part
            match = re.match(r"([a-zA-Z_]+)(\d*)$", var_str)  # Match label followed by optional digits
            
            if match:
                label_part = match.group(1)  # The label part
                number_part = match.group(2)  # The number part (if any)

                # Remove trailing underscores in label part
                label_part = label_part.rstrip("_")

                # Convert label part to Greek if applicable
                label_part = self._convert_to_greek(label_part)

                # Return LaTeX formatted string
                return f"\\overline{{{label_part}_{{{number_part}}}}}" if number_part else f"\\overline{{{label_part}}}"
        else:
            # Return LaTeX for non-BAR variables
            return latex(var)

    def _repr_latex_(self):
        terms = []
        for coeff, var in zip(self.coeffs, self.varSpace):
            if coeff == 0:
                continue
            # Process variable for BAR handling and LaTeX conversion
            latex_var = self._process_var_label(var)
            latex_coeff = self._format_coeff(coeff)
            terms.append(f"{latex_coeff}\\frac{{\\partial}}{{\\partial {latex_var}}}")
        
        if not terms:
            latex_var_0 = latex(self.varSpace[0])
            return f"$0 \\frac{{\\partial}}{{\\partial {latex_var_0}}}$"
        
        latex_str = terms[0]
        for term in terms[1:]:
            latex_str += f" {term}" if term.startswith('-') else f" + {term}"
        
        return f"${latex_str}$"

    def _format_coeff(self, coeff):
        """Format coefficients for LaTeX output."""
        if coeff == 1:
            return ""
        elif coeff == -1:
            return "-"
        elif sympify(coeff).is_Atom or len(coeff.as_ordered_terms()) == 1:
            return latex(coeff)
        else:
            return f"\\left({latex(coeff)}\\right)"

    def _sympystr(self, printer):
        """
        Overrides SymPy's default string formatting method.
        This ensures that custom __repr__ is used when printing VFClass objects.
        """
        return self.__repr__()

    def _latex(self, printer=None):
        """
        Overrides SymPy's default LaTeX formatting method.
        This ensures that custom _repr_latex_ is used when calling sympy.latex().
        """
        return self._repr_latex_()
    
    def __add__(self, other):
        """
        Adds two vector fields together. Both vector fields must be instances of `VFClass`.
        
        The addition is performed component-wise: the coefficients of each vector field are added together for 
        the corresponding variables in their respective variable spaces. If the variable spaces differ, the result 
        will have a union of both variable spaces.

        Parameters
        ----------
        other : VFClass
            Another vector field to be added to the current vector field.
        
        Returns
        -------
        VFClass
            A new `VFClass` instance representing the sum of the two vector fields.
        
        Raises
        ------
        TypeError
            If `other` is not an instance of `VFClass`.

        Examples
        --------
        >>> variableProcedure(['a', 'b'])
        >>> vf1 = VFClass([a, b], [a**2, b**2], 'standard')
        >>> vf2 = VFClass([a, b], [b**2, a**2], 'standard')
        >>> vf_sum = vf1 + vf2
        >>> display(vf_sum)  # Displays the sum of the two vector fields
        """
        if isinstance(other, VFClass):
            return addVF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with VFClass")

    def __sub__(self, other):
        """
        Subtracts one vector field from another. Both vector fields must be instances of `VFClass`.
        
        The subtraction is performed component-wise: the coefficients of each vector field are subtracted from each other 
        for the corresponding variables in their respective variable spaces. If the variable spaces differ, the result will 
        have a union of both variable spaces.

        Parameters
        ----------
        other : VFClass
            Another vector field to be subtracted from the current vector field.
        
        Returns
        -------
        VFClass
            A new `VFClass` instance representing the difference of the two vector fields.
        
        Raises
        ------
        TypeError
            If `other` is not an instance of `VFClass`.

        Examples
        --------
        >>> variableProcedure(['a', 'b'])
        >>> vf1 = VFClass([a, b], [a**2, b**2], 'standard')
        >>> vf2 = VFClass([a, b], [b**2, a**2], 'standard')
        >>> vf_diff = vf1 - vf2
        >>> display(vf_diff)  # Displays the difference between the two vector fields
        """
        if isinstance(other, VFClass):
            return addVF(self, scaleVF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with VFClass")

    def __mul__(self, scalar):
        """
        Multiplies the vector field by a scalar, which can be a SymPy expression or a number.
        
        The multiplication is performed by multiplying each coefficient in the vector field by the scalar.

        Parameters
        ----------
        scalar : sympy.Expr or number
            A scalar or SymPy expression to multiply the vector field by.
        
        Returns
        -------
        VFClass
            A new `VFClass` instance where each coefficient is scaled by the scalar.
        
        Examples
        --------
        >>> variableProcedure(['a', 'b'])
        >>> vf = VFClass([a, b], [a**2, b**2], 'standard')
        >>> vf_scaled = 2 * vf  # Multiplies the vector field by 2
        >>> display(vf_scaled)  # Displays the scaled vector field
        """
        return scaleVF(scalar, self)

    def __rmul__(self, scalar):
        """
        Allows scalar multiplication on the left-hand side, which can be a SymPy expression or a number.
        
        This method is invoked when the scalar is on the left-hand side of the multiplication operation. The multiplication 
        is performed by multiplying each coefficient in the vector field by the scalar.

        Parameters
        ----------
        scalar : sympy.Expr or number
            A scalar or SymPy expression to multiply the vector field by.
        
        Returns
        -------
        VFClass
            A new `VFClass` instance where each coefficient is scaled by the scalar.

        Examples
        --------
        >>> variableProcedure(['a', 'b'])
        >>> vf = VFClass([a, b], [a**2, b**2], 'standard')
        >>> vf_scaled = vf * 2  # Multiplies the vector field by 2 (right-hand side)
        >>> display(vf_scaled)  # Displays the scaled vector field
        """
        return scaleVF(scalar, self)

    def __call__(self, *args, ignore_complex_handling=None):
        """
        Applies the vector field to a scalar function or degree zero differential form, computing the directional derivative.
        
        If applied to a scalar function (SymPy expression), this computes the directional derivative of the function along 
        the vector field. If the vector field is complex, the computation may use either real or complex variables depending 
        on the `DGCVType` and the `ignore_complex_handling` flag.

        Parameters
        ----------
        args : sympy.Expr or degree zero DFClass
            A scalar function (SymPy expression) or a differential form to apply the vector field to.

        ignore_complex_handling : bool, optional
            If True, bypasses any special handling of complex variables and treats the vector field as a standard field. 
            Defaults to False.
        
        Returns
        -------
        sympy.Expr
            The result of applying the vector field to the given scalar function or degree zero differential form.
        
        Raises
        ------
        ValueError
            If the number of arguments passed is not 1.

        Examples
        --------
        >>> from sympy import exp
        >>> from DGCV import complexVarProc, VFClass
        >>> complexVarProc('z', 'x', 'y')
        >>> vf = VFClass([z], [z**2], 'complex')
        >>> f = z * exp(z)
        >>> vf(f)  # Applies the vector field to the function f(z) returning (z**3+z**2)*exp(z)
        """
        if len(args) == 1:
            arg = args[0]
            if self._varSpace_type=='complex': # this implies self.DGCVType=='complex'
                if isinstance(arg, DFClass) and arg.degree == 0:
                    arg = (arg.coeffsInKFormBasis[0],)
                return sum(self.coeffs[j] * diff(allToSym(arg), self.varSpace[j]) for j in range(len(self.coeffs)))
            else:
                if isinstance(arg, DFClass) and arg.degree == 0:
                    arg = (arg.coeffsInKFormBasis[0],)
                if self.DGCVType == 'standard' or ignore_complex_handling:
                    return sum(self.coeffs[j] * diff(arg, self.varSpace[j]) for j in range(len(self.coeffs)))
                elif self.DGCVType == 'complex':
                    return sum(self.coeffs[j] * diff(allToReal(arg), self.varSpace[j]) for j in range(len(self.coeffs)))

        else:
            raise ValueError('A vector field received a number of arguments different from 1.')
        
# Symmetric tensor field class
class STFClass(Basic):
    
    def __new__(cls, varSpace, data_dict, degree, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        # Call Basic.__new__ with only the positional arguments
        obj = Basic.__new__(cls, varSpace, data_dict, degree)
        return obj

    def __init__(self, varSpace, data_dict, degree, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        if len(varSpace)!=len(set(varSpace)):
            raise TypeError('`STFClass` expects the `varSpace` tuple not to have repeated variables.')
        self.varSpace = varSpace
        self.degree = degree
        self.simplifyKW = simplifyKW
        self.DGCVType = DGCVType
        variable_registry=get_variable_registry()
        if self.DGCVType == 'complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in self.varSpace):
                self._varSpace_type= 'real'
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in self.varSpace):
                self._varSpace_type = 'complex'
            else:
                raise KeyError('To initialize a STFClass instance with STFClass=\'complex\', `varSpace` must contain only variables from DGCV\'s complex variables systems, and all variables in `varSpace` must be simulataneously among the real and imaginary types, or simulateneously among the holomorphic and antiholomorphic types. Use `complexVarProc` to easily create DGCV complex variable systems.')
        else:
            self._varSpace_type = 'standard'

        coeffs=dict()
        for a, b in data_dict.items():
            key=tuple(sorted(a))
            if key in coeffs:
                if simplify(coeffs[key])!=simplify(b):
                    raise TypeError('The DGCVType initializer was given initialization data that does not describe a symmetric tensor.')
            else:
                coeffs[key]=b
        if coeffs == dict():
            coeffs = {(0,)*self.degree: 0}
        self.STFClassDataDict = coeffs
        self.STFClassDataMinimal = [[a,b] for a,b in coeffs.items()]
        self._STFClassDataDictFull = None
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._coeff_dicts = None
        self._coeffArray = None
        if self.degree == 0:
            self.coeffsInKFormBasis = [self.STFClassDataDict[a] for a in self.STFClassDataDict]
            self.kFormBasisGenerators = [[1]]
        else:
            oneFormsLabelsLoc = ['d_' + str(j) for j in self.varSpace]
            self.coeffsInKFormBasis = [j[1] for j in self.STFClassDataMinimal]
            self.kFormBasisGenerators = [[oneFormsLabelsLoc[k] for k in j[0]] for j in self.STFClassDataMinimal]

    @property
    def realVarSpace(self):
        if self.DGCVType=='standard':
            return self._realVarSpace
        if self._realVarSpace == None or self._imVarSpace == None:
            self.coeff_dicts
            return self._realVarSpace+self._imVarSpace
        return self._realVarSpace+self._imVarSpace
    
    @property
    def holVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace
        if self._holVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace
        return self._holVarSpace
    
    @property
    def antiholVarSpace(self):
        if self.DGCVType=='standard':
            return self._antiholVarSpace
        if self._antiholVarSpace == None:
            self.coeff_dicts
            return self._antiholVarSpace
        return self._antiholVarSpace    
    
    @property
    def compVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace == None or self._antiholVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace + self._antiholVarSpace
        return self._holVarSpace + self._antiholVarSpace

    @property
    def coeff_dicts(self): # Retrieves coeffs in different variable formats and updates *VarSpace and _coeff_dicts caches if needed
        if self.DGCVType == 'standard' or all(j!=None for j in [self._realVarSpace,self._holVarSpace,self._antiholVarSpace,self._imVarSpace,self._coeff_dicts]):
            return self._coeff_dicts
        variable_registry=get_variable_registry()
        CVS=variable_registry['complex_variable_systems']
        if self._coeff_dicts == None:
            exhaust1 = list(self.varSpace)
            populate = {'compCoeffDataDict':dict(),'realCoeffDataDict':dict(),'holVarDict':dict(),'antiholVarDict':dict(),'realVarDict':dict(),'imVarDict':dict(),'preProcessMinDataToHol':dict(),'preProcessMinDataToReal':dict()}
            if self._varSpace_type == 'real':
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust1:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][2:])-{var}).pop()
                                if cousin in exhaust1:
                                    exhaust1.remove(cousin)
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'real':
                                    realVar = var
                                    exhaust1.remove(var)
                                    imVar = cousin
                                else:
                                    realVar = cousin
                                    exhaust1.remove(var)
                                    imVar = var
                                holVar = parent['variable_relatives'][varStr]['complex_family'][0]
                                antiholVar = parent['variable_relatives'][varStr]['complex_family'][1]
                                populate['holVarDict'][holVar]=[realVar,imVar]
                                populate['antiholVarDict'][antiholVar]=[realVar,imVar]
                                populate['realVarDict'][realVar]=[holVar,antiholVar]
                                populate['imVarDict'][imVar]=[holVar,antiholVar]
            else: # self._varSpace_type == 'complex'
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust1:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][:2])-{var}).pop()
                                if cousin in exhaust1:
                                    exhaust1.remove(cousin)
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'holomorphic':
                                    holVar = var
                                    exhaust1.remove(var)
                                    antiholVar = cousin
                                else:
                                    holVar = cousin
                                    exhaust1.remove(var)
                                    antiholVar = var
                                realVar = parent['variable_relatives'][varStr]['complex_family'][2]
                                imVar = parent['variable_relatives'][varStr]['complex_family'][3]
                                populate['holVarDict'][holVar]=[realVar,imVar]
                                populate['antiholVarDict'][antiholVar]=[realVar,imVar]
                                populate['realVarDict'][realVar]=[holVar,antiholVar]
                                populate['imVarDict'][imVar]=[holVar,antiholVar]
            self._realVarSpace = tuple(populate['realVarDict'].keys())
            self._holVarSpace = tuple(populate['holVarDict'].keys())
            self._antiholVarSpace = tuple(populate['antiholVarDict'].keys())
            self._imVarSpace = tuple(populate['imVarDict'].keys())
            
            if self.degree==0:
                if self._varSpace_type=='real':
                    populate['realCoeffDataDict'] = [self.varSpace,self.STFClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,{(0,)*self.degree:self.STFClassDataDict[(0,)*self.degree]}]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.STFClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,{(0,)*self.degree:self.STFClassDataDict[(0,)*self.degree]}]
            else:
                def _retrieve_indices(term,typeSet=None):
                    if typeSet == 'symb':
                        dictLoc= populate['realVarDict'] | populate['imVarDict']
                        refTuple = self._holVarSpace + self._antiholVarSpace
                        termList = dictLoc[term]
                    elif typeSet == 'real':
                        dictLoc= populate['holVarDict'] | populate['antiholVarDict']
                        refTuple = self._realVarSpace + self._imVarSpace
                        termList = dictLoc[term]
                    index_a = refTuple.index(termList[0])
                    index_b = refTuple.index(termList[1], index_a + 1)
                    return [index_a, index_b]

                if self._varSpace_type == 'real':
                    populate['preProcessMinDataToHol'] = {j:_retrieve_indices(self.varSpace[j],'symb') for j in range(len(self.varSpace))}
                else: # if self._varSpace_type == 'complex'
                    populate['preProcessMinDataToReal'] = {j:_retrieve_indices(self.varSpace[j],'real') for j in range(len(self.varSpace))}

                def decorateWithWeights(index,target='symb'):
                    if target=='symb':
                        if 2*index<len(self.varSpace):
                            holScale=Rational(1,2) # d_z coeff of d_x
                            antiholScale=Rational(1,2) # d_BARz coeff of d_x
                        else:
                            holScale=-I/2 # d_z coeff of d_y
                            antiholScale=I/2 # d_BARz coeff of d_y
                        return [[populate['preProcessMinDataToHol'][index][0],holScale],[populate['preProcessMinDataToHol'][index][1],antiholScale]]
                    else:# converting from hol to real
                        if 2*index<len(self.varSpace):
                            realScale=1 # d_x coeff in d_z
                            imScale=I # d_y coeff in d_z
                        else:
                            realScale=1 # d_x coeff of d_BARz
                            imScale=-I # d_y coeff of d_BARz
                        return [[populate['preProcessMinDataToReal'][index][0],realScale],[populate['preProcessMinDataToReal'][index][1],imScale]]

                def weightedOrdering(arg):
                    permS,orderedList=permSign(arg[0],returnSorted=True)
                    orderedTuple=tuple(orderedList)
                    return [orderedTuple,permS*arg[1]]

                otherDict=dict()
                for j in self.STFClassDataMinimal:
                    if self._varSpace_type=='real':
                        reformatTarget='symb'
                    else:
                        reformatTarget='real'
                    termIndices=[decorateWithWeights(k,target=reformatTarget) for k in j[0]]
                    prodWithWeights=carProd_with_weights_without_R(*termIndices)
                    prodWWRescaled=[[k[0],j[1]*k[1]] for k in prodWithWeights]
                    prodWithWeightedOrder=[weightedOrdering(j) for j in prodWWRescaled]
                    for term in prodWithWeightedOrder:
                        if term[0] in otherDict:
                            oldVal=otherDict[term[0]]
                            otherDict[term[0]]=(allToSym(oldVal+term[1]))
                        else:
                            otherDict[term[0]]=(allToSym(term[1]))

                if self._varSpace_type=='real':
                    populate['realCoeffDataDict'] = [self.varSpace,self.STFClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,otherDict]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.STFClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,otherDict]

            self._coeff_dicts = populate

            return populate
        else:
            return self._coeff_dicts

    @property 
    def coeffArray(self):
        if self._STFClassDataDictFull == None:
            if self._coeffArray == None:
                def entry_rule(indexTuple):
                    sortedTuple=tuple(sorted(indexTuple))
                    if sortedTuple in self.STFClassDataDict:
                        return self.STFClassDataDict[sortedTuple]
                    else:
                        return 0
                def generate_indices(shape):
                    """Recursively generates all index tuples for an arbitrary dimensional array."""
                    if len(shape) == 1:
                        return [(i,) for i in range(shape[0])]
                    else:
                        return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
                shape = (len(self.varSpace),)*self.degree
                sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
                self._coeffArray = ImmutableSparseNDimArray(sparse_data, shape)
        else:
            if self._coeffArray == None:
                self._coeffArray = ImmutableSparseNDimArray(self._STFClassDataDictFull, shape)
            # Create the ImmutableSparseNDimArray

        return self._coeffArray

    @property
    def STFClassDataDictFull(self):
        if self._STFClassDataDictFull == None:
            def entry_rule(indexTuple):
                sortedTuple=tuple(sorted(indexTuple))
                if sortedTuple in self.STFClassDataDict:
                    return self.STFClassDataDict[sortedTuple]
                else:
                    return 0
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (len(self.varSpace),)*self.degree
            self._STFClassDataDictFull = {indices: entry_rule(indices) for indices in generate_indices(shape)}
        # Create and return the ImmutableSparseNDimArray
        return self._STFClassDataDictFull


    def __repr__(self):
        return self._generate_representation()

    def __str__(self):
        return self._generate_representation()

    def _sympystr(self, printer):
        """
        Overrides SymPy's default string formatting method.
        This ensures that custom __repr__ is used when printing STFClass objects.
        """
        return self.__repr__()

    def _latex(self, printer=None):
        """
        Overrides SymPy's default LaTeX formatting method.
        This ensures that custom _repr_latex_ is used when calling sympy.latex().
        """
        return self._repr_latex_()
    
    def _generate_representation(self):
        if self.degree == 0:
            return str(self.STFClassDataMinimal[0][1])

        basisLoc = [' '.join(j) for j in self.kFormBasisGenerators]

        termsLoc = ''.join(
            [self._labelerLoc(self.coeffsInKFormBasis[j]) + basisLoc[j]
             for j in range(len(basisLoc)) if self.coeffsInKFormBasis[j] != 0]
        )
        if not termsLoc:
            termsLoc = '0*' + basisLoc[0]
        elif termsLoc[0] == '+':
            termsLoc = termsLoc[1:]

        return termsLoc

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the differential dorm for custom simplification. 

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.
        
        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.
        
        Returns
        -------
        DFClass
            A new DFClass instance with updated simplification settings.
        """
        if format_type not in {None, 'holomorphic','real', 'symbolic_conjugate'}:
            warnings.warn('simplify_format() recieved an unsupported first argument. Try None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\' instead.')
        return STFClass(self.varSpace,self.STFClassDataDict,self.degree,DGCVType=self.DGCVType, simplifyKW={'simplify_rule': format_type, 'simplify_ignore_list': skipVar})

    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self.simplifyKW attribute.

        Returns
        -------
        DFClass
            A simplified DFClass object.
        """
        if self.simplifyKW['simplify_rule']==None: 
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.STFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='holomorphic':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToHol(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.STFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='real':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToReal(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.STFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='symbolic_conjugate':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToSym(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.STFClassDataDict.items()}
        else:
            warnings.warn('_eval_simplify recieved an unsupported STFClass.simplifyKW[\'simplify_rule\']. It is recommend to only set the simplifyKW[\'simplify_rule\'] attribute to None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\'.')
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.STFClassDataDict.items()}

        # Return a new instance of DFClass with simplified coeffs

        # Return a new instance of DFClass with simplified coeffs
        return STFClass(self.varSpace,simplified_coeffs,self.degree,DGCVType=self.DGCVType,simplifyKW=self.simplifyKW)

    def subs(self,subsData):
        newSTFData={a:sympify(b).subs(subsData) for a,b in self.STFClassDataDict.items()}
        return STFClass(self.varSpace, newSTFData, self.degree, DGCVType=self.DGCVType, simplifyKW=self.simplifyKW)

    def _labelerLoc(self, coeff):
        if str(coeff)[0] == '-':
            return f'{coeff}*' if '+' not in str(coeff)[1:] and '-' not in str(coeff)[1:] else f'+({coeff})*'
        elif coeff == 1:
            return '+'
        elif coeff == -1:
            return '-'
        elif '+' in str(coeff) or '-' in str(coeff):
            return f'+({coeff})*'
        else:
            return f'+{coeff}*'

    def _convert_to_greek(self,var_name):
        for name, greek in greek_letters.items():
            if var_name.startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def _process_var_label(self,var):
        var_str = str(var)
        if var_str.startswith("BAR"):
            # Remove "BAR" prefix
            var_str = var_str[3:]
            
            # Regular expression to match label part and trailing number part
            match = re.match(r"([a-zA-Z_]+)(\d*)$", var_str)  # Match label followed by optional digits
            
            if match:
                label_part = match.group(1)  # The label part
                number_part = match.group(2)  # The number part (if any)

                # Remove trailing underscores in label part
                label_part = label_part.rstrip("_")

                # Convert label part to Greek if applicable
                label_part = self._convert_to_greek(label_part)

                # Return LaTeX formatted string
                return f"\\overline{{{label_part}_{{{number_part}}}}}" if number_part else f"\\overline{{{label_part}}}"
        else:
            # Return LaTeX for non-BAR variables
            return latex(var)

    def _repr_latex_(self):
        if self.degree == 0:
            return f"${self._process_var_label(self.coeffsInKFormBasis[0])}$"
        
        terms = []
        varLabels = [[self.varSpace[k] for k in j[0]] for j in self.STFClassDataMinimal]
        for coeff, var_list in zip(self.coeffsInKFormBasis, varLabels):
            if coeff == 0:
                continue
            latex_vars = [f"d {self._process_var_label(var)}" for var in var_list]
            basis_form = " \\odot ".join(latex_vars)
            latex_coeff = self._format_coeff(coeff)
            term = f"{latex_coeff }{basis_form}"
            terms.append(term)
        
        if not terms:
            latex_var_0 = self._process_var_label(varLabels[0][0])
            return f"$0 d{latex_var_0}$"
        
        latex_str = terms[0]
        for term in terms[1:]:
            latex_str += f" {term}" if term.startswith('-') else f" + {term}"

        return f"${latex_str}$"

    def _format_coeff(self, coeff):
        if coeff == 1:
            return ""
        elif coeff == -1:
            return "-"
        elif sympify(coeff).is_Atom or len(coeff.as_ordered_terms()) == 1:
            return latex(coeff)
        else:
            return f"\\left({latex(coeff)}\\right)"

    def __add__(self, other):
        """
        Adds two symmetric tensor fields (i.e., STFClass instances). If the variable spaces of the fields 
        are not the same, the resulting field will have a variable space that is the union 
        of both input spaces.
        
        Parameters
        ----------
        other : STFClass
            Another symmetric tensor field to add. The variable spaces of the two forms need not 
            be identical. The result will use the union of the variable spaces of the two 
            forms.
        
        Returns
        -------
        STFClass
            A new symmetric tensor field representing the sum of the two input fields.
        
        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `STFClass`.
        """
        if isinstance(other, STFClass):
            return addSTF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with DFClass")

    def __sub__(self, other):
        """
        Subtracts one symmetric tensor field from another (i.e., STFClass instances). The subtraction 
        is performed by negating the second form and then adding it to the first form. If the 
        variable spaces of the forms are not the same, the resulting form will have a variable 
        space that is the sum of both input spaces.
        
        Parameters
        ----------
        other : STFClass
            Another symmetric tensor field to subtract. The variable spaces of the two fields need not 
            be identical. The result will use the sum of the variable spaces of the two forms.
        
        Returns
        -------
        STFClass
            A new symmetric tensor field representing the difference between the two input fields.
        
        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `STFClass`.
        """
        if isinstance(other, STFClass):
            return addSTF(self, scaleTF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with STFClass")

    def __mul__(self, arg1):
        """
        Scalar multiplication for STFClass objects. If the argument is a scalar (SymPy expression or numeric),
        the form is scaled by it.
        
        Parameters
        ----------
        arg1 : sympy expression or numeric

        Returns
        -------
        STFClass
            The result of rescaling the tensor field.

        Raises
        ------
        TypeError
            If the argument is not a scalar.
        """
        if isinstance(arg1, (TFClass,STFClass)):
            return tensorProduct(self,arg1)
        elif isinstance(arg1, (Basic, int, float, Rational)):  # Handle SymPy expressions and numeric types
            return scaleTF(arg1, self)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'STFClass' and '{type(arg1).__name__}'")

    def __rmul__(self, arg1):
        """
        """
        if isinstance(arg1, (Basic, int, float, Rational)):  # Handle SymPy expressions and numeric types
            return scaleTF(arg1, self)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'STFClass' and '{type(arg1).__name__}'")
    

    def __call__(self, *VFArgs):
        if self.degree == len(VFArgs):
            if self.degree == 0:
                return self.coeffsInKFormBasis[0]
            else:
                if all(isinstance(var,VFClass) for var in VFArgs):
                    if self.DGCVType == 'complex':
                        if self._varSpace_type == 'complex':
                            VFArgs=[allToSym(vf) if vf._varSpace_type!=self._varSpace_type else vf for vf in VFArgs]
                        else:
                            VFArgs=[allToReal(vf) if vf._varSpace_type!=self._varSpace_type else vf for vf in VFArgs]
                    if self.DGCVType == 'standard':
                        VFArgs=[_remove_complex_handling(vf) for vf in VFArgs]
                    inputCoeffs = contravariantVFTensorCoeffs(self.varSpace, *VFArgs)
                    # Access the elements of coeffArray directly
                    result = 0
                    for index, scalar in inputCoeffs:
                        _,orderedList=permSign(index,returnSorted=True)
                        index=tuple(orderedList)
                        if index in self.STFClassDataDict:
                            result += self.STFClassDataDict[index] * scalar
                    return result
                else:
                    raise ValueError('`STFClass` can only operator on `VFClass` objects.')
        else:
            raise ValueError('A symmetric tensor field received a number of arguments different from its degree.')

# Tensor field class
class TFClass(Basic):
    
    def __new__(cls, varSpace, data_dict, degree, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        # Call Basic.__new__ with only the positional arguments
        obj = Basic.__new__(cls, varSpace, data_dict, degree)
        return obj

    def __init__(self, varSpace, data_dict, degree, DGCVType='standard', simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': None}):
        if len(varSpace)!=len(set(varSpace)):
            raise TypeError('`TFClass` expects the `varSpace` tuple not to have repeated variables.')
        self.varSpace = varSpace
        self.degree = degree
        self.simplifyKW = simplifyKW
        self.DGCVType = DGCVType
        variable_registry=get_variable_registry()
        if self.DGCVType == 'complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in self.varSpace):
                self._varSpace_type= 'real'
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in self.varSpace):
                self._varSpace_type = 'complex'
            else:
                raise KeyError('To initialize a TFClass instance with TFClass=\'complex\', `varSpace` must contain only variables from DGCV\'s complex variables systems, and all variables in `varSpace` must be simulataneously among the real and imaginary types, or simulateneously among the holomorphic and antiholomorphic types. Use `complexVarProc` to easily create DGCV complex variable systems.')
        else:
            self._varSpace_type = 'standard'

        self.TFClassDataDict = data_dict
        self.TFClassDataMinimal = [[a,b] for a,b in data_dict.items() if b!=0]
        if self.TFClassDataMinimal == []:
            self.TFClassDataMinimal=[[(0,)*self.degree,0]]
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._coeffArray = None
        self._coeff_dicts = None
        if self.degree == 0:
            self.coeffsInKFormBasis = [self.TFClassDataDict[a] for a in self.TFClassDataDict]
            self.kFormBasisGenerators = [[1]]
        else:
            oneFormsLabelsLoc = ['d_' + str(j) for j in self.varSpace]
            self.coeffsInKFormBasis = [j[1] for j in self.TFClassDataMinimal]
            self.kFormBasisGenerators = [[oneFormsLabelsLoc[k] for k in j[0]] for j in self.TFClassDataMinimal]
            if self.simplifyKW['preferred_basis_element'] == None:
                if self.degree>0:
                    self._prefered_basis_element = self.kFormBasisGenerators[0]
                else:
                    self._prefered_basis_element = None
            else:
                if isinstance(self.simplifyKW['preferred_basis_element'],(list,tuple)):
                    if all(k in range(self.degree) for k in self.simplifyKW['preferred_basis_element']):
                        self._prefered_basis_element = [oneFormsLabelsLoc[k] for k in self.simplifyKW['preferred_basis_element']]
                    else:
                        raise TypeError('The \'preferred_basis_element\' keyword in the simplifyKW dict should be None or a tuple of integers in the range of the tensor field\'s degree.')


    @property
    def realVarSpace(self):
        if self.DGCVType=='standard':
            return self._realVarSpace
        if self._realVarSpace == None or self._imVarSpace == None:
            self.coeff_dicts
            return self._realVarSpace+self._imVarSpace
        return self._realVarSpace+self._imVarSpace
    
    @property
    def holVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace
        if self._holVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace
        return self._holVarSpace
    
    @property
    def antiholVarSpace(self):
        if self.DGCVType=='standard':
            return self._antiholVarSpace
        if self._antiholVarSpace == None:
            self.coeff_dicts
            return self._antiholVarSpace
        return self._antiholVarSpace    
    
    @property
    def compVarSpace(self):
        if self.DGCVType=='standard':
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace == None or self._antiholVarSpace == None:
            self.coeff_dicts
            return self._holVarSpace + self._antiholVarSpace
        return self._holVarSpace + self._antiholVarSpace

    @property
    def coeff_dicts(self): # Retrieves coeffs in different variable formats and updates *VarSpace and _coeff_dicts caches if needed
        if self.DGCVType == 'standard' or all(j!=None for j in [self._realVarSpace,self._holVarSpace,self._antiholVarSpace,self._imVarSpace,self._coeff_dicts]):
            return self._coeff_dicts
        variable_registry=get_variable_registry()
        CVS=variable_registry['complex_variable_systems']
        if self._coeff_dicts == None:
            exhaust1 = list(self.varSpace)
            populate = {'compCoeffDataDict':dict(),'realCoeffDataDict':dict(),'holVarDict':dict(),'antiholVarDict':dict(),'realVarDict':dict(),'imVarDict':dict(),'preProcessMinDataToHol':dict(),'preProcessMinDataToReal':dict()}
            if self._varSpace_type == 'real':
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust1:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][2:])-{var}).pop()
                                if cousin in exhaust1:
                                    exhaust1.remove(cousin)
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'real':
                                    realVar = var
                                    exhaust1.remove(var)
                                    imVar = cousin
                                else:
                                    realVar = cousin
                                    exhaust1.remove(var)
                                    imVar = var
                                holVar = parent['variable_relatives'][varStr]['complex_family'][0]
                                antiholVar = parent['variable_relatives'][varStr]['complex_family'][1]
                                populate['holVarDict'][holVar]=[realVar,imVar]
                                populate['antiholVarDict'][antiholVar]=[realVar,imVar]
                                populate['realVarDict'][realVar]=[holVar,antiholVar]
                                populate['imVarDict'][imVar]=[holVar,antiholVar]
            else: # self._varSpace_type == 'complex'
                for var in self.varSpace:
                    varStr = str(var)
                    if var in exhaust1:
                        for parent in CVS.values():
                            if varStr in parent['variable_relatives']:
                                cousin=(set(parent['variable_relatives'][varStr]['complex_family'][:2])-{var}).pop()
                                if cousin in exhaust1:
                                    exhaust1.remove(cousin)
                                if parent['variable_relatives'][varStr]['complex_positioning'] == 'holomorphic':
                                    holVar = var
                                    exhaust1.remove(var)
                                    antiholVar = cousin
                                else:
                                    holVar = cousin
                                    exhaust1.remove(var)
                                    antiholVar = var
                                realVar = parent['variable_relatives'][varStr]['complex_family'][2]
                                imVar = parent['variable_relatives'][varStr]['complex_family'][3]
                                populate['holVarDict'][holVar]=[realVar,imVar]
                                populate['antiholVarDict'][antiholVar]=[realVar,imVar]
                                populate['realVarDict'][realVar]=[holVar,antiholVar]
                                populate['imVarDict'][imVar]=[holVar,antiholVar]
            self._realVarSpace = tuple(populate['realVarDict'].keys())
            self._holVarSpace = tuple(populate['holVarDict'].keys())
            self._antiholVarSpace = tuple(populate['antiholVarDict'].keys())
            self._imVarSpace = tuple(populate['imVarDict'].keys())
            
            if self.degree==0:
                if self._varSpace_type=='real':
                    populate['realCoeffDataDict'] = [self.varSpace,self.TFClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,{(0,)*self.degree:self.TFClassDataDict[(0,)*self.degree]}]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.TFClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,{(0,)*self.degree:self.TFClassDataDict[(0,)*self.degree]}]
            else:
                def _retrieve_indices(term,typeSet=None):
                    if typeSet == 'symb':
                        dictLoc= populate['realVarDict'] | populate['imVarDict']
                        refTuple = self._holVarSpace + self._antiholVarSpace
                        termList = dictLoc[term]
                    elif typeSet == 'real':
                        dictLoc= populate['holVarDict'] | populate['antiholVarDict']
                        refTuple = self._realVarSpace + self._imVarSpace
                        termList = dictLoc[term]
                    index_a = refTuple.index(termList[0])
                    index_b = refTuple.index(termList[1], index_a + 1)
                    return [index_a, index_b]

                if self._varSpace_type == 'real':
                    populate['preProcessMinDataToHol'] = {j:_retrieve_indices(self.varSpace[j],'symb') for j in range(len(self.varSpace))}
                else: # if self._varSpace_type == 'complex'
                    populate['preProcessMinDataToReal'] = {j:_retrieve_indices(self.varSpace[j],'real') for j in range(len(self.varSpace))}

                def decorateWithWeights(index,target='symb'):
                    if target=='symb':
                        if 2*index<len(self.varSpace):
                            holScale=Rational(1,2) # d_z coeff of d_x
                            antiholScale=Rational(1,2) # d_BARz coeff of d_x
                        else:
                            holScale=-I/2 # d_z coeff of d_y
                            antiholScale=I/2 # d_BARz coeff of d_y
                        return [[populate['preProcessMinDataToHol'][index][0],holScale],[populate['preProcessMinDataToHol'][index][1],antiholScale]]
                    else:# converting from hol to real
                        if 2*index<len(self.varSpace):
                            realScale=1 # d_x coeff in d_z
                            imScale=I # d_y coeff in d_z
                        else:
                            realScale=1 # d_x coeff of d_BARz
                            imScale=-I # d_y coeff of d_BARz
                        return [[populate['preProcessMinDataToReal'][index][0],realScale],[populate['preProcessMinDataToReal'][index][1],imScale]]

                def weightedOrdering(arg):
                    permS,orderedList=permSign(arg[0],returnSorted=True)
                    orderedTuple=tuple(orderedList)
                    return [orderedTuple,permS*arg[1]]

                otherDict=dict()
                for j in self.TFClassDataMinimal:
                    if self._varSpace_type=='real':
                        reformatTarget='symb'
                    else:
                        reformatTarget='real'
                    termIndices=[decorateWithWeights(k,target=reformatTarget) for k in j[0]]
                    prodWithWeights=carProd_with_weights_without_R(*termIndices)
                    prodWWRescaled=[[k[0],j[1]*k[1]] for k in prodWithWeights]
                    prodWithWeightedOrder=[weightedOrdering(j) for j in prodWWRescaled]
                    for term in prodWithWeightedOrder:
                        if term[0] in otherDict:
                            oldVal=otherDict[term[0]]
                            otherDict[term[0]]=(allToSym(oldVal+term[1]))
                        else:
                            otherDict[term[0]]=(allToSym(term[1]))

                if self._varSpace_type=='real':
                    populate['realCoeffDataDict'] = [self.varSpace,self.TFClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,otherDict]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.TFClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,otherDict]

            self._coeff_dicts = populate

            return populate
        else:
            return self._coeff_dicts

    @property 
    def coeffArray(self):
        if self._coeffArray == None:
            shape = (len(self.varSpace),)*self.degree
            data = {a:b for a,b in self.TFClassDataDict.items()} #Double dict so that ImmutableSparseNDimArray doesn't destroy the original!!!
            self._coeffArray = ImmutableSparseNDimArray(data, shape) 
        return self._coeffArray


    def __repr__(self):
        return self._generate_representation()

    def __str__(self):
        return self._generate_representation()

    def _sympystr(self, printer):
        """
        Overrides SymPy's default string formatting method.
        This ensures that custom __repr__ is used when printing TFClass objects.
        """
        return self.__repr__()

    def _latex(self, printer=None):
        """
        Overrides SymPy's default LaTeX formatting method.
        This ensures that custom _repr_latex_ is used when calling sympy.latex().
        """
        return self._repr_latex_()
    
    def _generate_representation(self):
        if self.degree == 0:
            return str(self.TFClassDataMinimal[0][1])

        basisLoc = [' '.join(j) for j in self.kFormBasisGenerators]

        termsLoc = ''.join(
            [self._labelerLoc(self.coeffsInKFormBasis[j]) + basisLoc[j]
             for j in range(len(basisLoc)) if self.coeffsInKFormBasis[j] != 0]
        )
        if not termsLoc:
            termsLoc = '0*' + ' '.join(self._prefered_basis_element)
        elif termsLoc[0] == '+':
            termsLoc = termsLoc[1:]

        return termsLoc

    def simplify_format(self, format_type=None, skipVar=None):
        """
        Prepares the differential dorm for custom simplification. 

        Parameters
        ----------
        arg : str
            The simplification rule to apply. Options include 'real', 'holomorphic', and 'symbolic_conjugate'.
        
        skipVar : list, optional
            A list of strings that are parent labels for DGCV variable systems to exclude from the simplification process.
        
        Returns
        -------
        TFClass
            A new TFClass instance with updated simplification settings.
        """
        if format_type not in {None, 'holomorphic','real', 'symbolic_conjugate'}:
            warnings.warn('simplify_format() recieved an unsupported first argument. Try None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\' instead.')
        return TFClass(self.varSpace,self.TFClassDataDict,self.degree,DGCVType=self.DGCVType, simplifyKW={'simplify_rule': format_type, 'simplify_ignore_list': skipVar})

    def _eval_simplify(self, **kwargs):
        """
        Applies the simplification based on the current simplification settings in the self.simplifyKW attribute.

        Returns
        -------
        TFClass
            A simplified TFClass object.
        """
        if self.simplifyKW['simplify_rule']==None: 
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.TFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='holomorphic':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToHol(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.TFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='real':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToReal(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.TFClassDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='symbolic_conjugate':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToSym(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.TFClassDataDict.items()}
        else:
            warnings.warn('_eval_simplify recieved an unsupported TFClass.simplifyKW[\'simplify_rule\']. It is recommend to only set the simplifyKW[\'simplify_rule\'] attribute to None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\'.')
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.TFClassDataDict.items()}

        # Return a new instance of TFClass with simplified coeffs

        # Return a new instance of TFClass with simplified coeffs
        return TFClass(self.varSpace,simplified_coeffs,self.degree,DGCVType=self.DGCVType,simplifyKW=self.simplifyKW)

    def subs(self,subsData):
        newTFData={a:sympify(b).subs(subsData) for a,b in self.TFClassDataDict.items()}
        return TFClass(self.varSpace, newTFData, self.degree, DGCVType=self.DGCVType, simplifyKW=self.simplifyKW)

    def _labelerLoc(self, coeff):
        if str(coeff)[0] == '-':
            return f'{coeff}*' if '+' not in str(coeff)[1:] and '-' not in str(coeff)[1:] else f'+({coeff})*'
        elif coeff == 1:
            return '+'
        elif coeff == -1:
            return '-'
        elif '+' in str(coeff) or '-' in str(coeff):
            return f'+({coeff})*'
        else:
            return f'+{coeff}*'

    def _convert_to_greek(self,var_name):
        for name, greek in greek_letters.items():
            if var_name.startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def _process_var_label(self,var):
        var_str = str(var)
        if var_str.startswith("BAR"):
            # Remove "BAR" prefix
            var_str = var_str[3:]
            
            # Regular expression to match label part and trailing number part
            match = re.match(r"([a-zA-Z_]+)(\d*)$", var_str)  # Match label followed by optional digits
            
            if match:
                label_part = match.group(1)  # The label part
                number_part = match.group(2)  # The number part (if any)

                # Remove trailing underscores in label part
                label_part = label_part.rstrip("_")

                # Convert label part to Greek if applicable
                label_part = self._convert_to_greek(label_part)

                # Return LaTeX formatted string
                return f"\\overline{{{label_part}_{{{number_part}}}}}" if number_part else f"\\overline{{{label_part}}}"
        else:
            # Return LaTeX for non-BAR variables
            return latex(var)

    def _repr_latex_(self):
        if self.degree == 0:
            return f"${self._process_var_label(self.coeffsInKFormBasis[0])}$"
        
        terms = []
        varLabels = [[self.varSpace[k] for k in j[0]] for j in self.TFClassDataMinimal]
        for coeff, var_list in zip(self.coeffsInKFormBasis, varLabels):
            if coeff == 0:
                continue
            latex_vars = [f"d {self._process_var_label(var)}" for var in var_list]
            basis_form = " \\otimes ".join(latex_vars)
            latex_coeff = self._format_coeff(coeff)
            term = f"{latex_coeff }{basis_form}"
            terms.append(term)

        if not terms:
            latex_vars = [f"d {self._process_var_label(var)}" for var in [self.varSpace[k] for k in self.simplifyKW['preferred_basis_element']]]
            differential_form = " \\otimes ".join(latex_vars)
            term = f"{0}{differential_form}"
            terms.append(term)
        
        latex_str = terms[0]
        for term in terms[1:]:
            latex_str += f" {term}" if term.startswith('-') else f" + {term}"

        return f"${latex_str}$"

    def _format_coeff(self, coeff):
        if coeff == 1:
            return ""
        elif coeff == -1:
            return "-"
        elif sympify(coeff).is_Atom or len(coeff.as_ordered_terms()) == 1:
            return latex(coeff)
        else:
            return f"\\left({latex(coeff)}\\right)"

    def __add__(self, other):
        """
        Adds two symmetric tensor fields (i.e., TFClass instances). If the variable spaces of the fields 
        are not the same, the resulting field will have a variable space that is the union 
        of both input spaces.
        
        Parameters
        ----------
        other : TFClass
            Another symmetric tensor field to add. The variable spaces of the two forms need not 
            be identical. The result will use the union of the variable spaces of the two 
            forms.
        
        Returns
        -------
        TFClass
            A new symmetric tensor field representing the sum of the two input fields.
        
        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `TFClass`.
        """
        if isinstance(other, TFClass):
            return addTF(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for + with DFClass")

    def __sub__(self, other):
        """
        Subtracts one symmetric tensor field from another (i.e., TFClass instances). The subtraction 
        is performed by negating the second form and then adding it to the first form. If the 
        variable spaces of the forms are not the same, the resulting form will have a variable 
        space that is the sum of both input spaces.
        
        Parameters
        ----------
        other : TFClass
            Another symmetric tensor field to subtract. The variable spaces of the two fields need not 
            be identical. The result will use the sum of the variable spaces of the two forms.
        
        Returns
        -------
        TFClass
            A new symmetric tensor field representing the difference between the two input fields.
        
        Raises
        ------
        TypeError
            If the argument `other` is not an instance of `TFClass`.
        """
        if isinstance(other, TFClass):
            return addTF(self, scaleTF(-1, other))
        else:
            raise TypeError("Unsupported operand type(s) for - with TFClass")

    def __mul__(self, arg1):
        """
        Scalar multiplication for TFClass objects. If the argument is a scalar (SymPy expression or numeric),
        the form is scaled by it.
        
        Parameters
        ----------
        arg1 : sympy expression or numeric

        Returns
        -------
        TFClass
            The result of rescaling the tensor field.

        Raises
        ------
        TypeError
            If the argument is not a scalar.
        """
        if isinstance(arg1, (TFClass,STFClass)):
            return tensorProduct(self,arg1)
        elif isinstance(arg1, (Basic, int, float, Rational)):  # Handle SymPy expressions and numeric types
            return scaleTF(arg1, self)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'TFClass' and '{type(arg1).__name__}'")

    def __rmul__(self, arg1):
        """
        """
        if isinstance(arg1, (Basic, int, float, Rational)):  # Handle SymPy expressions and numeric types
            return scaleTF(arg1, self)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'TFClass' and '{type(arg1).__name__}'")
    


    def __call__(self, *VFArgs):
        if self.degree == len(VFArgs):
            if self.degree == 0:
                return self.coeffsInKFormBasis[0]
            else:
                if all(isinstance(var,VFClass) for var in VFArgs):
                    if self.DGCVType == 'complex':
                        if self._varSpace_type == 'complex':
                            VFArgs=[allToSym(vf) if vf._varSpace_type!=self._varSpace_type else vf for vf in VFArgs]
                        else:
                            VFArgs=[allToReal(vf) if vf._varSpace_type!=self._varSpace_type else vf for vf in VFArgs]
                    if self.DGCVType == 'standard':
                        VFArgs=[_remove_complex_handling(vf) for vf in VFArgs]
                    inputCoeffs = contravariantVFTensorCoeffs(self.varSpace, *VFArgs)
                    # Access the elements of coeffArray directly
                    result = 0
                    for index, scalar in inputCoeffs:
                            if index in self.TFClassDataDict:
                                result += self.TFClassDataDict[index] * scalar
                    return result
                else:
                    raise ValueError('`TFClass` can only operator on `VFClass` objects.')
        else:
            raise ValueError('A tensor field received a number of arguments different from its degree.')

# DGCV polynomial class
class DGCVPolyClass(Basic):
    """
    A polynomial class for handling both standard and complex polynomials, integrating SymPy's polynomial tools
    with DGCV's complex variable handling. The `DGCVPolyClass` wraps SymPy's core polynomial functionality, namely the Poly class objects, in a way that interacts well with the DGCV framework for managing complex variable systems.

    DGCVPolyClass augments the base SymPy `Poly` in a few ways:
    
    - **Support for multiple variable formats**: The class allows working with polynomials in unformatted, 
      real, or complex (holomorphic/antiholomorphic with symbolic conjuagtes) forms through different lazy attributes. 
      The unformatted form maintains whatever format the expression given to initialize the DGCVPolyClass was in.
    - **Holomorphic, antiholomorphic, pluriharmonic, and mixed term extraction**: This class has built-in methods to
      decompose polynomials based on these parts, leveraging DGCV's `allToSym` and `allToReal` functions.


    Parameters
    ----------
    polyExpr : sympy.Expr
        The polynomial expression to be handled.
    varSpace : tuple, optional
        The tuple of variables used in the polynomial. If not provided, it defaults to the free symbols in the given expression.
    degreeUpperBound : int, optional
      automatically inferred via sympy Poly class methods if not provided. `degreeUpperBound` is an attribute that a polynomial can be marked with for applications where it is preferable to evaluate only low degree terms (i.e., below the upper bound). It defaults to the polynomial's degree. 

    Attributes
    ----------
    polyExpr : sympy.Expr
        The polynomial expression wrapped by this class.
    varSpace : tuple
        The set of variables involved in the polynomial.
    degreeUpperBound : int or None
        The degree upper bound of the polynomial, inferred or provided during initialization.
    poly_obj_unformatted : sympy.Poly
        The unformatted SymPy `Poly` object constructed from the provided expression and variable space.
    poly_obj_complex : sympy.Poly
        The `Poly` object with holomorphic and antiholomorphic variables, constructed via DGCV's complex variable transformations.
    poly_obj_real : sympy.Poly
        The `Poly` object with real variables, constructed using DGCV's real variable transformations.
    
    Methods
    -------
    get_monomials(min_degree=0, max_degree=None, format='unformatted'):
        Returns the monomials of the polynomial within the specified degree range and format.
    
    holomorphic_part():
        Lazily computes and returns the holomorphic part of the polynomial.
    
    antiholomorphic_part():
        Lazily computes and returns the antiholomorphic part of the polynomial.
    
    pluriharmonic_part():
        Lazily computes and returns the pluriharmonic part of the polynomial (sum of holomorphic and antiholomorphic parts).
    
    mixed_terms():
        Lazily computes and returns the mixed terms (terms involving both holomorphic and antiholomorphic variables).
    
    __add__(other):
        Adds two `DGCVPolyClass` instances, combining their variable spaces and degree bounds.
    
    __mul__(other):
        Multiplies two `DGCVPolyClass` instances, combining their variable spaces and degree bounds.
    
    __subs__(substitutions):
        Substitutes variables in the polynomial expression and returns a new `DGCVPolyClass` instance with the substitutions applied.

    Example
    -------
    >>> from sympy import symbols
    >>> from dgcv import DGCVPolyClass, complexVarProc
    >>> x, y, z, BARz = symbols('x y z BARz')
    >>> complexVarProc('z', 'x', 'y')  # Initialize complex variables in DGCV
    >>> poly = DGCVPolyClass(y**2 + z**2)
    >>> print(poly.holomorphic_part)  # Extract the holomorphic part
    -z**2
    >>> print(poly.get_monomials(format='complex'))  # Get monomials in complex format
    [-z**2, y**2]
    """
    def __new__(cls, polyExpr, varSpace=None, degreeUpperBound=None):
        # Create a new instance using Basic's __new__
        return Basic.__new__(cls, polyExpr)


    def __init__(self, polyExpr, varSpace=None, degreeUpperBound=None):
        """
        Initializes the DGCVPolyClass with an optional variable space.

        Parameters:
        polyExpr : sympy.Expr
            The polynomial expression.
        varSpace : tuple, optional
            The tuple of variables (default is the free symbols in polyExpr).
        degreeUpperBound : int, optional
            The upper bound on the degree of the polynomial.
        """
        polyExpr = sympify(polyExpr)
        
        # If varSpace is not provided, infer it from the free symbols in polyExpr
        if varSpace is None:
            if isinstance(polyExpr, (int, float)):  # If polyExpr is a scalar
                self.varSpace = ()
            else:
                self.varSpace = tuple(polyExpr.free_symbols)
        else:
            self.varSpace = tuple(varSpace)
        
        self.polyExpr = polyExpr
        self.degree = total_degree(polyExpr, *self.varSpace)
        self.degreeUpperBound = degreeUpperBound 
        # Initialize private attributes for lazy evaluation
        self._holomorphic_part = None
        self._antiholomorphic_part = None
        self._pluriharmonic_part = None
        self._mixed_terms_part = None
        
        self._poly_obj_unformatted = None
        self._poly_obj_complex = None
        self._poly_obj_real = None

    @property
    def poly_obj_unformatted(self):
        """Lazily constructs and returns the unformatted Poly object."""
        if self._poly_obj_unformatted is None:
            self._poly_obj_unformatted = Poly(self.polyExpr, *self.varSpace)
        return self._poly_obj_unformatted

    @property
    def poly_obj_complex(self):
        """Lazily constructs and returns the Poly object with complex variable handling."""
        if self._poly_obj_complex is None:
            # Apply allToSym to convert to holomorphic and antiholomorphic variables
            complex_expr = allToSym(self.polyExpr)
            # Get the free symbols of the transformed expression for generators
            complex_varSpace = tuple(set.union(*[allToSym(j).free_symbols for j in self.varSpace]))
            self._poly_obj_complex = Poly(complex_expr, *complex_varSpace)
        return self._poly_obj_complex

    @property
    def poly_obj_real(self):
        """Lazily constructs and returns the Poly object with real variable handling."""
        if self._poly_obj_real is None:
            # Apply allToReal to convert to real variables
            real_expr = allToReal(self.polyExpr)
            # Get the free symbols of the transformed expression for generators
            real_varSpace = tuple(set.union(*[allToReal(j).free_symbols for j in self.varSpace]))
            self._poly_obj_real = Poly(real_expr, *real_varSpace)
        return self._poly_obj_real

    def get_monomials(self, min_degree=0, max_degree=None, format='unformatted', return_coeffs=False):
        """
        Computes and returns the monomials of the polynomial within the specified degree range,
        using the specified format for the polynomial generators.

        Parameters
        ----------
        min_degree : int, optional
            The minimum degree of monomials to return (default is 0).
        max_degree : int, optional
            The maximum degree of monomials to return (default is the polynomial's total degree).
        format : str, optional
            The format of the polynomial generators. Must be one of:
            - 'unformatted': Use the original variables provided or free symbols from polyExpr.
            - 'complex': Use holomorphic and antiholomorphic variables (z, BARz).
            - 'real': Use real variable representations (x, y).
            (default is 'unformatted').
        return_coeffs : bool, optional
            If true, only a list of coefficients is returned

        Returns
        -------
        list
            A list of monomials within the specified degree range, or just their coefficients if return_coeffs=True is set.
        """
        if format not in ['unformatted', 'complex', 'real']:
            raise ValueError("Invalid format. Choose 'unformatted', 'complex', or 'real'.")

        # Select the appropriate Poly object based on the format
        if format == 'unformatted':
            poly_obj = self.poly_obj_unformatted
        elif format == 'complex':
            poly_obj = self.poly_obj_complex
        else:
            poly_obj = self.poly_obj_real

        # Default max_degree to the total degree of the polynomial
        if max_degree is None:
            max_degree = self.degree

        # Extract monomials and coefficients
        monoms = poly_obj.monoms()
        coeffs = poly_obj.coeffs()

        # Filter monomials by degree
        if return_coeffs == True:
            filtered_monomials = [
                coeff
                for monom, coeff in zip(monoms, coeffs)
                if min_degree <= sum(monom) <= max_degree
            ]
        else:
            filtered_monomials = [
                Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                for monom, coeff in zip(monoms, coeffs)
                if min_degree <= sum(monom) <= max_degree
            ]

        return filtered_monomials

    def get_coeffs(self, min_degree=0, max_degree=None, format='unformatted', return_coeffs=False):
            return self.get_monomials(self, min_degree=0, max_degree=max_degree, format=format, return_coeffs=True)

    @property
    def holomorphic_part(self):
        """Lazily computes and returns the holomorphic part of the polynomial."""
        if self._holomorphic_part is None:
            # Use the 'complex' formatted Poly object
            poly_obj = self.poly_obj_complex
            variable_registry = get_variable_registry()
            conversion_dictionaries = variable_registry['conversion_dictionaries']
            
            holomorphic_terms = []
            
            for monom, coeff in zip(poly_obj.monoms(), poly_obj.coeffs()):
                # Rebuild the monomial from gens and exponents
                term = Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                free_symbols = term.free_symbols
                
                # Check if all symbols are holomorphic
                if all(symbol in conversion_dictionaries['holToReal'] for symbol in free_symbols):
                    holomorphic_terms.append(term)
            
            self._holomorphic_part = Add(*holomorphic_terms) if holomorphic_terms else 0
        return self._holomorphic_part

    @property
    def antiholomorphic_part(self):
        """Lazily computes and returns the antiholomorphic part of the polynomial."""
        if self._antiholomorphic_part is None:
            # Use the 'complex' formatted Poly object
            poly_obj = self.poly_obj_complex
            variable_registry = get_variable_registry()
            conversion_dictionaries = variable_registry['conversion_dictionaries']
            
            antiholomorphic_terms = []
            
            for monom, coeff in zip(poly_obj.monoms(), poly_obj.coeffs()):
                # Rebuild the monomial from gens and exponents
                term = Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                free_symbols = term.free_symbols
                
                # Check if all symbols are antiholomorphic
                if all(symbol in conversion_dictionaries['symToHol'] for symbol in free_symbols):
                    antiholomorphic_terms.append(term)
            
            self._antiholomorphic_part = Add(*antiholomorphic_terms) if antiholomorphic_terms else 0
        return self._antiholomorphic_part

    @property
    def pluriharmonic_part(self):
        """Lazily computes and returns the pluriharmonic part of the polynomial."""
        if self._pluriharmonic_part is None:
            # Sum of holomorphic and antiholomorphic parts, without double-counting the constant term
            holomorphic = self.holomorphic_part
            antiholomorphic = self.antiholomorphic_part
            
            # Constant term is common between holomorphic and antiholomorphic parts
            constant_term = holomorphic.as_coefficients_dict().get(1, 0) 
            
            # Subtract the constant term to avoid double-counting
            self._pluriharmonic_part = holomorphic + antiholomorphic - constant_term
        
        return self._pluriharmonic_part

    @property
    def mixed_terms(self):
        """Lazily computes and returns the mixed terms of the polynomial."""
        if self._mixed_terms_part is None:
            # Use the 'complex' formatted Poly object
            poly_obj = self.poly_obj_complex
            variable_registry = get_variable_registry()
            conversion_dictionaries = variable_registry['conversion_dictionaries']
            
            mixed_terms = []
            
            for monom, coeff in zip(poly_obj.monoms(), poly_obj.coeffs()):
                # Rebuild the monomial from gens and exponents
                term = Mul(*[gen**exp for gen, exp in zip(poly_obj.gens, monom)]) * coeff
                free_symbols = term.free_symbols
                
                # Check if the term contains both holomorphic and antiholomorphic variables
                is_holomorphic = all(symbol in conversion_dictionaries['holToReal'] for symbol in free_symbols)
                is_antiholomorphic = all(symbol in conversion_dictionaries['symToHol'] for symbol in free_symbols)
                
                if not (is_holomorphic or is_antiholomorphic):
                    mixed_terms.append(term)
            
            self._mixed_terms_part = Add(*mixed_terms) if mixed_terms else 0
        return self._mixed_terms_part

    # Custom simplify and latex methods
    def simplify_poly(self):
        return DGCVPolyClass(simplify(self.polyExpr),self.varSpace,degreeUpperBound=self.degreeUpperBound)

    def latex_representation(self):
        return latex(self.polyExpr)

    def _repr_latex_(self):
        return (self.polyExpr)._repr_latex_()

    def _eval_simplify(self, **kwargs):
        return self.simplify_poly()

    def _latex(self, printer=None):
        return self.latex_representation()

    # Additional methods
    def evaluate(self, **values):
        return self.polyExpr.subs(values)

    def get_degree(self):
        return self.degree

    def is_homogeneous(self):
        degrees = [sum(j[1]) for j in self.multiIndexEncoding if j[0] != 0]
        return all(degree == degrees[0] for degree in degrees)

    def pretty_print(self):
        return pretty(self.polyExpr)

    def expand(self, deep=True, modulus=None, **hints):
        """
        Expands the internal polynomial expression and returns a new DGCVPolyClass instance.
        Supports 'deep' and 'modulus' keywords, passed directly to SymPy's expand function.

        Parameters
        ----------
        deep : bool, optional
            Whether to apply expansion recursively to sub-expressions. Default is True.
        modulus : int, optional
            Apply expansion with respect to this modulus (modular arithmetic). Default is None.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the expanded polynomial expression.
        """
        # Pass 'deep', 'modulus', and any additional hints to SymPy's expand method
        expanded_expr = self.polyExpr.expand(deep=deep, modulus=modulus, **hints)
        return DGCVPolyClass(expanded_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    def factor(self, **hints):
        """
        Factors the internal polynomial expression and returns a new DGCVPolyClass instance.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the factored polynomial expression.
        """
        factored_expr = self.polyExpr.factor(**hints)
        return DGCVPolyClass(factored_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    def expand_trig(self, **hints):
        """
        Expands trigonometric functions in the polynomial expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with expanded trigonometric expressions.
        """
        expanded_trig_expr = self.polyExpr.expand(trig=True, **hints)
        return DGCVPolyClass(expanded_trig_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    def cancel(self, **hints):
        """
        Cancels common factors between the numerator and the denominator of the polynomial expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the canceled rational expression.
        """
        canceled_expr = self.polyExpr.cancel(**hints)
        return DGCVPolyClass(canceled_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    def diff(self, *symbols, **hints):
        """
        Differentiates the polynomial expression with respect to the given symbols.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the differentiated expression.
        """
        differentiated_expr = self.polyExpr.diff(*symbols, **hints)
        return DGCVPolyClass(differentiated_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    def integrate(self, *symbols, **hints):
        """
        Integrates the polynomial expression with respect to the given symbols.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the integrated expression.
        """
        integrated_expr = self.polyExpr.integrate(*symbols, **hints)
        return DGCVPolyClass(integrated_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    def subs(self, substitutions, **hints):
        """
        Substitutes the given variables in the polynomial expression and returns a new DGCVPolyClass instance.

        Parameters
        ----------
        substitutions : dict or list
            A dictionary or list of substitutions, where keys are variables to replace and values are the new values.
        hints : dict, optional
            Additional hints to pass to the substitution process.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the substituted polynomial expression.
        """
        # Perform the substitution on the internal polyExpr
        substituted_expr = self.polyExpr.subs(substitutions, **hints)
        
        # Return a new DGCVPolyClass with the substituted expression, preserving varSpace and degreeUpperBound
        return DGCVPolyClass(substituted_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)
    
    def _apply_sympy_function(self, sympy_func, *args, **kwargs):
        """
        Applies the given SymPy function to the internal polyExpr and returns a new DGCVPolyClass instance.

        Parameters
        ----------
        sympy_func : callable
            The SymPy function to apply (e.g., factor, collect, cancel).
        args : tuple
            Positional arguments to pass to the SymPy function.
        kwargs : dict
            Keyword arguments to pass to the SymPy function.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the transformed polynomial expression.
        """
        transformed_expr = sympy_func(self.polyExpr, *args, **kwargs)  # Correct function call
        return DGCVPolyClass(transformed_expr, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

    # Implement _eval_* methods using _apply_sympy_function
    def _eval_factor(self, **hints):
        return self._apply_sympy_function(sympy.factor, **hints)
    
    def _eval_cancel(self, **hints):
        return self._apply_sympy_function(self.polyExpr.cancel, **hints)

    def _eval_diff(self, *symbols, **hints):
        return self._apply_sympy_function(self.polyExpr.diff, *symbols, **hints)

    def _eval_integrate(self, *symbols, **hints):
        return self._apply_sympy_function(self.polyExpr.integrate, *symbols, **hints)
    
    def _eval_expand(self, **hints):
        return self._apply_sympy_function(self.polyExpr.expand, *symbols, **hints)

    def __add__(self, other):
        """
        Adds the current polynomial expression to another polynomial or scalar.

        Parameters
        ----------
        other : DGCVPolyClass or sympy.Expr
            The object to add.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the added expression.
        """
        if isinstance(other, DGCVPolyClass):
            # Combine varSpaces, preserving order and removing duplicates
            new_varSpace = tuple(dict.fromkeys(self.varSpace + other.varSpace))
            return DGCVPolyClass(self.polyExpr + other.polyExpr, varSpace=new_varSpace, degreeUpperBound=self.degreeUpperBound)

        elif isinstance(other, (int, float, sympy.Expr)):
            # Add directly
            return DGCVPolyClass(self.polyExpr + other, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

        return NotImplemented

    def __sub__(self, other):
        """
        Subtracts another polynomial or scalar from the current polynomial expression.

        Parameters
        ----------
        other : DGCVPolyClass or sympy.Expr
            The object to subtract.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the subtracted expression.
        """
        if isinstance(other, DGCVPolyClass):
            # Combine varSpaces, preserving order and removing duplicates
            new_varSpace = tuple(dict.fromkeys(self.varSpace + other.varSpace))
            return DGCVPolyClass(self.polyExpr - other.polyExpr, varSpace=new_varSpace, degreeUpperBound=self.degreeUpperBound)

        elif isinstance(other, (int, float, sympy.Expr)):
            # Subtract directly
            return DGCVPolyClass(self.polyExpr - other, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

        return NotImplemented

    def __mul__(self, other):
        """
        Multiplies the current polynomial expression by another polynomial or scalar.

        Parameters
        ----------
        other : DGCVPolyClass or sympy.Expr
            The object to multiply with. This can be a DGCVPolyClass instance, a scalar, or a SymPy expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the multiplied polynomial expression.

        Raises
        ------
        ValueError
            If the multiplication involves a non-polynomial or unsupported expression.
        """
        # Case 1: Multiplying by another DGCVPolyClass instance
        if isinstance(other, DGCVPolyClass):
            # Combine varSpaces, preserving order and removing duplicates
            new_varSpace = tuple(dict.fromkeys(self.varSpace + other.varSpace))
            
            # Determine the new degreeUpperBound as the minimum of the non-None values
            if self.degreeUpperBound is None and other.degreeUpperBound is None:
                new_degreeUpperBound = None
            else:
                degree_values = [deg for deg in [self.degreeUpperBound, other.degreeUpperBound] if deg is not None]
                new_degreeUpperBound = min(degree_values) if degree_values else None
            
            return DGCVPolyClass(self.polyExpr * other.polyExpr, varSpace=new_varSpace, degreeUpperBound=new_degreeUpperBound)

        # Case 2: Multiplying by a scalar (constant or polynomial)
        elif isinstance(other, (int, float, sympy.Number, sympy.Poly, sympy.Symbol)):
            # Multiply directly
            return DGCVPolyClass(self.polyExpr * other, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

        # Case 3: Handle non-polynomial expressions
        elif isinstance(other, sympy.Expr):
            # Check if the scalar is a polynomial
            if sympy.Poly(other, self.varSpace).is_zero:
                raise ValueError("Multiplication with non-polynomial expressions is not supported.")

            # Proceed with multiplication (but warn it's non-polynomial)
            return DGCVPolyClass(self.polyExpr * other, varSpace=self.varSpace, degreeUpperBound=self.degreeUpperBound)

        # Case 4: Unsupported types
        return NotImplemented

    def __rmul__(self, other):
        """
        Right-hand multiplication by a scalar (or polynomial).

        Parameters
        ----------
        other : int, float, sympy.Expr
            A scalar or polynomial expression.

        Returns
        -------
        DGCVPolyClass
            A new DGCVPolyClass instance with the multiplied expression.
        """
        return self.__mul__(other)

    def leading_term(self):
        terms = sorted(self.polyMonomials, key=lambda x: sum(x[1]), reverse=True)
        return terms[0] if terms else None
     
# finite dimensional algebra class
class FAClass(Basic):
    def __new__(cls, structure_data, *args, **kwargs):
        # Validate structure data
        validated_structure_data = cls.validate_structure_data(structure_data, process_matrix_rep=kwargs.get('process_matrix_rep', False))

        # Create the new instance
        obj = Basic.__new__(cls, validated_structure_data)

        # Attach structureData directly for use in __init__
        obj.structureData = validated_structure_data

        return obj

    def __init__(self, structure_data, grading=None, format_sparse=False, process_matrix_rep=False, _label=None, _basis_labels=None, _calledFromCreator=None):
        # Handle structure_data in __init__
        self.structureData = structure_data

        # Detect if initialized from creator (using the passkey)
        if _calledFromCreator == retrieve_passkey():
            self.label = _label
            self.basis_labels = _basis_labels
        else:
            self.label = None
            self.basis_labels = None

        # Other attributes
        self.is_sparse = format_sparse
        self.dimension = len(self.structureData)
        self.grading = Tuple(*grading) if grading else Tuple(*([0] * self.dimension))
        self.basis = [AlgebraElement(self, [1 if i == j else 0 for j in range(self.dimension)], format_sparse=format_sparse) for i in range(self.dimension)]
        self._built_from_matrices = process_matrix_rep

        # Caches for check methods
        self._skew_symmetric_cache = None
        self._jacobi_identity_cache = None

    @staticmethod
    def validate_structure_data(data, process_matrix_rep=False):
        """
        Validates the structure data and converts it to a list of lists of lists.

        Parameters
        ----------
        data : list or array-like
            The structure data to be validated. Can be:
            - A 3D list of structure constants (list of lists of lists)
            - A list of VFClass objects
            - A list of square matrices

        process_matrix_rep : bool, optional
            If True, the function will interpret the data as a list of square matrices
            and pass them to algebraDataFromMatRep to compute the structure data.

        Returns
        -------
        list
            The validated structure data as a list of lists of lists.

        Raises
        ------
        ValueError
            If the structure data is invalid or cannot be processed into a valid 3D list.
        """
        # Case 1: If process_matrix_rep is True, handle matrix representation
        if process_matrix_rep:
            if all(isinstance(Matrix(obj), Matrix) and Matrix(obj).shape[0] == Matrix(obj).shape[1] for obj in data):
                return algebraDataFromMatRep(data)
            else:
                raise ValueError("Matrix representation requires a list of square matrices.")

        # Case 2: If the input is a list of VFClass objects, handle vector fields
        if all(isinstance(obj, VFClass) for obj in data):
            return algebraDataFromVF(data)

        # Case 3: Validate and return the data as a list of lists of lists
        try:
            # Ensure the data is a 3D list-like structure
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                if len(data) == len(data[0]) == len(data[0][0]):
                    return data  # Return as a validated 3D list
                else:
                    raise ValueError("Structure data must have 3D shape (x, x, x).")
            else:
                raise ValueError("Structure data format must be a 3D list of lists.")
        except Exception as e:
            raise ValueError(f"Invalid structure data format: {type(data)} - {e}")

    def __getitem__(self, indices):
        return self.structureData[indices[0]][indices[1]][indices[2]]

    def __repr__(self):
        """
        Provides a detailed representation of the FAClass object.
        """
        return (f"FAClass(dim={self.dimension}, grading={self.grading}, "
                f"label={self.label}, basis_labels={self.basis_labels}, "
                f"struct_data={self.structureData})")  # Avoid SymPy Matrix
    
    def _structure_data_summary(self):
        """
        Provides a summary of the structure data to avoid printing large matrices.
        """
        if self.dimension <= 3:
            return self.structureData
        return "Structure data is large. Access the `structureData` attribute for details."
    
    def __str__(self):
        """
        Provides a string representation of the FAClass object.
        """
        def format_basis_label(label):
            """Leave the basis label as is without splitting or rearranging."""
            return label

        # Keep the algebra label as is and format the basis labels properly
        formatted_label = self.label if self.label else "Unnamed Algebra"
        formatted_basis_labels = ", ".join([format_basis_label(bl) for bl in self.basis_labels])

        return (f"Algebra: {formatted_label}\n"
                f"Dimension: {self.dimension}\n"
                f"Grading: {self.grading}\n"
                f"Basis: {formatted_basis_labels}")

    def _display_DGCV_hook(self):
        def format_algebra_label(label):
            """Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
            if label[-1].isdigit():
                # Split into text and number parts for subscript formatting
                label_text = ''.join(filter(str.isalpha, label))
                label_number = ''.join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label.islower():
                return rf"\mathfrak{{{label}}}"
            return label
        return format_algebra_label(self.label)

    def _repr_latex_(self):
        """
        Provides a LaTeX representation of the FAClass object for Jupyter notebooks.
        """
        def format_algebra_label(label):
            """Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
            if label[-1].isdigit():
                # Split into text and number parts for subscript formatting
                label_text = ''.join(filter(str.isalpha, label))
                label_number = ''.join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label.islower():
                return rf"\mathfrak{{{label}}}"
            return label

        def format_basis_label(label):
            """Leave the basis label as is without splitting or rearranging."""
            return rf"{label}"

        # Format the algebra label using \mathfrak{} and subscript
        formatted_label = format_algebra_label(self.label) if self.label else "Unnamed Algebra"

        # Format the basis labels and join them with commas
        formatted_basis_labels = ", ".join([format_basis_label(bl) for bl in self.basis_labels])

        # LaTeX output for the algebra
        return f"Algebra: ${formatted_label}$, Basis: ${formatted_basis_labels}$, Dimension: ${self.dimension}$, Grading: ${latex(self.grading)}$"
    
    def _sympystr(self):
        """
        SymPy string representation for FAClass.
        """
        if self.label:
            return f"FAClass({self.label}, dim={self.dimension})"
        else:
            return f"FAClass(dim={self.dimension})"

    def _structure_data_summary_latex(self):
        """
        Provides a LaTeX summary of the structure data.
        Converts only numerical/symbolic elements to a matrix when possible.
        """
        try:
            # Check if structureData contains only symbolic or numeric elements
            if self._is_symbolic_matrix(self.structureData):
                return sympy.latex(Matrix(self.structureData))  # Convert to matrix if valid
            else:
                return str(self.structureData)  # Fallback to basic string representation
        except Exception:
            return str(self.structureData)  # Fallback in case of an error

    def _is_symbolic_matrix(self, data):
        """
        Checks if the matrix contains only symbolic or numeric entries.
        """
        return all(all(isinstance(elem, sympy.Basic) for elem in row) for row in data)

    def is_skew_symmetric(self, verbose=False):
        if self._skew_symmetric_cache is None:
            result, failure = self._check_skew_symmetric()
            self._skew_symmetric_cache = (result, failure)
        else:
            result, failure = self._skew_symmetric_cache

        if verbose:
            if result:
                print('The algebra is skew symmetric.')
            else:
                i, j, k = failure
                print(f'Skew symmetry fails for basis elements {i}, {j}, at vector index {k}.')
        
        return result

    def _check_skew_symmetric(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(len(self.structureData[i][j])):
                    vector_sum_element = simplify(self.structureData[i][j][k] + self.structureData[j][i][k])
                    if vector_sum_element != 0:
                        return False, (i, j, k)
        return True, None

    def satisfies_jacobi_identity(self, verbose=False):
        if self._jacobi_identity_cache is None:
            result, fail_list = self._check_jacobi_identity()
            self._jacobi_identity_cache = (result, fail_list)
        else:
            result, fail_list = self._jacobi_identity_cache

        if verbose:
            if result:
                print('The algebra satisfies the Jacobi identity.')
            else:
                print(f'Jacobi identity fails for the following triples: {fail_list}')
        
        return result

    def _check_jacobi_identity(self):
        fail_list = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    if not (self.basis[i] * self.basis[j] * self.basis[k] +
                            self.basis[j] * self.basis[k] * self.basis[i] +
                            self.basis[k] * self.basis[i] * self.basis[j]).is_zero():
                        fail_list.append((i, j, k))
        if fail_list:
            return False, fail_list
        return True, None
    
    def is_lie_algebra(self, verbose=False):
        # Lie algebra requires both skew symmetry and Jacobi identity
        if not self.is_skew_symmetric(verbose=verbose):
            return False
        if not self.satisfies_jacobi_identity(verbose=verbose):
            return False
        if verbose:
            if self.label==None:
                print('The algebra is a Lie algebra.')
            else:
                print(f'{self.label} is a Lie algebra.')
        return True

    def is_semisimple(self, verbose=False):
        if not self.is_lie_algebra(verbose=False):
            return False
        det = simplify(killingForm(self).det())
        if verbose:
            if det != 0:
                if self.label==None:
                    print('The algebra is a semisimple.')
                else:
                    print(f'{self.label} is semisimple.')
            else:
                if self.label==None:
                    print('The algebra is a not semisimple.')
                else:
                    print(f'{self.label} is not semisimple.')                
        return det != 0
    
# algebra element class
class AlgebraElement(Basic):
    def __new__(cls, algebra, coeffs, format_sparse=False):
        # Ensure the algebra is of type FAClass
        if not isinstance(algebra, FAClass):
            raise TypeError('AlgebraElement expects the first argument to be an instance of FAClass.')

        # Prepare coefficients as a symbolic tuple or matrix
        coeffs = tuple(coeffs)

        # Call Basic.__new__ with these symbolic properties
        obj = Basic.__new__(cls, algebra, coeffs, format_sparse)

        # Return the new instance
        return obj

    def __init__(self, algebra, coeffs, format_sparse=False):
        # Initialize data that doesn't need to be part of __new__
        self.algebra = algebra
        if format_sparse:
            self.coeffs = SparseMatrix(coeffs)
        else:
            self.coeffs = Matrix(coeffs)

        self.is_sparse = format_sparse

    def __str__(self):
        """
        Custom string representation for AlgebraElement.
        Displays the linear combination of basis elements with coefficients.
        Wraps multi-term coefficients in parentheses and handles the '+-' case.
        Special case: displays "0 * first_basis_element" when all coefficients are zero.
        """
        terms = []
        for coeff, basis_label in zip(self.coeffs, self.algebra.basis_labels):
            if coeff == 0:
                continue  # Skip zero terms
            elif coeff == 1:
                terms.append(f"{basis_label}")  # Suppress 1 as coefficient
            elif coeff == -1:
                terms.append(f"-{basis_label}")  # Suppress 1 but keep the negative sign
            else:
                # Check if the coefficient has more than one term (e.g., 1 + I)
                if isinstance(coeff, sympy.Expr) and len(coeff.args) > 1:
                    terms.append(f"({coeff}) * {basis_label}")  # Wrap multi-term coefficient in parentheses
                else:
                    terms.append(f"{coeff} * {basis_label}")  # Single-term coefficient

        # Handle special case: all zero coefficients
        if not terms:
            return f"0 * {self.algebra.basis_labels[0]}"

        # Join terms with proper sign handling
        result = " + ".join(terms).replace("+ -", "- ")
        return result

    def _repr_latex_(self):
        """
        Provides a LaTeX representation of AlgebraElement for Jupyter notebooks.
        Displays the linear combination of basis elements with coefficients.
        Wraps multi-term coefficients in parentheses and handles the '+-' case in LaTeX.
        Special case: displays "0 \cdot first_basis_element" when all coefficients are zero.
        """
        terms = []
        for coeff, basis_label in zip(self.coeffs, self.algebra.basis_labels):
            if coeff == 0:
                continue  # Skip zero terms
            elif coeff == 1:
                terms.append(rf"{basis_label}")  # Suppress 1 as coefficient
            elif coeff == -1:
                terms.append(rf"-{basis_label}")  # Suppress 1 but keep the negative sign
            else:
                # Check if the coefficient has more than one term (e.g., 1 + I)
                if isinstance(coeff, sympy.Expr) and len(coeff.args) > 1:
                    terms.append(rf"({sympy.latex(coeff)}) \cdot {basis_label}")  # Wrap multi-term coefficient in parentheses
                else:
                    terms.append(rf"{sympy.latex(coeff)} \cdot {basis_label}")  # Single-term coefficient

        # Handle special case: all zero coefficients
        if not terms:
            return rf"$0 \cdot {self.algebra.basis_labels[0]}$"

        def format_algebra_label(label):
            """Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
            if label[-1].isdigit():
                # Split into text and number parts for subscript formatting
                label_text = ''.join(filter(str.isalpha, label))
                label_number = ''.join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label.islower():
                return rf"\mathfrak{{{label}}}"
            return label

        # Join terms with proper LaTeX sign handling
        result = " + ".join(terms).replace("+ -", "- ")
        return rf"$\text{{Element of }} {format_algebra_label(self.algebra.label)}: {result}$"

    def _sympystr(self):
        """
        SymPy string representation for AlgebraElement.
        """
        coeffs_str = ", ".join(map(str, self.coeffs))
        if self.algebra.label:
            return f"AlgebraElement({self.algebra.label}, coeffs=[{coeffs_str}])"
        else:
            return f"AlgebraElement(coeffs=[{coeffs_str}])"

    def __repr__(self):
        """
        Provides a detailed representation of the AlgebraElement object.
        """
        return (f"AlgebraElement(algebra={self.algebra.label}, "
                f"coefficients={self.coeffs}, basis_labels={self.algebra.basis_labels})")

    def is_zero(self):
        for j in self.coeffs:
            if simplify(j) != 0:
                return False
        else:
            return True

    def subs(self,subsData):
        newCoeffs=[sympify(j).subs(subsData) for j in self.coeffs]
        return AlgebraElement(self.algebra, newCoeffs, format_sparse = self.is_sparse)

    def __add__(self, other):
        if isinstance(other, AlgebraElement):
            if self.algebra == other.algebra:
                return AlgebraElement(self.algebra, [self.coeffs[j] + other.coeffs[j] for j in range(len(self.coeffs))], format_sparse=self.is_sparse)
            else:
                raise TypeError("AlgebraElement operands for + must belong to the same FAClass.")
        else:
            raise TypeError("Unsupported operand type(s) for + with the AlgebraElement class")

    def __sub__(self, other):
        if isinstance(other, AlgebraElement):
            if self.algebra == other.algebra:
                return AlgebraElement(self.algebra, [self.coeffs[j] - other.coeffs[j] for j in range(len(self.coeffs))], format_sparse=self.is_sparse)
            else:
                raise TypeError("AlgebraElement operands for - must belong to the same FAClass.")
        else:
            raise TypeError("Unsupported operand type(s) for - with the AlgebraElement class")

    def __mul__(self, other):
        """
        Multiplies two AlgebraElement objects by multiplying their coefficients
        and summing the results based on the algebra's structure constants. Also handles 
        multiplication with scalars.

        Args:
            other (AlgebraElement) or (scalar): The algebra element or scalar to multiply with.

        Returns:
            AlgebraElement: The result of the multiplication.
        """
        if isinstance(other, AlgebraElement):
            if self.algebra == other.algebra:
                # Initialize result coefficients as a list of zeros
                result_coeffs = [0] * self.algebra.dimension
                
                # Loop over each pair of basis element coefficients
                for i in range(self.algebra.dimension):
                    for j in range(self.algebra.dimension):
                        # Compute the scalar product of the current coefficients
                        scalar_product = self.coeffs[i] * other.coeffs[j]
                        
                        # Multiply scalar_product with the corresponding vector from structureData
                        structure_vector_product = [
                            scalar_product * element for element in self.algebra.structureData[i][j]
                        ]
                        
                        # Sum the resulting vector into result_coeffs element-wise
                        result_coeffs = [
                            sympify(result_coeffs[k] + structure_vector_product[k]) for k in range(len(result_coeffs))
                        ]
                
                # Convert result_coeffs to an ImmutableDenseNDimArray
                result_coeffs = ImmutableDenseNDimArray(result_coeffs)
                
                # Return a new AlgebraElement with the updated coefficients
                return AlgebraElement(self.algebra, result_coeffs, format_sparse=self.is_sparse)
            else:
                raise TypeError("Both operands for * must be AlgebraElement instances from the same FAClass.")
        elif isinstance(other, (int, float, sympy.Expr)):
            # Scalar multiplication case
            new_coeffs = [coeff * other for coeff in self.coeffs]
            # Convert to ImmutableDenseNDimArray
            new_coeffs = ImmutableDenseNDimArray(new_coeffs)
            # Return a new AlgebraElement with the updated coefficients
            return AlgebraElement(self.algebra, new_coeffs, format_sparse=self.is_sparse)   
        else:
            raise TypeError(f"Multiplication is only supported for scalars and the AlegebraElement class, not {type(other)}")

    def __rmul__(self, other):
        # If other is a scalar, treat it as commutative
        if isinstance(other, (int, float, sympy.Expr)):  # Handles numeric types and SymPy scalars
            return self * other  # Calls __mul__ (which is already implemented)
        elif isinstance(other, AlgebraElement):
            other * self
        else:
            raise TypeError(f"Right multiplication is only supported for scalars and the AlegebraElement class, not {type(other)}")


############## variable creation
def createVariables(variable_label,real_label=None,imaginary_label=None, number_of_variables=None, initialIndex=1, withVF=None, complex=None, multiindex_shape=None, assumeReal=None, remove_guardrails=None, default_var_format=None):
    """
    This function serves as the default interface for creating variables within the DGCV package. It supports creating both standard variable systems and complex variable systems, with options for initializing coordinate vector fields and differential forms. Variables created through `createVariables` are automatically tracked within DGCV’s Variable Management Framework (VMF) and are assigned labels validated through a safeguards routine that prevents overwriting important labels.

    Parameters
    ----------
    variable_label : str
        The label for the primary variable or system of variables to be created. If creating a complex variable system, 
        this will correspond to the holomorphic variable(s), whilst antiholomorphic variable(s) recieve this label pre-pended with "BAR".
        
    real_label : str, optional
        The label for the real part of the complex variable system. Required only when creating complex variable systems.
        
    imaginary_label : str, optional
        The label for the imaginary part of the complex variable system. Required only when creating complex variable systems.
        
    number_of_variables : int, optional
        The number of variables to be created, used to initialize a tuple of variables rather than a single variable (e.g., x=(x1, x2, x3) rather than just x).

    initialIndex : int, optional, default=1
        The starting index for tuple variables, allowing for flexible indexing when initializing variable systems 
        as tuples (e.g., x=(x0, x1, x2) with `initialIndex=0`).
        
    withVF : bool, optional
        If set to True, creates associated coordinate vector fields and differential forms for the variable(s) in the system.

    complex : bool, optional
        Specifies whether to create a complex variable system. If not provided, the function will infer whether to create 
        a complex system based on whether `real_label` or `imaginary_label` is provided. If provided a complex variable system will be created regardless of `real_label` and  `imaginary_label` settings, and string labels are automaticaly created for `real_label` and  `imaginary_label` if they are not provided.

    assumeReal : bool, optional
        If set to True, specifies that the variables being created are real-valued. This is only relevant for standard 
        variable systems and is ignored for complex systems.

    remove_guardrails : bool, optional
        If set to True, bypasses DGCV's safeguard system for variable labeling, allowing one to overwrite certain 
        reserved labels. Use with caution, as it may overwrite important variables in the global namespace.
        
    default_var_format : {'complex', 'real'}, optional
        Relevant only for complex variable systems. Specifies whether the system's vector fields and differential forms 
        default to real coordinate expressions (`real`) or holomorphic coordinate expressions (`complex`). If not provided, 
        the default is holomorphic coordinates.

    Returns
    -------
    None
        This function creates the specified variable system and registers it within DGCV’s Variable Management Framework.

    Functionality
    -------------
    - Creates standard or complex variable systems.
    - Automatically registers all created variables, vector fields, and differential forms in DGCV’s VMF.
    - Safeguards are applied to ensure that no critical Python or DGCV internal functions are overwritten.

    Notes
    -----
    - If `complex=True`, the function will automatically generate holomorphic, antiholomorphic, real, and imaginary parts 
    for the variable system, along with corresponding coordinate vector fields and differential forms.
    - If `withVF=True`, the function will create both vector fields and differential forms for standard variable systems.

    Examples
    --------
    # Creating a single standard variable 'alpha'
    >>> createVariables('alpha')

    # Creating a tuple of 3 standard variables with associated vector fields and differential forms
    >>> createVariables('alpha', 3, withVF=True)

    # Creating a single complex variable system with real part 'beta' and imaginary part 'gamma'
    >>> createVariables('alpha', 'beta', 'gamma')

    # Creating a tuple of 3 complex variables with real parts 'beta1', 'beta2', 'beta3' and imaginary parts 'gamma1', 'gamma2', 'gamma3'
    >>> createVariables('alpha', 'beta', 'gamma', 3)

    # Creating a tuple of 3 standard variables with custom initial index
    >>> createVariables('alpha', 3, initialIndex=0)

    # Creating a complex variable system without specifying real and imaginary part labels. Note, this automatically creates intentionally obscure real and imaginary part labels.
    >>> createVariables('alpha', complex=True)

    Warnings
    --------
    If `complex=True` is provided along with incompatible keywords, such as `withVF=False`, or `complex=False` is provided while values are also given for `real_label` and `imaginary_label`, the function will resolve conflicts internally and issue a warning about the resolution.

    Use `DGCV_snapshot()` for a clear summary of the variables created and tracked within the DGCV VMF.

    """    
    if multiindex_shape != None:
        if any(j!=None for j in [real_label,imaginary_label,withVF,complex]) or default_var_format=='complex':
            warnings.warn('A value for `multiindex_shape` was provided, so a standard variable system without vector fields or differential forms was created alligned with the multiindex_shape value. Multiindex variable labeling is not yet supported by DGCV\'s automated variable creation with vector fields or for complex variable systems. It will be in future DGCV updates. ')
            real_label = None
            imaginary_label = None
            withVF = None
            complex = None
            default_var_format = None
    if not isinstance(variable_label,str):
        raise TypeError('`createVariables` requires its first argument to be a string, which will be used in lables for the created variables.')
    if isinstance(real_label,int) and imaginary_label==None and number_of_variables==None:
        number_of_variables=real_label
        real_label=None
    if real_label!=None and not isinstance(real_label,str):
            raise TypeError('A non-string value cannot be assigned to the `real_label` keyword of `createVariables`')
    if real_label!=None and not isinstance(imaginary_label,str):
            raise TypeError('A non-string value cannot be assigned to the `imaginary_label` keyword of `createVariables`')
    if complex==True and withVF==False:
        warnings.warn('`createVariables` was called with `complex=True` and `withVF=False`. The latter keyword was disregarded because DGCV automatically initializes associated differential objects whenever complex variable systems are created.')
    if complex==True and assumeReal==True:
        warnings.warn('`createVariables` was called with `complex=True` and `assumeReal=True`. The latter keyword was disregarded because DGCV has fixed variable assumptions for elements in its complex variable systems.')
    if complex==False and any([real_label!=None,imaginary_label!=None]):
        warnings.warn('`createVariables` recieved `complex=False` and recieved values for the `real_label` or `imaginary_label` keywords. Honoring `complex=False`, only a standard variable system was created, and latter labels were disregarded. Set `complex=True` if a complex variable system is needed instead.')
        real_label=None
        imaginary_label=None
    elif complex==True and all([real_label==None,imaginary_label==None]):
        real_label=variable_label+'REAL'+retrieve_public_key()
        imaginary_label=variable_label+'IM'+retrieve_public_key()
        warnings.warn('`createVariables` recieved `complex=True` did not recieve value assignements for `imaginary_label` or `real_label`, so intentionally obscure labels were created for both the real and imaginary variables in the created complex variable system. To have nicer labling while using `complex=True`, provide a preferred string label for the `real_label` and `imaginary_label` keywords.')
    elif any([real_label!=None,imaginary_label!=None]):
        if complex!=True and complex!=None:
            warnings.warn('The keyword \'complex\' was set to a non-Bool value. Since a string value was also assigned to either `real_label` or `imaginary_label`, `createVariables` proceeded under the assumption that it should create a complex variable system. If a standard variable system was prefered then set `complex=False` instead.')
        complex=True
        if real_label==None:
            real_label=variable_label+'REAL'+retrieve_public_key()
            warnings.warn('`createVariables` recieved a value assigned to `imaginary_label` but not `real_label`, so an intentionally obscure label was created for the real variables in the created complex variable system. To have nice nicer labling, provide a preferred string label to `real_label` instead.')
        if imaginary_label==None:
            imaginary_label=variable_label+'IM'+retrieve_public_key()
            warnings.warn('`createVariables` recieved a value assigned to `real_label` but not `imaginary_label`, so an intentionally obscure label was created for the imaginary variables in the created complex variable system. To have nice nicer labling, provide a preferred string label to `imaginary_label` instead.')

    def reformat_string(input_string: str):
        # Replace commas with spaces, then split on spaces
        substrings = input_string.replace(",", " ").split()
        # Return the list of non-empty substrings
        return [s for s in substrings if len(s) > 0]
    if isinstance(variable_label,str):
        variable_label=reformat_string(variable_label)
    if isinstance(real_label,str):
        real_label=reformat_string(real_label)
    if isinstance(imaginary_label,str):
        imaginary_label=reformat_string(imaginary_label)

    if complex==True:
        complexVarProc(variable_label, real_label, imaginary_label, number_of_variables=number_of_variables, initialIndex=initialIndex, default_var_format=default_var_format, remove_guardrails=remove_guardrails)
        return
    elif withVF==True:
        varWithVF(variable_label, number_of_variables=number_of_variables, initialIndex=initialIndex, _doNotUpdateVar=False, assumeReal=assumeReal, _calledFromCVP=None, remove_guardrails=remove_guardrails)
        return
    else:
        variableProcedure(variable_label, number_of_variables=number_of_variables, initialIndex=initialIndex, assumeReal=assumeReal, multiindex_shape=multiindex_shape, _tempVar=False, _doNotUpdateVar=None, _calledFromCVP=None, remove_guardrails=remove_guardrails)

############## standard variables

def variableProcedure(variables_label, number_of_variables=None, initialIndex=1, multiindex_shape=None, assumeReal=None, _tempVar=None, _doNotUpdateVar=None, _calledFromCVP=None, remove_guardrails=None, _obscure=None):
    """
    Initializes one or more standard variables (single or tuples) and integrates them into DGCV's Variable Management Framework.

    Parameters:
    ----------
    variables_label : str or list of str
        The label for the variable(s). If a string, it defines one variable set. If a list of strings, it initializes multiple variable sets.
        
    number_of_variables : positive int, optional
        If provided, specifies the number of variables to be initialized and labels them in an an enumerated tuple creating labels based on variables_label. If not provided, a single variable is initialized.
        
    initialIndex : int, optional
        The starting index for enumerating variable names in a tuple. Defaults to 1.
        
    assumeReal : bool, optional
        If provided, specifies whether to treat the variable(s) as real. If True, the variable(s) are assumed to be real.
     
    _tempVar : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, marks the variable system as temporary. Temporary variable systems will be tracked in the 'temporary_variables' set of the internal variable_registry dict.

    _obscure : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, marks the variable system as \'obscure\'. This is used by functions that need to create variables on the fly with sufficiently obscure labels such that they are unlikely to overwrite anything. Unlike `temporary` variables, the only DGCV function authorized to clear `obscure` variables is `clearVar`.  They are tracked in the 'obscure_variables' set of the internal variable_registry dict.


    _doNotUpdateVar : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, prevents the variable(s) from being cleared and reinitialized. Used when avoiding overwriting existing variables.
           
    _calledFromCVP : None
        Intended for internal DGCV use. If set to the internal DGCV passkey, appropriately interfaces with the Variable Management Framework supposing this is called from `complexVarProc`.

    Updates the internal variable_registry dict:
    --------------------------------------------
    - For a single variable:
        - Adds a new entry in 'standard_variable_systems' with:
            - 'family_type': 'single'
            - 'family_names': (str,)
            - 'family_values': (variable,)
            - 'differential_system': bool (default None)
            - 'tempVar': bool (default None)
            - 'initial_index': None
            - 'variable_relatives': A dictionary containing the variable's label and associated properties.
    
    - For a tuple of variables:
        - Adds a new entry in 'standard_variable_systems' with:
            - 'family_type': 'tuple'
            - 'family_names': tuple of strings
            - 'family_values': tuple of variables
            - 'differential_system': bool (default None)
            - 'tempVar': bool (default None)
            - 'initial_index': The starting index of the tuple
            - 'variable_relatives': A dictionary mapping each variable in the tuple to its associated properties.
    
    Raises:
    -------
    Exception:
        If the variable label is part of the 'protected_variables' set in the internal variable_registry dict and this function was not called from complexVarProc.
    """
    variable_registry = get_variable_registry()
    # Check if the variable is part of the protected variables when not called from complexVarProc
    if not _calledFromCVP==retrieve_passkey():
        for j in (tuple(variables_label) if isinstance(variables_label, (list,tuple)) else (variables_label,)):
            if j in variable_registry.get('protected_variables', set()):
                raise Exception(f"{variables_label} is already assigned to the real or imaginary part of a complex variable system, so DGCV variable creation functions will not reassign it as a standard variable. Instead, use the clearVar function to remove the conflicting CV system first before implementing such reassignments.")

    for j in (tuple(variables_label) if isinstance(variables_label, (list,tuple)) else (variables_label,)):
        if not _calledFromCVP==retrieve_passkey() and not remove_guardrails:
            labelLoc=validate_label(j)
        else:
            labelLoc = j

        # Clear variable if necessary and _doNotUpdateVar is not set
        if _doNotUpdateVar != retrieve_passkey():
            clearVar(j,report=False)

        # Add each tuple variable to temporary variables if marked as temporary
        if _tempVar==retrieve_passkey():
            variable_registry['temporary_variables'].add(labelLoc)
            _tempVar=True

        # Add each tuple variable to temporary variables if marked as temporary
        if _obscure==retrieve_passkey():
            variable_registry['obscure_variables'].add(labelLoc)
            _obscure=True


        # Handle lone variable
        if isinstance(multiindex_shape, (list, tuple)) and all(isinstance(n, int) and n > 0 for n in multiindex_shape):
            # Ensure that all elements in multiindex_shape are positive integers
            
            # Generate multi-index variable names using carProd
            indices = list(carProd(*[range(initialIndex, initialIndex + n) for n in multiindex_shape]))
            var_names = [f"{labelLoc}_{'_'.join(map(str, idx))}" for idx in indices]
            vars = [symbols(f"{labelLoc}_{'_'.join(map(str, idx))}", real=assumeReal) for idx in indices]

            # Update globals
            _cached_caller_globals.update(zip(var_names, vars))
            _cached_caller_globals[labelLoc] = vars  # Storing as list, since multi-index systems may not naturally tuple together

            # Create parent dictionary for the multi-indexed variables
            if _doNotUpdateVar != retrieve_passkey():
                variable_registry['standard_variable_systems'][labelLoc] = {
                    'family_type': 'multi_index',
                    'family_shape': multiindex_shape,
                    'family_names': tuple(var_names),
                    'family_values': _cached_caller_globals[labelLoc],
                    'differential_system': None,
                    'tempVar': _tempVar,
                    'obsVar': _obscure,
                    'initial_index': initialIndex,
                    'variable_relatives': {
                        var_name: {'VFClass': None, 'DFClass': None, 'assumeReal': assumeReal} for var_name in var_names
                    }
                }

        # Handle lone variable
        elif number_of_variables==None:
            symbol = symbols(labelLoc, real=assumeReal)
            _cached_caller_globals[labelLoc] = symbol

            # Create a parent dictionary for lone variable
            if _doNotUpdateVar != retrieve_passkey():
                variable_registry['standard_variable_systems'][labelLoc] = {
                'family_type': 'single',
                'family_names': (labelLoc,),
                'family_values': (_cached_caller_globals[labelLoc],),
                'differential_system': None,
                'tempVar': _tempVar,
                'obsVar': _obscure,
                'initial_index': None,
                'variable_relatives': {
                    labelLoc: {'VFClass': None, 'DFClass': None, 'assumeReal': assumeReal}
                }
                }

        # Handle tuple of variables
        elif isinstance(number_of_variables, int) and number_of_variables>=0:
            lengthLoc = number_of_variables
            var_names = [f"{labelLoc}{i}" for i in range(initialIndex, lengthLoc + initialIndex)]
            vars = [symbols(f"{labelLoc}{i}", real=assumeReal) for i in range(initialIndex, lengthLoc + initialIndex)]
            _cached_caller_globals.update(zip(var_names, vars))
            _cached_caller_globals[labelLoc] = tuple(vars)

            # Create parent dictionary for the tuple of variables
            if _doNotUpdateVar != retrieve_passkey():
                variable_registry['standard_variable_systems'][labelLoc] = {
                'family_type': 'tuple',
                'family_names': tuple(var_names),
                'family_values': _cached_caller_globals[labelLoc],
                'differential_system': None,
                'tempVar': _tempVar,
                'obsVar': _obscure,
                'initial_index': initialIndex,
                'variable_relatives': {
                    var_name: {'VFClass': None, 'DFClass': None, 'assumeReal': assumeReal} for var_name in var_names
                }
                }
        else:
            raise ValueError('variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided.')

def varProcMultiIndex(arg1,arg2,arg3):
    """
    Initializes a variable with label arg1 equal to a 2d array of Symbol objects labeled arg1=(arg1_1_1,arg1_1_2,...,arg1_arg2_arg3).
    
    Args:
        arg1: string
        arg2: positive integer
        arg3: positive integer
    
    Returns:
        Nothing
    
    Raises:
        NA
    """
    warnings.warn('`varProcMultiIndex` is depricated and will be removed from DGCV in future updates. Use `createVariables` instead.')
    _cached_caller_globals.update(zip([arg1], [Array([[symbols(arg1+'_{}_{}'.format(k,j)) for j in range(1,arg3+1)] for k in range(1,arg2+1)])]))
    _cached_caller_globals.update(zip([arg1+'_{}_{}'.format(j,k) for j in range(1,arg3+1) for k in range(1,arg2+1)], [symbols(arg1+'_{}_{}'.format(j,k)) for j in range(1,arg3+1) for k in range(1,arg2+1)]))

def varWithVF(variables_label, number_of_variables=None, initialIndex=1, _doNotUpdateVar=False, assumeReal=None, _calledFromCVP=None, remove_guardrails=None):
    """
    Initializes one or more standard variables with accompanying vector fields and differential 1-forms.
    Updates the internal variable_registry dict accordingly.

    Parameters:
    ----------
    variables_label : str or list
        The label for the variable(s). If a string, it defines a single variable. If a list, it initializes multiple variables.
        
    number_of_variables : positive int, optional
        If provided, specifies the number of variables in a tuple. The first argument in `args` defines the length of the tuple.
        
    initialIndex : int, optional
        The starting index for enumerating variable names in a tuple. Defaults to 1.
        
    _doNotUpdateVar : bool, optional
        If True, prevents the variable(s) from being cleared and reinitialized. Used when avoiding overwriting existing variables.
        
    assumeReal : bool, optional
        If provided, specifies whether to treat the variable(s) as real. If True, the variable(s) are assumed to be real.
        
    _calledFromCVP : None
        Reserved for internal DGCV use when called from complexVarProc. Does not affect the behavior in this function.

    Updates the internal variable_registry dict:
    --------------------------
    - For a single variable:
        - Adds a new entry in 'standard_variable_systems' with:
            - 'family_type': 'single'
            - 'family_names': (str,)
            - 'family_values': (variables,)
            - 'differential_system': bool (default None)
            - 'tempVar': bool (default None)
            - 'initial_index': None
            - 'variable_relatives': A dictionary containing the variable's label and its associated properties (VFClass, DFClass, assumeReal).
    
    - For a tuple of variables:
        - Adds a new entry in 'standard_variable_systems' with:
            - 'family_type': 'tuple'
            - 'family_names': tuple of strings
            - 'family_values': tuple of variables
            - 'differential_system': bool (default None)
            - 'tempVar': bool (default None)
            - 'initial_index': The starting index of the tuple
            - 'variable_relatives': A dictionary mapping each variable in the tuple to its associated properties (VFClass, DFClass, assumeReal).
        - Also adds the tuple of variables (e.g., y = (y1, y2, y3)) to _cached_caller_globals.

    Raises:
    -------
    Exception:
        If the variable label is part of the 'protected_variables' set in the internal variable_registry dict and this function was not called from complexVarProc.
    """    
    variable_registry = get_variable_registry()

    # Convert arg1 to a list if it's a single string
    if isinstance(variables_label, str):
        variables_label = [variables_label]

    for labelLoc in variables_label:
        if not _calledFromCVP==retrieve_passkey() and not remove_guardrails:
            labelLoc=validate_label(labelLoc)

        if not _doNotUpdateVar==retrieve_passkey():
            clearVar(labelLoc,report=False)

        # Handle lone variable case
        if number_of_variables==None:
            symbol = symbols(labelLoc, real=assumeReal)
            _cached_caller_globals[labelLoc] = symbol

            # Create vector field and differential form for the lone variable
            vf_instance = VFClass((symbol,), [1])
            df_instance = DFClass((symbol,), {(0,): 1}, 1)
            _cached_caller_globals[f'D_{labelLoc}'] = vf_instance
            _cached_caller_globals[f'd_{labelLoc}'] = df_instance

            # Update standard_variable_systems for the lone variable (only if not called from CVP)
            if not _calledFromCVP==retrieve_passkey():
                variable_registry['standard_variable_systems'][labelLoc] = {
                    'family_type': 'single',
                    'family_values': (symbol,),
                    'family_names': (labelLoc,),
                    'differential_system': True,
                    'tempVar': None,
                    'initial_index': None,
                    'variable_relatives': {
                        labelLoc: {'VFClass': vf_instance, 'DFClass': df_instance, 'assumeReal': assumeReal}
                    }
                }

        # Handle the tuple case
        elif isinstance(number_of_variables,int) and number_of_variables>=0:
            lengthLoc = number_of_variables
            var_names = [f'{labelLoc}{i}' for i in range(initialIndex, lengthLoc + initialIndex)]
            vars = [symbols(f'{labelLoc}{i}', real=assumeReal) for i in range(initialIndex, lengthLoc + initialIndex)]

            # Add each variable to _cached_caller_globals
            _cached_caller_globals.update(zip(var_names, vars))

            # Create the tuple of variables and add it to _cached_caller_globals
            _cached_caller_globals[labelLoc] = tuple(vars)

            # Create individual vector fields
            vf_instances = [
                VFClass(tuple(vars), [1 if k == j else 0 for k in range(len(vars))]) for j in range(len(vars))
            ]
            _cached_caller_globals.update(zip([f'D_{var_name}' for var_name in var_names], vf_instances))

            # Create individual differential 1-forms
            df_instances = [DFClass(tuple(vars), {(j,): 1}, 1) for j in range(len(vars))]
            _cached_caller_globals.update(zip([f'd_{var_name}' for var_name in var_names], df_instances))

            # Update standard_variable_systems for the parent variable (only if not called from CVP)
            if not _calledFromCVP==retrieve_passkey():
                variable_registry['standard_variable_systems'][labelLoc] = {
                    'family_type': 'tuple',
                    'family_values': tuple(vars),
                    'family_names': tuple(var_names),
                    'differential_system': True,
                    'tempVar': None,
                    'initial_index': initialIndex,
                    'variable_relatives': {
                        var_name: {'VFClass': vf_instances[i], 'DFClass': df_instances[i], 'assumeReal': assumeReal} for i, var_name in enumerate(var_names)
                    }
                }
        else:
            raise ValueError('variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided.')


############## complex variables

def complexVarProc(holom_label, real_label, im_label, number_of_variables=None, initialIndex=1, default_var_format='complex', remove_guardrails=None):
    """
    Initializes a complex variable system, linking a holomorphic variable with its real and imaginary parts and 
    a symbolic representative of its complex conjugate (labeled by prepending BAR to the complex variable's label). 

    Parameters
    ----------
    holom_label : str or list of str
        The label or labels for the complex variable(s) are constructed from this. (e.g., If 'z' is provided, constructs 'z' for a single variable or ['z1', 'z2'] etc. for a tuple).
    
    real_label : str or list of str
        The label or labels for the real part(s) of the complex variable(s) are constructed from this. (e.g., If 'x' is provided, constructs 'x' for a single variable or ['x1', 'x2'] etc. for a tuple).
    
    im_label : str or list of str
        The label or labels for the imaginary part(s) of the complex variable(s) are constructed from this. (e.g., If 'y' is provided, constructs 'y' for a single variable or ['y1', 'y2'] etc. for a tuple).
    
    number_of_variables : positive int, optional
        The length of the variable tuple (if initializing a tuple system). If provided, this will initialize the complex variable 
        system as a tuple of length `number_of_variables`.

    initialIndex : int, optional, default=1
        The starting index for tuple variables, used when initializing a tuple of complex variables.

    Functionality
    -------------
    - Creates the specified holomorphic variable(s), its conjugate (antiholomorphic) counterpart(s), and their associated real and imaginary parts.
    - Adds the created variables to the global namespace and tracks them in the internal variable_registry dict.
    - Updates conversion dictionaries (`holToReal`, `realToSym`, `symToHol`, `symToReal`, `realToHol`, and `find_parents`) which allow transformations between the holomorphic and real coordinate systems.
    - Supports both single complex variable systems and tuple-based systems, where multiple complex variables are initialized together.

    Example Usage
    -------------
    >>> complexVarProc('z', 'x', 'y')
    Initializes the complex variable system (z, BARz, x, y).

    >>> complexVarProc('z', 'x', 'y', 2)
    Initializes the complex variable system tuple ((z1, z2), (BARz1, BARz2), (x1, x2), (y1, y2)).

    Conversion Dictionaries
    -----------------------
    The following dictionaries are updated when initializing the complex variable system:
    - holToReal : Maps holomorphic variables to their real and imaginary parts.
    - realToSym : Maps real variables to their symbolic holomorphic representations.
    - symToHol : Maps symbolic conjugates (BAR) to their holomorphic counterparts.
    - symToReal : Maps holomorphic variables to their real and imaginary parts.
    - realToHol : Maps real variables to their holomorphic counterparts.
    - find_parents : Tracks the relationship between real/imaginary variables and their holomorphic counterparts.
    
    Raises
    ------
    Exception
        If a variable already exists in the global namespace or if an invalid argument is provided.

    Notes
    -----
    This function integrates tightly with the internal variable_registry structure, ensuring that all complex variable systems are properly tracked and accessible for subsequent operations (such as vector fields and differential forms).
    """
    if default_var_format !='real' and default_var_format !='complex':
        if default_var_format != None:
            warnings.warn('`default_var_format` was set to an unsuported value, so it was reset to the default \'complex\'.')
        default_var_format='complex'
    if default_var_format == 'complex':
        _complexVarProc_default_to_hol(holom_label, real_label, im_label, number_of_variables=number_of_variables, initialIndex=initialIndex, remove_guardrails=remove_guardrails)
    elif default_var_format == 'real':
        variable_registry = get_variable_registry()

        conversion_dicts = variable_registry['conversion_dictionaries']
        find_parents = conversion_dicts['find_parents']

        def validate_variable_labels(*labels):
            """
            Checks if any of the provided variable labels start with 'BAR', reformats them to 'anti_', 
            and ensures no duplicate labels are provided. Also checks if the labels are protected globals.
            
            Parameters
            ----------
            labels : str
                Any number of strings representing variable labels to be validated.
                
            Returns
            -------
            tuple
                A tuple containing the reformatted labels.
            """
            reformatted_labels = []
            seen_labels = set()
            protectedGlobals=protected_caller_globals()

            for label in labels:
                # Check if the label is a protected global
                if  label in protectedGlobals:
                    raise ValueError(f"DGCV recognizes label '{label}' as a protected global name and recommends not using it as a variable name. Set remove_guardrails=True in the variable creation functions to force it.")


                # Check if the label starts with "BAR" and reformat if necessary
                if label.startswith("BAR"):
                    reformatted_label = "anti_" + label[3:]
                    warnings.warn(f"Label '{label}' starts with 'BAR', which has special meaning in DGCV. It has been automatically reformatted to '{reformatted_label}'.")
                else:
                    reformatted_label = label

                # Check for duplicate labels
                if reformatted_label in seen_labels:
                    raise ValueError(f"Duplicate label found: '{reformatted_label}'. Each label must be unique.")
                
                seen_labels.add(reformatted_label)
                reformatted_labels.append(reformatted_label)

            return tuple(reformatted_labels)

        # Convert to lists if single string arguments are provided
        if isinstance(holom_label, str):
            holom_label = [holom_label]
            real_label = [real_label]
            im_label = [im_label]

        for j in range(len(holom_label)):
            if remove_guardrails:
                labelLoc1 = holom_label[j] # Complex variable (e.g., z)
                labelLoc2 = real_label[j] # Real part (e.g., x)
                labelLoc3 = im_label[j] # Imaginary part (e.g., y)
            else:
                labelLoc1, labelLoc2, labelLoc3 = validate_variable_labels(holom_label[j], real_label[j], im_label[j]) # Complex variable, Real part, Imaginary part
            labelLocBAR = f'BAR{labelLoc1}'  # Antiholomorphic variable (e.g., BARz)

            # Clear existing variables
            clearVar(labelLoc1,report=False)
            clearVar(labelLoc2,report=False)
            clearVar(labelLoc3,report=False)
            clearVar(labelLocBAR,report=False)

            # Add real and imaginary part variables to protected_variables
            variable_registry['protected_variables'].update({labelLoc2, labelLoc3})

            # Define the complex family of variables
            complex_family = (labelLoc1, labelLocBAR, labelLoc2, labelLoc3)

            # Handle the lone variable system case (no args provided)
            if number_of_variables==None:
                # Create the variables
                variableProcedure(labelLoc1, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
                variableProcedure(labelLocBAR, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
                variableProcedure(labelLoc2, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())
                variableProcedure(labelLoc3, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())

                # Update find_parents
                find_parents[_cached_caller_globals[labelLoc2]] = (_cached_caller_globals[labelLoc1], _cached_caller_globals[labelLocBAR])
                find_parents[_cached_caller_globals[labelLoc3]] = (_cached_caller_globals[labelLoc1], _cached_caller_globals[labelLocBAR])

                # Update conversion dictionaries
                conversion_dicts['conjugation'][_cached_caller_globals[labelLoc1]] = _cached_caller_globals[labelLocBAR]
                conversion_dicts['conjugation'][_cached_caller_globals[labelLocBAR]] = _cached_caller_globals[labelLoc1]
                conversion_dicts['holToReal'][_cached_caller_globals[labelLoc1]] = _cached_caller_globals[labelLoc2] + I * _cached_caller_globals[labelLoc3]
                conversion_dicts['realToSym'][_cached_caller_globals[labelLoc2]] = Rational(1, 2) * (_cached_caller_globals[labelLoc1] + _cached_caller_globals[labelLocBAR])
                conversion_dicts['realToSym'][_cached_caller_globals[labelLoc3]] = -I * Rational(1, 2) * (_cached_caller_globals[labelLoc1] - _cached_caller_globals[labelLocBAR])
                conversion_dicts['symToHol'][_cached_caller_globals[labelLocBAR]] = conjugate(_cached_caller_globals[labelLoc1])
                conversion_dicts['symToReal'][_cached_caller_globals[labelLoc1]] = _cached_caller_globals[labelLoc2] + I * _cached_caller_globals[labelLoc3]
                conversion_dicts['symToReal'][_cached_caller_globals[labelLocBAR]] = _cached_caller_globals[labelLoc2] - I * _cached_caller_globals[labelLoc3]
                conversion_dicts['realToHol'][_cached_caller_globals[labelLoc2]] = Rational(1, 2) * (_cached_caller_globals[labelLoc1] + conjugate(_cached_caller_globals[labelLoc1]))
                conversion_dicts['realToHol'][_cached_caller_globals[labelLoc3]] = I * Rational(1, 2) * (conjugate(_cached_caller_globals[labelLoc1]) - _cached_caller_globals[labelLoc1])

                # Create holomorphic and antiholomorphic differential objects
                vf_instance_hol = VFClass((_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), [Rational(1,2),-I/2], DGCVType='complex')
                vf_instance_aHol = VFClass((_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), [Rational(1,2),+I/2], DGCVType='complex')
                df_instance_hol = DFClass((_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), {(0,): 1,(1,): I}, 1, DGCVType='complex')
                df_instance_aHol = DFClass((_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), {(0,): 1,(1,): -I}, 1, DGCVType='complex')
                _cached_caller_globals[f'D_{labelLoc1}'] = vf_instance_hol
                _cached_caller_globals[f'D_{labelLocBAR}'] = vf_instance_aHol
                _cached_caller_globals[f'd_{labelLoc1}'] = df_instance_hol
                _cached_caller_globals[f'd_{labelLocBAR}'] = df_instance_aHol

                # Create differential objects for real and imaginary coordinates
                _cached_caller_globals[f'D_{labelLoc2}'] = vf_instance_hol+vf_instance_aHol
                _cached_caller_globals[f'D_{labelLoc3}'] = I*(vf_instance_hol-vf_instance_aHol)
                _cached_caller_globals[f'd_{labelLoc2}'] = Rational(1,2)*(df_instance_hol+df_instance_aHol)
                _cached_caller_globals[f'd_{labelLoc3}'] = -I*Rational(1,2)*(df_instance_hol-df_instance_aHol)



                # Update complex_variable_systems for the complex variable and its parts
                variable_registry['complex_variable_systems'][labelLoc1] = {
                    'family_type': 'single',
                    'family_names': ((labelLoc1,), (labelLocBAR,), (labelLoc2,), (labelLoc3,)),
                    'family_values': (_cached_caller_globals[labelLoc1],
                                    _cached_caller_globals[labelLocBAR],
                                    _cached_caller_globals[labelLoc2],
                                    _cached_caller_globals[labelLoc3]),
                    'family_houses': (labelLoc1,labelLocBAR,labelLoc2,labelLoc3),
                    'differential_system': True,
                    'initial_index': None,
                    'variable_relatives': {
                        labelLoc1: {'complex_positioning': 'holomorphic', 'complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]),'variable_value' : _cached_caller_globals[labelLoc1], 'VFClass': _cached_caller_globals[f'D_{labelLoc1}'], 'DFClass': _cached_caller_globals[f'd_{labelLoc1}'], 'assumeReal': None},
                        labelLocBAR: {'complex_positioning': 'antiholomorphic', 'complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]),'variable_value' : _cached_caller_globals[labelLocBAR], 'VFClass': _cached_caller_globals[f'D_{labelLocBAR}'], 'DFClass': _cached_caller_globals[f'd_{labelLocBAR}'], 'assumeReal': None},
                        labelLoc2: {'complex_positioning': 'real','complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), 'variable_value' : _cached_caller_globals[labelLoc2], 'VFClass': _cached_caller_globals[f'D_{labelLoc2}'], 'DFClass': _cached_caller_globals[f'd_{labelLoc2}'], 'assumeReal': True},
                        labelLoc3: {'complex_positioning': 'imaginary','complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), 'variable_value' : _cached_caller_globals[labelLoc3], 'VFClass': _cached_caller_globals[f'D_{labelLoc3}'], 'DFClass': _cached_caller_globals[f'd_{labelLoc3}'], 'assumeReal': True}
                    }
                }


            # Handle the tuple system case (args provided)
            elif isinstance(number_of_variables,int) and number_of_variables>0:
                lengthLoc = number_of_variables

                # Create the variables
                variableProcedure(labelLoc1, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
                variableProcedure(labelLocBAR, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
                variableProcedure(labelLoc2, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())
                variableProcedure(labelLoc3, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())

                var_names1 = _cached_caller_globals[labelLoc1]  # Complex variables
                var_namesBAR = _cached_caller_globals[labelLocBAR]  # Antiholomorphic variables
                var_names2 = _cached_caller_globals[labelLoc2]  # Real parts
                var_names3 = _cached_caller_globals[labelLoc3]  # Imaginary parts

                # Create variable name strings for registry
                var_str1 = tuple([f"{labelLoc1}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])
                var_strBAR = tuple([f"{labelLocBAR}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])
                var_str2 = tuple([f"{labelLoc2}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])
                var_str3 = tuple([f"{labelLoc3}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])

                # Update complex_variable_systems for the holomorphic variable (as the parent)
                variable_registry['complex_variable_systems'][labelLoc1] = {
                    'family_type': 'tuple',
                    'family_names': (var_str1, var_strBAR, var_str2, var_str3),
                    'family_values': (var_names1, var_namesBAR, var_names2, var_names3),
                    'family_houses': (labelLoc1,labelLocBAR,labelLoc2,labelLoc3),
                    'differential_system': True,
                    'initial_index': initialIndex,
                    'variable_relatives': dict()
                }


                # Update complex_variable_systems for the child variables
                totalVarListLoc=tuple(zip(var_names1, var_namesBAR, var_names2, var_names3))
                
                for j in range(lengthLoc):
                    comp_var, bar_comp_var, real_var, imag_var = totalVarListLoc[j]
                    # Update find_parents for real and imaginary parts
                    find_parents[real_var] = (comp_var, bar_comp_var)
                    find_parents[imag_var] = (comp_var, bar_comp_var)  

                    # Update conversion dictionaries
                    conversion_dicts['conjugation'][comp_var] = bar_comp_var
                    conversion_dicts['conjugation'][bar_comp_var] = comp_var
                    conversion_dicts['holToReal'][comp_var] = real_var + I * imag_var
                    conversion_dicts['realToSym'][real_var] = Rational(1, 2) * (comp_var + bar_comp_var)
                    conversion_dicts['realToSym'][imag_var] = -I * Rational(1, 2) * (comp_var - bar_comp_var)
                    conversion_dicts['symToHol'][bar_comp_var] = conjugate(comp_var)
                    conversion_dicts['symToReal'][comp_var] = real_var + I * imag_var
                    conversion_dicts['symToReal'][bar_comp_var] = real_var - I * imag_var
                    conversion_dicts['realToHol'][real_var] = Rational(1, 2) * (comp_var + conjugate(comp_var))
                    conversion_dicts['realToHol'][imag_var] = I * Rational(1, 2) * (conjugate(comp_var) - comp_var)
                
                
                for j in range(lengthLoc):
                    comp_var, bar_comp_var, real_var, imag_var = totalVarListLoc[j]

                    # Create holomorphic and antiholomorphic vector fields and differential forms
                    _cached_caller_globals[f'D_{comp_var}'] = VFClass(var_names2 + var_names3, [Rational(1, 2) if i == j else 0 for i in range(lengthLoc)] + [-I/2 if i == j else 0 for i in range(lengthLoc)], 'complex')
                    _cached_caller_globals[f'D_{bar_comp_var}'] = VFClass(var_names2 + var_names3, [Rational(1, 2) if i == j else 0 for i in range(lengthLoc)] + [I/2 if i == j else 0 for i in range(lengthLoc)], 'complex')
                    _cached_caller_globals[f'd_{comp_var}'] = DFClass(var_names2 + var_names3, {(j,): 1, (j + len(var_names2),): I}, 1, 'complex')
                    _cached_caller_globals[f'd_{bar_comp_var}'] = DFClass(var_names2 + var_names3, {(j,): 1, (j + len(var_names2),): -I}, 1, 'complex')

                    # Create differential objects for real and imaginary coordinates
                    _cached_caller_globals[f'D_{real_var}'] = _cached_caller_globals[f'D_{comp_var}']+_cached_caller_globals[f'D_{bar_comp_var}']
                    _cached_caller_globals[f'D_{imag_var}'] = I*(_cached_caller_globals[f'D_{comp_var}']-_cached_caller_globals[f'D_{bar_comp_var}'])
                    _cached_caller_globals[f'd_{real_var}'] = Rational(1,2)*(_cached_caller_globals[f'd_{comp_var}']+_cached_caller_globals[f'd_{bar_comp_var}'])
                    _cached_caller_globals[f'd_{imag_var}'] = -I*Rational(1,2)*(_cached_caller_globals[f'd_{comp_var}']-_cached_caller_globals[f'd_{bar_comp_var}'])

                    variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(comp_var)] = {
                        'complex_positioning': 'holomorphic',
                        'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                        'variable_value' : comp_var,
                        'VFClass':  _cached_caller_globals[f'D_{comp_var}'], 
                        'DFClass':  _cached_caller_globals[f'd_{comp_var}'],
                        'assumeReal': None
                    }

                    variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(bar_comp_var)] = {
                        'complex_positioning': 'antiholomorphic',
                        'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                        'variable_value' : bar_comp_var,
                        'VFClass':  _cached_caller_globals[f'D_{bar_comp_var}'], 
                        'DFClass':  _cached_caller_globals[f'd_{bar_comp_var}'],
                        'assumeReal': None
                    }

                    variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(real_var)] = {
                        'complex_positioning': 'real',
                        'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                        'variable_value' : real_var,
                        'VFClass':  _cached_caller_globals[f'D_{real_var}'], 
                        'DFClass':  _cached_caller_globals[f'd_{real_var}'],
                        'assumeReal': True
                    }

                    variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(imag_var)] = {
                        'complex_positioning': 'imaginary',
                        'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                        'variable_value' : imag_var,                    
                        'VFClass':  _cached_caller_globals[f'D_{imag_var}'], 
                        'DFClass':  _cached_caller_globals[f'd_{imag_var}'],
                        'assumeReal': True
                    }

            else:
                raise ValueError('variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided.')
    else:
        raise KeyError('`default_var_format` key should only be set to \'real\' or \'complex\'.')

def _complexVarProc_default_to_hol(holom_label, real_label, im_label, number_of_variables=None, initialIndex=1, remove_guardrails=None):
    """
    Use `complexVarProc` with the default_to_hol=True` keyword instead. 
    
    This is one of two routines that excecutes when `complexVarProc` is called, this branch executes when `complexVarProc` is called with `default_to_hol=True`.
    """
    variable_registry = get_variable_registry()

    conversion_dicts = variable_registry['conversion_dictionaries']
    find_parents = conversion_dicts['find_parents']

    def validate_variable_labels(*labels):
        """
        Checks if any of the provided variable labels start with 'BAR', reformats them to 'anti_', 
        and ensures no duplicate labels are provided. Also checks if the labels are protected globals.
        
        Parameters
        ----------
        labels : str
            Any number of strings representing variable labels to be validated.
            
        Returns
        -------
        tuple
            A tuple containing the reformatted labels.
        """
        reformatted_labels = []
        seen_labels = set()
        protectedGlobals=protected_caller_globals()

        for label in labels:
            # Check if the label is a protected global
            if  label in protectedGlobals:
                raise ValueError(f"DGCV recognizes label '{label}' as a protected global name and recommends not using it as a variable name. Set remove_guardrails=True in the variable creation functions to force it.")


            # Check if the label starts with "BAR" and reformat if necessary
            if label.startswith("BAR"):
                reformatted_label = "anti_" + label[3:]
                warnings.warn(f"Label '{label}' starts with 'BAR', which has special meaning in DGCV. It has been automatically reformatted to '{reformatted_label}'.")
            else:
                reformatted_label = label

            # Check for duplicate labels
            if reformatted_label in seen_labels:
                raise ValueError(f"Duplicate label found: '{reformatted_label}'. Each label must be unique.")
            
            seen_labels.add(reformatted_label)
            reformatted_labels.append(reformatted_label)

        return tuple(reformatted_labels)

    # Convert to lists if single string arguments are provided
    if isinstance(holom_label, str):
        holom_label = [holom_label]
        real_label = [real_label]
        im_label = [im_label]

    for j in range(len(holom_label)):
        if remove_guardrails:
            labelLoc1 = holom_label[j] # Complex variable (e.g., z)
            labelLoc2 = real_label[j] # Real part (e.g., x)
            labelLoc3 = im_label[j] # Imaginary part (e.g., y)
        else:
            labelLoc1, labelLoc2, labelLoc3 = validate_variable_labels(holom_label[j], real_label[j], im_label[j]) # Complex variable, Real part, Imaginary part
        labelLocBAR = f'BAR{labelLoc1}'  # Antiholomorphic variable (e.g., BARz)

        # Clear existing variables
        clearVar(labelLoc1,report=False)
        clearVar(labelLoc2,report=False)
        clearVar(labelLoc3,report=False)
        clearVar(labelLocBAR,report=False)

        # Add real and imaginary part variables to protected_variables
        variable_registry['protected_variables'].update({labelLoc2, labelLoc3})

        # Define the complex family of variables
        complex_family = (labelLoc1, labelLocBAR, labelLoc2, labelLoc3)

        # Handle the lone variable system case (no args provided)
        if number_of_variables==None:
            # Create the variables
            variableProcedure(labelLoc1, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
            variableProcedure(labelLocBAR, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
            variableProcedure(labelLoc2, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())
            variableProcedure(labelLoc3, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())

            # Update find_parents
            find_parents[_cached_caller_globals[labelLoc2]] = (_cached_caller_globals[labelLoc1], _cached_caller_globals[labelLocBAR])
            find_parents[_cached_caller_globals[labelLoc3]] = (_cached_caller_globals[labelLoc1], _cached_caller_globals[labelLocBAR])

            # Update conversion dictionaries
            conversion_dicts['conjugation'][_cached_caller_globals[labelLoc1]] = _cached_caller_globals[labelLocBAR]
            conversion_dicts['conjugation'][_cached_caller_globals[labelLocBAR]] = _cached_caller_globals[labelLoc1]
            conversion_dicts['holToReal'][_cached_caller_globals[labelLoc1]] = _cached_caller_globals[labelLoc2] + I * _cached_caller_globals[labelLoc3]
            conversion_dicts['realToSym'][_cached_caller_globals[labelLoc2]] = Rational(1, 2) * (_cached_caller_globals[labelLoc1] + _cached_caller_globals[labelLocBAR])
            conversion_dicts['realToSym'][_cached_caller_globals[labelLoc3]] = -I * Rational(1, 2) * (_cached_caller_globals[labelLoc1] - _cached_caller_globals[labelLocBAR])
            conversion_dicts['symToHol'][_cached_caller_globals[labelLocBAR]] = conjugate(_cached_caller_globals[labelLoc1])
            conversion_dicts['symToReal'][_cached_caller_globals[labelLoc1]] = _cached_caller_globals[labelLoc2] + I * _cached_caller_globals[labelLoc3]
            conversion_dicts['symToReal'][_cached_caller_globals[labelLocBAR]] = _cached_caller_globals[labelLoc2] - I * _cached_caller_globals[labelLoc3]
            conversion_dicts['realToHol'][_cached_caller_globals[labelLoc2]] = Rational(1, 2) * (_cached_caller_globals[labelLoc1] + conjugate(_cached_caller_globals[labelLoc1]))
            conversion_dicts['realToHol'][_cached_caller_globals[labelLoc3]] = I * Rational(1, 2) * (conjugate(_cached_caller_globals[labelLoc1]) - _cached_caller_globals[labelLoc1])


            # Create holomorphic and antiholomorphic differential objects
            vf_instance_hol = VFClass((_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR]), [1,0], DGCVType='complex')
            vf_instance_aHol = VFClass((_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR]), [0,1], DGCVType='complex')
            df_instance_hol = DFClass((_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR]), {(0,): 1}, 1, DGCVType='complex')
            df_instance_aHol = DFClass((_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR]), {(1,): 1}, 1, DGCVType='complex')
            _cached_caller_globals[f'D_{labelLoc1}'] = vf_instance_hol
            _cached_caller_globals[f'D_{labelLocBAR}'] = vf_instance_aHol
            _cached_caller_globals[f'd_{labelLoc1}'] = df_instance_hol
            _cached_caller_globals[f'd_{labelLocBAR}'] = df_instance_aHol

            # Create differential objects for real and imaginary coordinates
            _cached_caller_globals[f'D_{labelLoc2}'] = vf_instance_hol+vf_instance_aHol
            _cached_caller_globals[f'D_{labelLoc3}'] = I*(vf_instance_hol-vf_instance_aHol)
            _cached_caller_globals[f'd_{labelLoc2}'] = Rational(1,2)*(df_instance_hol+df_instance_aHol)
            _cached_caller_globals[f'd_{labelLoc3}'] = -I*Rational(1,2)*(df_instance_hol-df_instance_aHol)

            # Update complex_variable_systems for the complex variable and its parts
            variable_registry['complex_variable_systems'][labelLoc1] = {
                'family_type': 'single',
                'family_names': ((labelLoc1,), (labelLocBAR,), (labelLoc2,), (labelLoc3,)),
                'family_values': (_cached_caller_globals[labelLoc1],
                                  _cached_caller_globals[labelLocBAR],
                                  _cached_caller_globals[labelLoc2],
                                  _cached_caller_globals[labelLoc3]),
                'family_houses': (labelLoc1,labelLocBAR,labelLoc2,labelLoc3),
                'differential_system': True,
                'initial_index': None,
                'variable_relatives': {
                    labelLoc1: {'complex_positioning': 'holomorphic', 'complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]),'variable_value' : _cached_caller_globals[labelLoc1], 'VFClass': _cached_caller_globals[f'D_{labelLoc1}'], 'DFClass': _cached_caller_globals[f'd_{labelLoc1}'], 'assumeReal': None},
                    labelLocBAR: {'complex_positioning': 'antiholomorphic', 'complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), 'variable_value' : _cached_caller_globals[labelLocBAR], 'VFClass': _cached_caller_globals[f'D_{labelLocBAR}'], 'DFClass': _cached_caller_globals[f'd_{labelLocBAR}'], 'assumeReal': None},
                    labelLoc2: {'complex_positioning': 'real', 'complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), 'variable_value' : _cached_caller_globals[labelLoc2], 'VFClass': _cached_caller_globals[f'D_{labelLoc2}'], 'DFClass': _cached_caller_globals[f'd_{labelLoc2}'], 'assumeReal': True},
                    labelLoc3: {'complex_positioning': 'imaginary', 'complex_family': (_cached_caller_globals[labelLoc1],_cached_caller_globals[labelLocBAR],_cached_caller_globals[labelLoc2],_cached_caller_globals[labelLoc3]), 'variable_value' : _cached_caller_globals[labelLoc3], 'VFClass': _cached_caller_globals[f'D_{labelLoc3}'], 'DFClass': _cached_caller_globals[f'd_{labelLoc3}'], 'assumeReal': True}
                }
            }



        # Handle the tuple system case (args provided)
        elif isinstance(number_of_variables,int) and number_of_variables>0:
            lengthLoc = number_of_variables

            # Create the variables
            variableProcedure(labelLoc1, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
            variableProcedure(labelLocBAR, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), _calledFromCVP=retrieve_passkey())
            variableProcedure(labelLoc2, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())
            variableProcedure(labelLoc3, lengthLoc, initialIndex=initialIndex, _doNotUpdateVar=retrieve_passkey(), assumeReal=True, _calledFromCVP=retrieve_passkey())

            var_names1 = _cached_caller_globals[labelLoc1]  # Complex variables
            var_namesBAR = _cached_caller_globals[labelLocBAR]  # Antiholomorphic variables
            var_names2 = _cached_caller_globals[labelLoc2]  # Real parts
            var_names3 = _cached_caller_globals[labelLoc3]  # Imaginary parts

            # Create variable name strings for registry
            var_str1 = tuple([f"{labelLoc1}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])
            var_strBAR = tuple([f"{labelLocBAR}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])
            var_str2 = tuple([f"{labelLoc2}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])
            var_str3 = tuple([f"{labelLoc3}{i}" for i in range(initialIndex, lengthLoc + initialIndex)])

            # Update complex_variable_systems for the holomorphic variable (as the parent)
            variable_registry['complex_variable_systems'][labelLoc1] = {
                'family_type': 'tuple',
                'family_names': (var_str1, var_strBAR, var_str2, var_str3),
                'family_values': (var_names1, var_namesBAR, var_names2, var_names3),
                'family_houses': (labelLoc1,labelLocBAR,labelLoc2,labelLoc3),
                'differential_system': True,
                'initial_index': initialIndex,
                'variable_relatives': dict()
            }


            # Update complex_variable_systems for the child variables
            totalVarListLoc=tuple(zip(var_names1, var_namesBAR, var_names2, var_names3))
            
            for j in range(lengthLoc):
                comp_var, bar_comp_var, real_var, imag_var = totalVarListLoc[j]
                # Update find_parents for real and imaginary parts
                find_parents[real_var] = (comp_var, bar_comp_var)
                find_parents[imag_var] = (comp_var, bar_comp_var)  

                # Update conversion dictionaries
                conversion_dicts['conjugation'][comp_var] = bar_comp_var
                conversion_dicts['conjugation'][bar_comp_var] = comp_var
                conversion_dicts['holToReal'][comp_var] = real_var + I * imag_var
                conversion_dicts['realToSym'][real_var] = Rational(1, 2) * (comp_var + bar_comp_var)
                conversion_dicts['realToSym'][imag_var] = -I * Rational(1, 2) * (comp_var - bar_comp_var)
                conversion_dicts['symToHol'][bar_comp_var] = conjugate(comp_var)
                conversion_dicts['symToReal'][comp_var] = real_var + I * imag_var
                conversion_dicts['symToReal'][bar_comp_var] = real_var - I * imag_var
                conversion_dicts['realToHol'][real_var] = Rational(1, 2) * (comp_var + conjugate(comp_var))
                conversion_dicts['realToHol'][imag_var] = I * Rational(1, 2) * (conjugate(comp_var) - comp_var)
            
            for j in range(lengthLoc):
                comp_var, bar_comp_var, real_var, imag_var = totalVarListLoc[j]

                # Create holomorphic and antiholomorphic vector fields and differential forms
                _cached_caller_globals[f'D_{comp_var}'] = VFClass(var_names1 + var_namesBAR, [1 if i == j else 0 for i in range(2*lengthLoc)], 'complex')
                _cached_caller_globals[f'D_{bar_comp_var}'] = VFClass(var_names1 + var_namesBAR, [0 for i in range(lengthLoc)] + [1 if i == j else 0 for i in range(lengthLoc)], 'complex')
                _cached_caller_globals[f'd_{comp_var}'] = DFClass(var_names1 + var_namesBAR, {(j,): 1}, 1, 'complex')
                _cached_caller_globals[f'd_{bar_comp_var}'] = DFClass(var_names1 + var_namesBAR, {(j + len(var_names2),): 1}, 1, 'complex')

                # Create differential objects for real and imaginary coordinates
                _cached_caller_globals[f'D_{real_var}'] = _cached_caller_globals[f'D_{comp_var}']+_cached_caller_globals[f'D_{bar_comp_var}']
                _cached_caller_globals[f'D_{imag_var}'] = I*(_cached_caller_globals[f'D_{comp_var}']-_cached_caller_globals[f'D_{bar_comp_var}'])
                _cached_caller_globals[f'd_{real_var}'] = Rational(1,2)*(_cached_caller_globals[f'd_{comp_var}']+_cached_caller_globals[f'd_{bar_comp_var}'])
                _cached_caller_globals[f'd_{imag_var}'] = -I*Rational(1,2)*(_cached_caller_globals[f'd_{comp_var}']-_cached_caller_globals[f'd_{bar_comp_var}'])


                variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(comp_var)] = {
                    'complex_positioning': 'holomorphic',
                    'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                    'variable_value' : comp_var,
                    'VFClass':  _cached_caller_globals[f'D_{comp_var}'], 
                    'DFClass':  _cached_caller_globals[f'd_{comp_var}'],
                    'assumeReal': None
                }

                variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(bar_comp_var)] = {
                    'complex_positioning': 'antiholomorphic',
                    'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                    'variable_value' : bar_comp_var,
                    'VFClass':  _cached_caller_globals[f'D_{bar_comp_var}'], 
                    'DFClass':  _cached_caller_globals[f'd_{bar_comp_var}'],
                    'assumeReal': None
                }

                variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(real_var)] = {
                    'complex_positioning': 'real',
                    'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                    'variable_value' : real_var,
                    'VFClass':  _cached_caller_globals[f'D_{real_var}'], 
                    'DFClass':  _cached_caller_globals[f'd_{real_var}'],
                    'assumeReal': True
                }

                variable_registry['complex_variable_systems'][labelLoc1]['variable_relatives'][str(imag_var)] = {
                    'complex_positioning': 'imaginary',
                    'complex_family': (comp_var, bar_comp_var, real_var, imag_var),
                    'variable_value' : imag_var,                    
                    'VFClass':  _cached_caller_globals[f'D_{imag_var}'], 
                    'DFClass':  _cached_caller_globals[f'd_{imag_var}'],
                    'assumeReal': True
                }

        else:
            raise ValueError('variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided.')

def _format_complex_coordinates(coordinate_tuple, default_var_format='complex', pass_error_report=None):
    """
    Format var. lists consisting of variables within DGCV complex variable systems, formatting as real or holomorphic and completeing the basis as needed (i.e., adding BARz if only z is present, adding y if only x, etc.)
    """
    vr=get_variable_registry()
    exaustList=list(coordinate_tuple)
    newList1=[]
    newList2=[]
    try:
        for var in coordinate_tuple:
            if var in exaustList:
                varStr = str(var)
                for parent in vr['complex_variable_systems']:
                    if varStr in vr['complex_variable_systems'][parent]['variable_relatives']:
                        foundVars = vr['complex_variable_systems'][parent]['variable_relatives'][varStr]['complex_family']
                        if default_var_format == 'complex':
                            newList1 = newList1 + [foundVars[0]]
                            newList2 = newList2 + [foundVars[1]]
                        else:
                            newList1 = newList1 + [foundVars[2]]
                            newList2 = newList2 + [foundVars[3]]
                        for j in foundVars:
                            if j in exaustList:
                                exaustList.remove(j)
    except:
        if pass_error_report==retrieve_passkey():
            return 'At least one element in the given variable list is not registered as part of a complex variable system in the DGCV variable management framework.'
    return tuple(newList1+newList2)




############## variable format conversion

def _VFDF_conversion(obj, default_var_format=None, _converter=None):
    def converter(expr, _conv):
        if _conv == None:
            return expr
        return _conv(expr)
    
    if default_var_format == 'complex':
        if isinstance(obj,VFClass):
            if obj.DGCVType == 'standard':
                return VFClass(obj.varSpace, [converter(j,_converter) for j in coeffs], DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
            varSpace = obj.compVarSpace
            coeffsDict = obj.coeff_dicts['holCoeffs'] | obj.coeff_dicts['antiholCoeffs']
            coeffs = [converter(coeffsDict[a],_converter) for a in varSpace]
            return VFClass(varSpace, coeffs, DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
        elif isinstance(obj,DFClass):
            specialCase = obj.DGCVType=='complex' and obj._varSpace_type=='complex'
            if obj.DGCVType == 'standard' or specialCase:
                return DFClass(obj.varSpace, {a:converter(b,_converter) for a,b in obj.DFClassDataDict.items()}, obj.degree,DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
            varSpace = obj.coeff_dicts['compCoeffDataDict'][0]
            dataDict = obj.coeff_dicts['compCoeffDataDict'][1]
            dataDict = {a:converter(b,_converter) for a,b in dataDict.items()}
            return DFClass(varSpace,dataDict,obj.degree, DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
    elif default_var_format == 'real':
        if isinstance(obj,VFClass):
            if obj.DGCVType == 'standard':
                return VFClass(obj.varSpace, [converter(a, _converter) for a in obj.coeffs], DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
            varSpace = obj.realVarSpace
            coeffsDict = obj.coeff_dicts['realCoeffs'] | obj.coeff_dicts['imCoeffs']
            coeffs = [converter(coeffsDict[a], _converter) for a in varSpace]
            return VFClass(varSpace, coeffs, DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
        elif isinstance(obj,DFClass):
            specialCase = obj.DGCVType=='complex' and obj._varSpace_type=='real'
            if obj.DGCVType == 'standard' or specialCase:
                return DFClass(obj.varSpace, {a:converter(b,_converter) for a,b in obj.DFClassDataDict.items()}, obj.degree,DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)
            varSpace = obj.coeff_dicts['realCoeffDataDict'][0]
            dataDict = obj.coeff_dicts['realCoeffDataDict'][1]
            dataDict = {a:converter(b,_converter) for a,b in dataDict.items()}
            return DFClass(varSpace,dataDict,obj.degree, DGCVType=obj.DGCVType, simplifyKW=obj.simplifyKW)

def holToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts holomorphic variables in the expression to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of holomorphic variable system labels to skip during conversion. 
        For any variable in skipVar, the associated holomorphic variables will 
        not be substituted.

    Returns:
    sympy.Expr
        The expression with holomorphic variables replaced by real variables, 
        except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    conversion_dict = variable_registry.get('conversion_dictionaries', {}).get('holToReal', {}).copy()
    def format(expr):
        return sympify(expr).subs(conversion_dict)
    # Skip specified holomorphic variable systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get('complex_variable_systems', {}).get(var, {})
            family_values = complex_system.get('family_values', ((), (), (), ()))
            if complex_system.get('family_type', None)=='single':
                family_values = tuple([(j,) for j in family_values])
            
            # Remove the holomorphic variables (first tuple) from the conversion_dict
            holomorphic_vars = family_values[0]
            for hol_var in holomorphic_vars:
                if hol_var in conversion_dict:
                    del conversion_dict[hol_var]

    if isinstance(expr,(VFClass,DFClass)) and simplify_everything:
        return _VFDF_conversion(expr,default_var_format='real',_converter=holToReal)
    elif hasattr(expr, 'applyfunc'):
        return expr.applyfunc(format)
    else:
        return format(expr)

def realToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts real variables in the expression to symbolic conjugates.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of real variable system labels to skip during conversion. 
        For any variable in skipVar, the associated real and imaginary variables 
        will not be substituted.

    Returns:
    sympy.Expr
        The expression with real variables replaced by symbolic conjugates, 
        except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    conversion_dict = variable_registry.get('conversion_dictionaries', {}).get('realToSym', {}).copy()
    def format(expr):
        return sympify(expr).subs(conversion_dict)
        
    # Skip specified real and imaginary variable systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get('complex_variable_systems', {}).get(var, {})
            family_values = complex_system.get('family_values', ((), (), (), ()))
            if complex_system.get('family_type', None)=='single':
                family_values = tuple([(j,) for j in family_values])

            # Remove both the real (third tuple) and imaginary (fourth tuple) variables from the conversion_dict
            real_vars = family_values[2] + family_values[3]
            for real_var in real_vars:
                if real_var in conversion_dict:
                    del conversion_dict[real_var]
    
    if isinstance(expr,(VFClass,DFClass)) and simplify_everything:
        return _VFDF_conversion(expr,default_var_format='complex',_converter=realToSym)
    elif hasattr(expr, 'applyfunc'):
        return expr.applyfunc(format)
    else:
        return format(expr)

def symToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts symbolic conjugated variables in the expression to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of holomorphic variable system labels to skip during conversion. 
        For any variable system labels in skipVar, the associated antiholomorphic variables 
        (i.e., conjugates) will not be substituted.

    Returns:
    sympy.Expr
        The expression with symbolic conjugates replaced by holomorphic variables, 
        except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    conversion_dict = variable_registry.get('conversion_dictionaries', {}).get('symToHol', {}).copy()
    def format(expr):
        return sympify(expr).subs(conversion_dict)
    
    # If skipVar is provided, modify the conversion_dict to exclude specified variables.
    if skipVar:
        for var in skipVar:
            # Access the complex variable system for the skipped variable
            complex_system = variable_registry.get('complex_variable_systems', {}).get(var, {})
            family_values = complex_system.get('family_values', ((), (), (), ()))
            if complex_system.get('family_type', None)=='single':
                family_values = tuple([(j,) for j in family_values])
                            
            # The second tuple contains the antiholomorphic variables
            antiholomorphic_vars = family_values[1]
            
            # Remove the antiholomorphic variables from the conversion_dict
            for anti_var in antiholomorphic_vars:
                if anti_var in conversion_dict:
                    del conversion_dict[anti_var]
    
    if isinstance(expr,(VFClass,DFClass)) and simplify_everything:
        return _VFDF_conversion(expr,default_var_format='complex',_converter=symToHol)
    elif hasattr(expr, 'applyfunc'):
        return expr.applyfunc(format)
    else:
        return format(expr)

def holToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts holomorphic variables in the expression to symbolic conjugates. 
    This is done by first converting holomorphic variables to real variables, 
    and then converting real variables to symbolic conjugates.

    Note: This process will also convert any present real variables 
    (both real and imaginary parts) to their symbolic conjugate format.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of holomorphic variable system labels to skip during conversion. 
        For any variable in skipVar, the associated holomorphic variables and 
        their real counterparts will not be substituted.

    Returns:
    sympy.Expr
        The expression with holomorphic variables (and any present real variables) 
        replaced by symbolic conjugates, except for those specified in skipVar.
    """
    # First apply holToReal()
    expr = holToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)
    
    # Then apply realToSym()
    expr = realToSym(expr, skipVar=skipVar,simplify_everything=simplify_everything)
    
    return expr

def cleanUpConjugation(arg1,skipVar=None):
    """
    While working with symbolic conjugate expressions it can easily happen that sympy generates expressions having actual conjugation applied to symbolic conjugate variables. Apply this command to simplify such redundancies.

    This is often necessary for *solve* functions to succeed.

    Args:
        arg1: sympy expression
    
    Returns:
        sympy expression
    
    Raises:
        NA
    """
    return realToSym(symToReal(arg1,skipVar=skipVar, simplify_everything=False),skipVar=skipVar, simplify_everything=False)

def realToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts real variables in the expression to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of real variable system labels to skip during conversion. 
        For any variable in skipVar, the associated real and imaginary variables 
        will not be substituted.

    Returns:
    sympy.Expr
        The expression with real variables replaced by holomorphic variables, 
        except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    conversion_dict = variable_registry.get('conversion_dictionaries', {}).get('realToHol', {}).copy()
    def format(expr):
        return sympify(expr).subs(conversion_dict)
    
    # Skip specified real and imaginary variable systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get('complex_variable_systems', {}).get(var, {})
            family_values = complex_system.get('family_values', ((), (), (), ()))
            if complex_system.get('family_type', None)=='single':
                family_values = tuple([(j,) for j in family_values])

            # Remove both the real (third tuple) and imaginary (fourth tuple) variables from the conversion_dict
            real_vars = family_values[2] + family_values[3]
            for real_var in real_vars:
                if real_var in conversion_dict:
                    del conversion_dict[real_var]
    
    if isinstance(expr,(VFClass,DFClass)) and simplify_everything:
        return _VFDF_conversion(expr,default_var_format='complex',_converter=realToHol)
    elif hasattr(expr, 'applyfunc'):
        return expr.applyfunc(format)
    else:
        return format(expr)

def symToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts symbolic conjugates in the expression to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of symbolic conjugate system labels to skip during conversion. 
        For any variable in skipVar, the associated symbolic conjugates will 
        not be substituted.

    Returns:
    sympy.Expr
        The expression with symbolic conjugates replaced by real variables, 
        except for those specified in skipVar.
    """
    variable_registry = get_variable_registry()
    conversion_dict = variable_registry.get('conversion_dictionaries', {}).get('symToReal', {}).copy()
    def format(expr):
        return sympify(expr).subs(conversion_dict)
        
    # Skip specified symbolic conjugate systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get('complex_variable_systems', {}).get(var, {})
            family_values = complex_system.get('family_values', ((), (), (), ()))
            if complex_system.get('family_type', None)=='single':
                family_values = tuple([(j,) for j in family_values])

            # Remove the symbolic conjugates (second tuple) from the conversion_dict
            antiholomorphic_vars = family_values[1]
            for anti_var in antiholomorphic_vars:
                if anti_var in conversion_dict:
                    del conversion_dict[anti_var]
    
    if isinstance(expr,(VFClass,DFClass)) and simplify_everything:
        return _VFDF_conversion(expr,default_var_format='real',_converter=symToReal)
    elif hasattr(expr, 'applyfunc'):
        return expr.applyfunc(format)
    else:
        return format(expr)

def allToReal(expr, skipVar=None, simplify_everything=True):
    """
    ⚠️ ... not to be confused with "all too real"! 😊 

    Converts all variables (holomorphic, symbolic conjugates, and real) to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
        The expression with all variables converted to real variables, 
        except for those specified in skipVar.
    """
    # First convert symbolic conjugates to real variables
    expr = symToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)
    
    # Then convert holomorphic variables to real variables
    expr = holToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)
    
    return expr

def allToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables (real, symbolic conjugates, and holomorphic) to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
        The expression with all variables converted to holomorphic variables, 
        except for those specified in skipVar.
    """

    if isinstance(expr, list):
        return [allToHol(j) for j in expr]
    if isinstance(expr, tuple):
        return [allToHol(j) for j in expr]

    # First convert symbolic conjugates to real variables
    expr = symToHol(expr, skipVar=skipVar, simplify_everything=simplify_everything)
    
    # Then convert real variables to holomorphic variables
    expr = realToHol(expr, skipVar=skipVar, simplify_everything=simplify_everything)
    
    return expr

def allToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables (holomorphic, real, and symbolic conjugates) to symbolic conjugates.

    Note: This process will also convert any present real variables 
    (both real and imaginary parts) to their symbolic conjugate format.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
        The expression with all variables converted to symbolic conjugates, 
        except for those specified in skipVar.
    """
    return holToSym(expr, skipVar=skipVar, simplify_everything=simplify_everything)

def _remove_complex_handling(arg):
    if isinstance(arg,VFClass):
        return VFClass(arg.varSpace,arg.coeffs)
    if isinstance(arg,DFClass):
        return DFClass(arg.varSpace,arg.DFClassDataDict,arg.degree)

def complex_struct_op(vf):
    if isinstance(vf,VFClass):
        if vf.DGCVType=='standard':
            return vf
        elif vf._varSpace_type=='real':
            newVarSpace=vf.realVarSpace
            CDim=len(vf.holVarSpace)
            coeffs1=[]
            coeffs2=[]
            for j in range(CDim):
                yVal=newVarSpace[CDim+j]
                xVal=newVarSpace[j]
                coeffs1=coeffs1+[-vf.coeff_dicts['imCoeffs'][yVal]]
                coeffs2=coeffs2+[vf.coeff_dicts['realCoeffs'][xVal]]
            return VFClass(newVarSpace,coeffs1+coeffs2,DGCVType='complex',simplifyKW=vf.simplifyKW)
        elif vf._varSpace_type=='complex':
            newVarSpace=vf.compVarSpace
            CDim=len(vf.holVarSpace)
            coeffs1=[]
            coeffs2=[]
            for j in range(CDim):
                zVal=newVarSpace[j]
                BARzVal=newVarSpace[CDim+j]
                coeffs1=coeffs1+[I*vf.coeff_dicts['holCoeffs'][zVal]]
                coeffs2=coeffs2+[-I*vf.coeff_dicts['antiholCoeffs'][BARzVal]]
            return VFClass(newVarSpace,coeffs1+coeffs2,DGCVType='complex',simplifyKW=vf.simplifyKW)

def conjugate_DGCV(expr):
    if isinstance(expr,(VFClass,DFClass)):
        return _conjComplexVFDF(expr)
    else:
        return conjugate(expr)

def conj_with_real_coor(expr):
    return allToReal(expr).subs({I:-I})

def re_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(Rational(1,2)*(expr+conj_with_real_coor(expr)))
    return s

def im_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(-I*Rational(1,2)*(expr-conj_with_real_coor(expr)))
    return s

def conj_with_hol_coor(expr):
    vr = get_variable_registry()
    subsDictA = vr['conversion_dictionaries']['conjugation']
    subsDict = subsDictA | {I:-I}
    return allToSym(expr).subs(subsDict, simultaneous=True)

def re_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(Rational(1,2)*(expr+conj_with_hol_coor(expr)))
    return s

def im_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(-I*Rational(1,2)*(expr-conj_with_hol_coor(expr)))
    return s


############## basic vector field and differential forms operations

def VF_coeffs_direct(vf, var_space, sparse=False):
    """
    Computes the coefficients of a vector field `vf` when expressed in terms of the coordinate vector fields
    corresponding to the variables in `var_space`. The result is either a full list of coefficients or a 
    sparse list, depending on the `sparse` argument.

    This function applies the vector field to each element in `var_space`, returning the associated coefficient
    for each direction. In the sparse case, only non-zero coefficients are returned.

    Parameters
    ----------
    vf : VFClass
        A vector field object defined in the DGCV system.
    
    var_space : list or tuple
        A list or tuple of SymPy Symbol objects representing the variable space on which the vector field is defined.
    
    sparse : bool, optional
        If True, returns a sparse list containing only the non-zero coefficients, with each entry represented as a 
        tuple of the index and the coefficient. If False (default), returns the full list of coefficients, including 
        zeros.

    Returns
    -------
    list
        A list of SymPy expressions representing the coefficients of the vector field along each direction in `var_space`. 
        If `sparse=True`, returns a sparse list where each entry is a tuple of the form `((index,), coefficient)` 
        for non-zero coefficients, or `[((0,), 0)]` if all coefficients are zero.

    Raises
    ------
    TypeError
        If `vf` is not an instance of `VFClass` or if `var_space` is not a list or tuple of SymPy symbols.

    Examples
    --------
    >>> from sympy import symbols
    >>> from DGCV import VFClass, VF_coeffs_direct
    >>> x, y, z = symbols('x y z')
    >>> vf = VFClass([x, y, z], [y - z, z - x, x - y])
    >>> VF_coeffs_direct(vf, [x, y, z])
    [y - z, z - x, x - y]
    
    # Sparse example with non-zero coefficients only
    >>> VF_coeffs_direct(vf, [x, y, z], sparse=True)
    [((0,), y - z), ((1,), z - x), ((2,), x - y)]
    """
    # Ensure that vf is an instance of VFClass
    if not isinstance(vf, VFClass):
        raise TypeError('Expected first argument to be an instance of VFClass')
    
    # Ensure var_space is a list or tuple
    if not isinstance(var_space, (list, tuple)):
        raise TypeError('Expected second argument to be a list or tuple of variables')

    # Evaluate the vector field on each element in var_space
    coeffs = [vf(var) for var in var_space]
    
    # Return sparse or full result
    if sparse:
        return [((i,), coeffs[i]) for i in range(len(coeffs)) if coeffs[i] != 0] or [((0,), 0)]
    return coeffs

def minimalVFDataDict(vf):
    if isinstance(vf,VFClass):
        return {vf.varSpace[j]:vf.coeffs[j] for j in range(len(vf.varSpace)) if vf.coeffs[j]!=0}

def minimalDFDataDict(df):
    varTuple=df.varSpace
    list_collection=df.DFClassDataMinimal
    # Step 1: Find all integers missing from the lists in list_collection
    full_range = set(range(len(varTuple)))
    found_indices = set()

    for lst, _ in list_collection:
        found_indices.update(lst)

    missing_indices = full_range - found_indices

    # Step 2: Create new_varTuple by removing elements at missing indices
    new_varTuple = tuple(varTuple[i] for i in range(len(varTuple)) if i not in missing_indices)

    # Step 3: Create a mapping of old indices to new indices
    index_mapping = {}
    new_index = 0
    for old_index in range(len(varTuple)):
        if old_index not in missing_indices:
            index_mapping[old_index] = new_index
            new_index += 1

    # Step 4: Reformat the lists in list_collection to use new indices
    new_DFDataDict = dict()
    for lst, val in list_collection:
        new_inex = tuple([index_mapping[old_index] for old_index in lst])
        new_DFDataDict[new_inex] = val

    return new_varTuple, new_DFDataDict

def compressDGCVClass(obj):
    """Removes superfluous variables from the variable space that a VFClass or DFClass object is defined w.r.t."""
    if isinstance(obj, DFClass):
        newVarSpace,newDFData = minimalDFDataDict(obj)
        return DFClass(newVarSpace,newDFData,obj.degree,DGCVType=obj.DGCVType,simplifyKW=obj.simplifyKW)
    elif isinstance(obj,VFClass):
        VFData=minimalVFDataDict(obj)
        newVarSpace = list(VFData.keys())
        newCoeffs = [VFData[j] for j in newVarSpace]
        return VFClass(newVarSpace,newCoeffs,DGCVType=obj.DGCVType,simplifyKW=obj.simplifyKW)

def VF_coeffs(arg1,arg2):
    """
     Expresses coefficients of the vector feild *arg1* with respect to the list of variables in *arg2* assuming *arg2* contains at least all of the basis variables found in the the *varSpace* attribute of *arg1*. It also works if *arg2* merely contains the minimum set of such variables needed to define the vector field, and returns an error if not.
        
    Args:
        arg1: a vector field
        arg2: a list or tuple containing Symbol objects
    
    Returns:
        List of sympy expressions.
    
    Raises:
        NA
    """
    variable_registry=get_variable_registry()
    if isinstance(arg1,VFClass):
        if arg1.DGCVType=='complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in arg2):
                if arg1._varSpace_type == 'complex':
                    arg1=allToReal(arg1)
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in arg2):
                if arg1._varSpace_type == 'real':
                    arg1=allToSym(arg1)
            else:
                raise TypeError('The VFClass `VF_coeffs` was given has `DGCVType=\'complex\'` while the variable list given to decompose the VF w.r.t. were not either simultaneously among real coordinates or simulaneously among holomorphic (and antiholomorphic) coordinates. Variables should be within just one coordinate format.')
    else:
        raise TypeError('The first argument in `VF_coeffs` has to be VFClass type.')
    MVFD=minimalVFDataDict(arg1)
    return [0 if j not in MVFD else MVFD[j] for j in arg2]

def changeVFBasis(arg1,arg2):
    return VFClass(arg2,VF_coeffs(arg1,arg2),arg1.DGCVType)

def addVF(*args):
    """
    Adds the given vector fields (i.e., VFClass instances).
    
    Args:
        *args: any number of VFClass class instances
    
    Returns:
        A VFClass instance

    Raises:
        Exception: If any argument is not an instance of VFClass
    """
    if len(args) == 0:
        return args
    if all([isinstance(j,VFClass) for j in args]):
        typeList = list(set([j.DGCVType for j in args]))
        if len(typeList) == 1 and typeList[0] == 'complex':
            typeLoc = 'complex'
        else:
            typeLoc = 'standard'

        # Initialize an empty list for varSpace and extend it with each j.varSpace
        varSpaceLoc = []
        for j in args:
            varSpaceLoc.extend(j.varSpace)
        
        # Convert to a tuple with unique entries using dict.fromkeys
        varSpaceLoc = tuple(dict.fromkeys(varSpaceLoc))

        coeffListLoc = [VF_coeffs_direct(j, varSpaceLoc) for j in args]
        coeffsLoc = [sum(j) for j in zip(*coeffListLoc)]
        
        return VFClass(varSpaceLoc, coeffsLoc, typeLoc)
    else:
        raise Exception('Expected all arguments to be instances of the VFClass class.')

def scaleVF(scalar, vector_field):
    """
    Scales the given vector field (i.e., VFClass instance) *vector_field* by the scalar function (i.e., SymPy expression) *scalar*.
    
    Args:
        scalar: A SymPy expression representing the scalar multiplier.
        vector_field: A VFClass instance representing the vector field to scale.
    
    Returns:
        A new VFClass instance representing the scaled vector field.
    
    Raises:
        TypeError: If the second argument is not an instance of the VFClass class.
    """
    if isinstance(vector_field, VFClass):
        new_coeffs = [scalar * coeff for coeff in vector_field.coeffs]
        return VFClass(vector_field.varSpace, new_coeffs, vector_field.DGCVType)
    else:
        raise TypeError('Expected second argument to be an instance of the VFClass class.')

def VF_bracket(arg1, arg2, doNotSimplify=False, fast_algorithm=True):
    """
    Computes the Lie bracket of two vector fields (i.e., VFClass instances) *arg1* and *arg2* that are 
    defined in the same or related variable space(s). The function supports both standard and complex 
    vector fields and offers an optimized fast algorithm for standard cases.

    Parameters:
    -----------
    arg1 : VFClass
        The first vector field to compute the Lie bracket.
    arg2 : VFClass
        The second vector field to compute the Lie bracket.
    doNotSimplify : bool, optional
        If True, the result is returned without simplification (default is False).
    fast_algorithm : bool, optional
        If True, the function uses a faster algorithm that converts all coefficients to real variables 
        for vector fields, reducing overhead but sacrificing variable format fidelity (default is True).

    Returns:
    --------
    VFClass
        A new vector field instance representing the Lie bracket of *arg1* and *arg2*.

    Raises:
    -------
    Exception
        If either *arg1* or *arg2* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import VF_bracket, complexVarProc
    >>> complexVarProc('z','x','y',3)
    >>> print(VF_bracket(x2*D_x1 - x1*D_x2, x3*D_x1 - x1*D_x3))
    x3*D_x2 - x2*D_x3

    >>> print(VF_bracket(z2*D_z1 - z1*D_z2, z3*D_z1 - z1*D_z3))
    z3/2*D_x2 - z2/2*D_x3 - I*z3/2*D_y2 + I*z2/2*D_y3
    """
    # Ensure both arguments are instances of VFClass
    if {arg1.__class__.__name__, arg2.__class__.__name__} == {'VFClass'}:
        # Determine the type (standard/complex) based on DGCVType attribute
        if arg1.DGCVType == 'complex' or arg2.DGCVType == 'complex':
            typeLoc = 'standard' if (arg1.DGCVType != 'complex' or arg2.DGCVType != 'complex') else 'complex'
            fast_handling = False
        else:
            typeLoc = 'standard'
            fast_handling = True

        # Get the variable spaces, ensuring no duplicates if different
        varSpaceLoc = arg1.varSpace if arg1.varSpace == arg2.varSpace else tuple(dict.fromkeys(arg1.varSpace + arg2.varSpace))

        # Fast-handling: Avoid conversions if both are standard vector fields
        if fast_handling:
            coefLoc1 = VF_coeffs_direct(arg1, varSpaceLoc)
            coefLoc2 = VF_coeffs_direct(arg2, varSpaceLoc)
        elif fast_algorithm:
            # Convert both vector fields to real variables so that complex handling can be ignored later
            coefLoc1 = [symToReal(j) for j in VF_coeffs_direct(arg1, varSpaceLoc)]
            coefLoc2 = [symToReal(j) for j in VF_coeffs_direct(arg2, varSpaceLoc)]            
        else:
            coefLoc1 = VF_coeffs_direct(allToReal(arg1), varSpaceLoc)
            coefLoc2 = VF_coeffs_direct(allToReal(arg2), varSpaceLoc)

        # Combine the coefficients and optionally simplify
        if doNotSimplify:
            result_coefs = [arg1(coefLoc2[j], ignore_complex_handling=fast_handling or fast_algorithm) - arg2(coefLoc1[j], ignore_complex_handling=fast_handling or fast_algorithm) for j in range(len(varSpaceLoc))]
        else:
            result_coefs = [simplify(arg1(coefLoc2[j], ignore_complex_handling=fast_handling or fast_algorithm) - arg2(coefLoc1[j], ignore_complex_handling=fast_handling or fast_algorithm)) for j in range(len(varSpaceLoc))]

        return VFClass(varSpaceLoc, result_coefs, typeLoc)

    else:
        raise Exception('Expected arguments to be VFClass instances')

def contravariantVFTensorCoeffs(varSpace, *vector_fields):
    """
    Computes the contravariant tensor product of k vector fields with respect to the given variable space.

    This function evaluates the coefficients of the given vector fields in `vector_fields` on the specified 
    `varSpace` and returns a list representing the tensor product of these coefficients. It is primarily 
    used when evaluating a k-form on k vector fields.

    Parameters
    ----------
    varSpace : list or tuple
        A list or tuple of SymPy Symbol objects representing the variable space on which the vector fields are defined.
    
    *vector_fields : VFClass instances
        Any number of vector fields (instances of VFClass) that are evaluated in terms of the `varSpace`.
    
    Returns
    -------
    list
        A list of tuples, where each tuple contains:
        - A tuple of coordinate indices corresponding to the variables involved in each product term.
        - A SymPy expression representing the coefficient of the corresponding product term.

    Raises
    ------
    TypeError
        If any of the provided `vector_fields` is not an instance of `VFClass`.

    Examples
    --------
    >>> from DGCV import VFClass, contravariantVFTensorCoeffs, creatVariables
    >>> creatVariables('x1','x2')
    >>> vf1 = VFClass([x, y], [1, -y])
    >>> vf2 = VFClass([x, y], [y, 1])
    
    # Compute the contravariant tensor product
    >>> contravariantVFTensorCoeffs([x, y], vf1, vf2)
    [((0, 0), 1), ((0, 1), -y), ((1, 0), y), ((1, 1), -y**2)]
    """
    
    # Evaluate the coefficients of each vector field with respect to varSpace
    coeff_values = [VF_coeffs_direct(vf, varSpace, sparse=True) for vf in vector_fields] ### Update to the new VF_coeffs!!!
    
    # Get the number of terms in each vector field's coefficient list
    term_counts = [len(coeff_list) for coeff_list in coeff_values]
    
    # Initialize product of coefficients with the first vector field
    product_coeffs = coeff_values[0]
    
    # Compute the tensor product for the remaining vector fields
    for jj in range(1, len(term_counts)):
        product_coeffs = [(k[0] + l[0], k[1] * l[1]) for k in product_coeffs for l in coeff_values[jj]]
    
    return product_coeffs

def DF_coeffs_slow(arg1,arg2,doNotSimplify=False):
    """
    Evaluates the differential form (i.e., DFClass instance) *arg1* on coordinate vector fields corresponding to each variable in *arg2*, and returns the result as a list. The returned array represents the multilinear operator associated with the differential form *arg1*.
    
    Args:
        arg1: a DFClass class instance
        arg2: a list or tuple containing variables initialized with either *varWithVF* or *complexVarProc*.
        doNotSimplify: (optional keyword) True or False
    
    Returns:
        k-dimensional array of sympy expressions.
    
    Raises:
        NA
    """
    if arg1.__class__.__name__=='DFClass':
        formDegreeLoc=arg1.degree
        if formDegreeLoc>0:
            arrayLoc=MutableDenseNDimArray.zeros(*[len(arg2) for j in range(formDegreeLoc)])
            indexListLoc=[list(j) for j in chooseOp(range(len(arg2)),formDegreeLoc)]
            indexListFilteredLoc=[list(j) for j in chooseOp(range(len(arg2)),formDegreeLoc,withoutReplacement=True)]
            VFListLoc=['D_'+str(j) for j in arg2]
            for j in indexListLoc:
                arrayLoc[j]=arg1(*[eval(VFListLoc[k],_cached_caller_globals) for k in j]) if j in indexListFilteredLoc else 0
            if doNotSimplify==True:
                return arrayLoc
            else:
                return simplify(arrayLoc)
        else:
            return arg1.coeffs
    else:
        raise Exception('DF_coeffs expected first positional argument to be a DFClass instance.')

def changeDFBasis(arg1,arg2):
    newDFDataLoc=sparseKFormDataNewBasis(arg1.DFClassDataMinimal,arg1.varSpace,arg2)
    return DFClass(arg2,newDFDataLoc,arg1.degree,DGCVType=arg1.DGCVType,simplifyKW=arg1.simplifyKW)

def changeTFBasis(arg1,arg2):
    newTFDataLoc=sparseKFormDataNewBasis(arg1.TFClassDataMinimal,arg1.varSpace,arg2)
    return TFClass(arg2,newTFDataLoc,arg1.degree,DGCVType=arg1.DGCVType,simplifyKW=arg1.simplifyKW)

def changeSTFBasis(arg1,arg2):
    newSTFDataLoc=sparseKFormDataNewBasis(arg1.STFClassDataMinimal,arg1.varSpace,arg2)
    return STFClass(arg2,newSTFDataLoc,arg1.degree,DGCVType=arg1.DGCVType,simplifyKW=arg1.simplifyKW)

def addDF(*args, doNotSimplify=False):
    """
    Adds the given differential k-forms (DFClass instances). All forms must have the same degree.

    This function adds any number of differential forms, returning a single differential form representing their sum.
    If the forms are defined on different variable spaces, the function will handle the change of basis and align the 
    forms on a common variable space. The resulting form is returned as a DFClass instance.

    Parameters
    ----------
    *args : DFClass
        Any number of differential forms to be added. All forms must have the same degree.
    
    doNotSimplify : bool, optional
        If True, the result is returned without simplifying the coefficients (default is False).

    Returns
    -------
    DFClass
        A differential form (DFClass instance) representing the sum of the input forms.

    Raises
    ------
    Exception
        If the input forms have different degrees, or if the input arguments are not instances of DFClass.

    Examples
    --------
    >>> from sympy import symbols
    >>> from DGCV import DFClass, addDF
    >>> x, y, z = symbols('x y z')
    >>> df1 = DFClass([x, y], {(0,): 1, (1,): 2}, 1)
    >>> df2 = DFClass([x, y], {(0,): 3, (1,): 4}, 1)
    >>> df_sum = addDF(df1, df2)
    >>> df_sum
    DFClass([x, y], {(0,): 4, (1,): 6}, 1)
    
    # Adding forms with change of basis
    >>> df3 = DFClass([y, z], {(0,): 1, (1,): 2}, 1)
    >>> df_sum_with_basis_change = addDF(df1, df3)
    >>> df_sum_with_basis_change
    DFClass([x, y, z], {...}, 1)  # Aligned on common variable space
    """
    if len(args) == 0:
        return args
    if len(args) == 1:
        return args[0]
    if all([isinstance(j,DFClass) for j in args]):
        if len(set([j.degree for j in args])) == 1:
            typeList = list(set([j.DGCVType for j in args]))
            if len(typeList) == 1 and typeList[0] == 'complex':
                typeLoc = 'complex'
                if args[0]._varSpace_type == 'real':
                    args=[allToReal(df) if df._varSpace_type!='real' else df for df in args]
                else:
                    args=[allToSym(df) if df._varSpace_type!='complex' else df for df in args]
            else:
                typeLoc = 'standard'
                if len(typeList) != 1:
                    warnings.warn('Addition was performed between differential forms of `DGCVType` both `complex` and `standard`, so the resulting DFClass object has `DGCVType=\'standard\'`, which disables complex variable handling. To preserves `DGCVType=\'complex\'` ensure all DFClass objects in sum have `DGCVType=\'complex\'`.')

            # Fix here: Convert varSpace to tuples
            if len(set([tuple(j.varSpace) for j in args])) != 1:
                varSpaceLoc = list(dict.fromkeys(sum([j.varSpace for j in args], ())))
                args = [changeDFBasis(j, varSpaceLoc) for j in args]
            else:
                varSpaceLoc = args[0].varSpace
            
            coeffsLoc=dict()
            if doNotSimplify:
                for j in args:
                    dictLoc=j.DFClassDataDict
                    for a,b in dictLoc.items():
                        permS,orderedList=permSign(a,returnSorted=True)
                        orderedTuple=tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = coeffsLoc[orderedTuple]+permS*b
                        else:
                            coeffsLoc[orderedTuple] = permS*b
            else:
                for j in args:
                    dictLoc=j.DFClassDataDict
                    for a,b in dictLoc.items():
                        permS,orderedList=permSign(a,returnSorted=True)
                        orderedTuple=tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = simplify(coeffsLoc[orderedTuple]+permS*b)
                        else:
                            coeffsLoc[orderedTuple] = permS*b            

            return DFClass(tuple(varSpaceLoc), coeffsLoc, args[0].degree, typeLoc)
        else:
            raise Exception('cannot add differential forms of different degrees. Suggestion: organize components of different degrees in a list that represents their sum and compute the addition with list comprehension.')
    else:
        raise Exception('addDF expected all arguments to be DFClass instances.')

def scaleDF(arg1, arg2):
    """
    Multiplies the given differential form (DFClass instance) by a scalar or function.

    This function scales a differential form `arg2` by a scalar or a symbolic expression `arg1`. The scaling is applied
    to each coefficient in the differential form, resulting in a new scaled differential form.

    Parameters
    ----------
    arg1 : Expr
        A SymPy expression representing the scalar or function to scale the differential form by.
    
    arg2 : DFClass
        A differential form (DFClass instance) that will be scaled by `arg1`.

    Returns
    -------
    DFClass
        A new differential form (DFClass instance) where each coefficient has been multiplied by `arg1`.

    Raises
    ------
    Exception
        If `arg2` is not an instance of DFClass.

    Examples
    --------
    >>> from DGCV import DFClass, scaleDF, creatVariables
    >>> creatVariables('x1','x2')
    >>> df = DFClass([x, y], {(0,): 1, (1,): 2}, 1)
    >>> scale = 3
    >>> scaled_df = scaleDF(scale, df)
    >>> scaled_df
    DFClass([x, y], {(0,): 3, (1,): 6}, 1)
    
    # Scaling by a symbolic function
    >>> f = x + y
    >>> scaled_df_func = scaleDF(f, df)
    >>> scaled_df_func
    DFClass([x, y], {(0,): x + y, (1,): 2*(x + y)}, 1)
    """
    if arg2.__class__.__name__ == 'DFClass':
        return DFClass(arg2.varSpace, {a: arg1 * b for a, b in arg2.DFClassDataDict.items()}, arg2.degree, arg2.DGCVType)
    else:
        raise Exception('scaleDF expected second positional argument to be a DFClass instance.')

def exteriorProduct(*args, doNotSimplify=False):
    """
    Computes the exterior (wedge) product of differential forms (DFClass instances).

    This function takes any number of DFClass instances and computes their exterior product. It works by first
    aligning the variable spaces of the forms, multiplying terms, and enforcing antisymmetry by using a permutation-based
    approach. The result is a new DFClass instance representing the product.

    Parameters
    ----------
    args : DFClass
        One or more differential forms (DFClass instances).
    
    doNotSimplify : bool, optional
        If True, the resulting form's coefficients will not be simplified (default is False).

    Returns
    -------
    DFClass
        The exterior product of the input forms, as a new DFClass instance.

    Raises
    ------
    Exception
        If any of the arguments are not instances of DFClass.
    
    Examples
    --------
    >>> from DGCV import DFClass, exteriorProduct, creatVariables
    >>> creatVariables('x1','x2')
    >>> df1 = DFClass([x], {0: 1}, 1)
    >>> df2 = DFClass([y], {0: 1}, 1)
    >>> exteriorProduct(df1, df2)
    DFClass([x, y], {(0, 1): 1}, 2)
    """
    # Check if all arguments are DFClass instances
    if not all(isinstance(j, DFClass) for j in args):
        raise Exception("Expected all arguments to be instances of DFClass")

    # Determine the type of the exterior product (standard or complex)
    typeList = {j.DGCVType for j in args}
    typeLoc = 'complex' if len(typeList) == 1 and 'complex' in typeList else 'standard'
    if typeLoc == 'complex':
        if args[0]._varSpace_type == 'real':
            if any(j._varSpace_type != 'real' for j in args[1:]):
                args = [allToReal(j) for j in args]
        else:
            if any(j._varSpace_type != 'complex' for j in args[1:]):
                args = [allToSym(j) for j in args]            

    # Efficient term combination for k-forms
    def kFormTermMultiplier(arg1, arg2, degr):
        result = dict()
        for j in arg1:
            for k in arg2:
                combined_indices = j[0] + k[0]
                if len(combined_indices) == len(set(combined_indices)):  # Ensure antisymmetry
                    permS,orderedList=permSign(combined_indices,returnSorted=True)
                    orderedTuple=tuple(orderedList)
                    if orderedTuple in result:
                        result[orderedTuple]=result[orderedTuple]+permS*j[1]*k[1]
                    else:
                        result[orderedTuple]=permS*j[1]*k[1]
        return result if result else {(0,)*degr:0}

    # Compute the exterior product of two forms
    def exteriorProductOf2(arg1, arg2):
        varSpaceLoc = tuple(dict.fromkeys(arg1.varSpace + arg2.varSpace))  # Combined variable space
        prodDegree = arg1.degree + arg2.degree

        # Align the variable spaces and get terms in the new basis
        newForm1Loc = changeDFBasis(arg1, varSpaceLoc).DFClassDataMinimal
        newForm2Loc = changeDFBasis(arg2, varSpaceLoc).DFClassDataMinimal

        # Multiply terms and enforce antisymmetry
        productTermsData = kFormTermMultiplier(newForm1Loc, newForm2Loc, prodDegree)

        # Create new DFClass instances for the product terms
        return DFClass(varSpaceLoc, productTermsData, prodDegree, typeLoc)

    # Handle multiple differential forms
    if len(args) > 1:
        resultLoc = args[0]
        for j in range(1, len(args)):
            resultLoc = exteriorProductOf2(resultLoc, args[j])
        return resultLoc
    elif len(args) == 1:
        return args[0]

def sparseKFormDataNewBasis(sparseKFormData,oldBasis,newBasis):
    """
    Converts the indices of a k-form's sparse data representation from an old basis to a new basis.

    This function is primarily a helper for DGCV's exterior product calculus, allowing sparse data representing 
    differential forms to be transformed into a new variable basis.

    Parameters
    ----------
    sparseKFormData : list of tuples
        A list where each tuple consists of the indices of variables in the old basis and the corresponding coefficient.
    
    oldBasis : list
        A list of variables representing the current basis.
    
    newBasis : list
        A list of variables representing the new basis, into which the k-form will be transformed.

    Returns
    -------
    list of tuples
        A transformed list where the indices are mapped from the old basis to the new basis, preserving the 
        corresponding coefficients.

    Raises
    ------
    ValueError
        If a variable in the old basis is not found in the new basis.
    
    Examples
    --------
    >>> oldBasis = ['x', 'y', 'z']
    >>> newBasis = ['y', 'x', 'z']
    >>> sparseKFormData = [((0, 1), 1), ((1, 2), -1)]
    >>> sparseKFormDataNewBasis(sparseKFormData, oldBasis, newBasis)
    {(0, 2):−1, (1, 0):1}
    """
    if not sparseKFormData: # Maybe safe to remove following October DFClassDataDict reformat!!!
        return {tuple(): 0}
    degree=len(sparseKFormData[0][0])
    try:
        dataDict=dict([(tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1]) for j in sparseKFormData if j[1]!=0])
    except ValueError as e:
        raise ValueError(f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form\'s definition: {e}")
    if not dataDict:
        dataDict={(0,)*degree:0}
    return dataDict


############## tensor fields

def addSTF(*args, doNotSimplify=False):
    """
    Adds symmetric tensor fields (STFClass objects) of the same degree.
    """
    if len(args) == 0:
        return args
    if len(args) == 1:
        return args[0]
    if all([isinstance(j,STFClass) for j in args]):
        if len(set([j.degree for j in args])) == 1:
            typeList = list(set([j.DGCVType for j in args]))
            if len(typeList) == 1 and typeList[0] == 'complex':
                typeLoc = 'complex'
                if args[0]._varSpace_type == 'real':
                    args=[allToReal(stf) if stf._varSpace_type!='real' else stf for stf in args]
                else:
                    args=[allToSym(stf) if stf._varSpace_type!='complex' else stf for stf in args]
            else:
                typeLoc = 'standard'
                if len(typeList) != 1:
                    warnings.warn('Addition was performed between symmetric tensor fields of `DGCVType` both `complex` and `standard`, so the resulting STFClass object has `DGCVType=\'standard\'`, which disables complex variable handling. To preserves `DGCVType=\'complex\'` ensure all STFClass objects in sum have `DGCVType=\'complex\'`.')

            # Fix here: Convert varSpace to tuples
            if len(set([tuple(j.varSpace) for j in args])) != 1:
                varSpaceLoc = list(dict.fromkeys(sum([j.varSpace for j in args], ())))
                args = [changeSTFBasis(j, varSpaceLoc) for j in args]
            else:
                varSpaceLoc = args[0].varSpace
            
            coeffsLoc=dict()
            if doNotSimplify:
                for j in args:
                    dictLoc=j.STFClassDataDict
                    for a,b in dictLoc.items():
                        _,orderedList=permSign(a,returnSorted=True)
                        orderedTuple=tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = coeffsLoc[orderedTuple]+b
                        else:
                            coeffsLoc[orderedTuple] = b
            else:
                for j in args:
                    dictLoc=j.STFClassDataDict
                    for a,b in dictLoc.items():
                        _,orderedList=permSign(a,returnSorted=True)
                        orderedTuple=tuple(orderedList)
                        if orderedTuple in coeffsLoc:
                            coeffsLoc[orderedTuple] = simplify(coeffsLoc[orderedTuple]+ b)
                        else:
                            coeffsLoc[orderedTuple] = b            

            return STFClass(tuple(varSpaceLoc), coeffsLoc, args[0].degree, typeLoc)
        else:
            raise Exception('Adding symmetric tensor fields of different degrees is not supported at this time. Suggestion: organize components of different degrees in a list that represents their sum and compute the addition with list comprehension.')
    else:
        raise Exception('addSTF expected all arguments to be STFClass instances.')

def addTF(*args, doNotSimplify=False):
    """
    Adds tensor fields (TFClass objects) of the same degree.
    """
    if len(args) == 0:
        return args
    if len(args) == 1:
        return args[0]
    if all([isinstance(j,TFClass) for j in args]):
        if len(set([j.degree for j in args])) == 1:
            typeList = list(set([j.DGCVType for j in args]))
            if len(typeList) == 1 and typeList[0] == 'complex':
                typeLoc = 'complex'
                if args[0]._varSpace_type == 'real':
                    args=[allToReal(tf) if tf._varSpace_type!='real' else tf for tf in args]
                else:
                    args=[allToSym(tf) if tf._varSpace_type!='complex' else tf for tf in args]
            else:
                typeLoc = 'standard'
                if len(typeList) != 1:
                    warnings.warn('Addition was performed between tensor fields of `DGCVType` both `complex` and `standard`, so the resulting STFClass object has `DGCVType=\'standard\'`, which disables complex variable handling. To preserves `DGCVType=\'complex\'` ensure all TFClass objects in sum have `DGCVType=\'complex\'`.')

            # Fix here: Convert varSpace to tuples
            if len(set([tuple(j.varSpace) for j in args])) != 1:
                varSpaceLoc = list(dict.fromkeys(sum([j.varSpace for j in args], ())))
                args = [changeTFBasis(j, varSpaceLoc) for j in args]
            else:
                varSpaceLoc = args[0].varSpace
            
            coeffsLoc=dict()
            if doNotSimplify:
                for j in args:
                    dictLoc=j.TFClassDataDict
                    for a,b in dictLoc.items():
                        if a in coeffsLoc:
                            coeffsLoc[a] = coeffsLoc[a]+b
                        else:
                            coeffsLoc[a] = b
            else:
                for j in args:
                    dictLoc=j.TFClassDataDict
                    for a,b in dictLoc.items():
                        if a in coeffsLoc:
                            coeffsLoc[a] = simplify(coeffsLoc[a]+ b)
                        else:
                            coeffsLoc[a] = b            

            return TFClass(tuple(varSpaceLoc), coeffsLoc, args[0].degree, typeLoc)
        else:
            raise Exception('Adding tensor fields of different degrees is not supported at this time. Suggestion: organize components of different degrees in a list that represents their sum and compute the addition with list comprehension.')
    else:
        raise Exception('addTF expected all arguments to be TFClass instances.')

def scaleTF(arg1, arg2):
    """
    Scales tensor fields TFClass and STFClass alike.
    """
    if isinstance(arg2,(TFClass)):
        return TFClass(arg2.varSpace, {a: arg1 * b for a, b in arg2.TFClassDataDict.items()}, arg2.degree, DGCVType=arg2.DGCVType,simplifyKW=arg2.simplifyKW)
    elif isinstance(arg2,(STFClass)):
        return STFClass(arg2.varSpace, {a: arg1 * b for a, b in arg2.STFClassDataDict.items()}, arg2.degree, DGCVType=arg2.DGCVType,simplifyKW=arg2.simplifyKW)
    else:
        raise Exception('scaleTF expected second positional argument to be a TFClass or STFClass instance.')

def tensorProduct(*args, doNotSimplify=False):
    """
    Computes the tensor product of tensor fields (TFClass or STFClass instances).

    This function takes any number of TFClass or STFClass instances and computes their tensor product. It works by first
    aligning the variable spaces of the forms, reindexing structure data, and then multiplying terms. The result is a new TFClass instance representing the product.
    
    Note products of STFClass objects always return TFClass.

    Parameters
    ----------
    args : TFClass or STFClass
        One or more tensor fields (TFClass or STFClass instances).
    
    doNotSimplify : bool, optional
        If True, the resulting form's coefficients will not be simplified (default is False).

    Returns
    -------
    TFClass
        The tensor product of the input tensor fields, as a new TFClass instance.

    Raises
    ------
    Exception
        If any of the arguments are not instances of TFClass or STFClass.
    """
    # Check if all arguments are TFClass or STFClass instances
    if not all(isinstance(j, (TFClass, STFClass)) for j in args):
        raise Exception("Expected all arguments to be instances of TFClass or STFClass")

    # Determine the type of the tensor product (standard or complex)
    format_preference = [j._varSpace_type for j in args]
    if len(set(format_preference))!=1:
        raise TypeError('`tensorProduct` between tensorfields with different `_varSpace_type` attribute values is not yet supported.')

    typeList = {j.DGCVType for j in args}
    typeLoc = 'complex' if len(typeList) == 1 and 'complex' in typeList else 'standard'

    # Efficient term combination for k-forms
    def TermMultiplier(arg1, arg2, degr):
        result = dict()
        for j in arg1:
            for k in arg2:
                combined_indices = j[0] + k[0]
                multiplied_value = j[1] * k[1]
                result[combined_indices] = multiplied_value
        return result if result else {(0,)*degr:0}

    # Compute the exterior product of two forms
    def tensorProductOf2(arg1, arg2):
        if isinstance(arg1, STFClass):
            arg1 = TFClass(arg1.varSpace,arg1.STFClassDataDictFull,arg1.degree,DGCVType=arg1.DGCVType,simplifyKW=arg1.simplifyKW)
        if isinstance(arg2, STFClass):
            arg2 = TFClass(arg2.varSpace,arg2.STFClassDataDictFull,arg2.degree,DGCVType=arg2.DGCVType,simplifyKW=arg2.simplifyKW)

        varSpaceLoc = tuple(dict.fromkeys(arg1.varSpace + arg2.varSpace))  # Combined variable space
        prodDegree = arg1.degree + arg2.degree

        # Align the variable spaces and get terms in the new basis
        newField1Loc = changeTFBasis(arg1, varSpaceLoc).TFClassDataMinimal
        newField2Loc = changeTFBasis(arg2, varSpaceLoc).TFClassDataMinimal

        # Multiply terms and enforce antisymmetry
        productTermsData = TermMultiplier(newField1Loc, newField2Loc, prodDegree)

        # Create new DFClass instances for the product terms
        return TFClass(varSpaceLoc, productTermsData, prodDegree, typeLoc)

    # Handle multiple differential forms
    if len(args) > 1:
        resultLoc = args[0]
        for j in range(1, len(args)):
            resultLoc = tensorProductOf2(resultLoc, args[j])
        return resultLoc
    elif len(args) == 1:
        return args[0]


############## complex vector fields

def holVF_coeffs(arg1, arg2, doNotSimplify=False):
    """
    Evaluates the vector field (i.e., VFClass instance) *arg1* on each holomorphic variable in *arg2*, 
    and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the 
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate 
    vector fields.

    Parameters:
    -----------
    arg1 : VFClass
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).
    
    Returns:
    --------
    list
        A list of sympy expressions representing the coefficients in holomorphic coordinates.

    Raises:
    -------
    Exception
        If any variables in *arg2* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import complexVarProc, holVF_coeffs
    >>> complexVarProc('z','x','y',3)
    >>> holVF_coeffs(D_z1 + D_BARz2, [z1, z2, z3])
    [1, 0, 0]
    """
    if doNotSimplify:
        return [realToHol(arg1(j)) for j in arg2]
    else:
        return [simplify(allToHol(arg1(j))) for j in arg2]

def antiholVF_coeffs(arg1, arg2, doNotSimplify=False):
    """
    Evaluates the vector field (i.e., VFClass instance) *arg1* on the conjugate of each holomorphic variable 
    in *arg2*, and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the 
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate 
    vector fields.

    Parameters:
    -----------
    arg1 : VFClass
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).
    
    Returns:
    --------
    list
        A list of sympy expressions representing the coefficients in antiholomorphic coordinates.

    Raises:
    -------
    Exception
        If any variables in *arg2* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import complexVarProc, antiholVF_coeffs
    >>> complexVarProc('z','x','y',3)
    >>> antiholVF_coeffs(D_z1 + D_BARz2, [z1, z2, z3])
    [0, 1, 0]
    """
    if doNotSimplify:
        return [realToHol(arg1(conjugate(j))) for j in arg2]
    else:
        return [simplify(realToHol(arg1(conjugate(j)))) for j in arg2]

def complexVFC(arg1, arg2, doNotSimplify=False):
    """
    Evaluates the vector field (i.e., VFClass instance) *arg1* on the holomorphic variables in *arg2* 
    and their complex conjugates, returning the result as two lists of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the 
    coefficients for both the holomorphic and antiholomorphic parts of the vector field when expressed in 
    terms of the respective coordinate vector fields.

    Parameters:
    -----------
    arg1 : VFClass
        A vector field instance to evaluate on the holomorphic and antiholomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).
    
    Returns:
    --------
    tuple of two lists
        The first list contains the coefficients of the holomorphic part, and the second list contains 
        the coefficients of the antiholomorphic part.

    Raises:
    -------
    Exception
        If *arg1* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import complexVarProc, complexVFC
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> complexVFC(D_z1 + D_BARz2, [z1, z2, z3])
    ([1, 0, 0], [0, 1, 0])
    """
    if arg1.__class__.__name__ == 'VFClass':
        # Use holVF_coeffs and antiholVF_coeffs, with optional simplification
        hol_coeffs = holVF_coeffs(arg1, arg2, doNotSimplify=doNotSimplify)
        antihol_coeffs = antiholVF_coeffs(arg1, arg2, doNotSimplify=doNotSimplify)
        return hol_coeffs, antihol_coeffs
    else:
        raise Exception('Expected first positional argument to be of type VFClass')

def conjComplex(arg):
    """
    Computes the complex conjugate of a complex vector field (i.e., VFClass instance).

    Parameters:
    -----------
    arg : VFClass
        A complex vector field instance whose complex conjugate is to be computed.
    
    Returns:
    --------
    VFClass
        A new vector field instance representing the complex conjugate of the input vector field.

    Raises:
    -------
    Exception
        If *arg* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import complexVarProc, conjComplexVF
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> vf = assembleFromCompVFC((z1, z2, 0), (BARz1, BARz2, 0), [z1, z2, z3])
    >>> conjComplexVF(vf)
    VFClass instance representing the complex conjugate of *vf*
    """
    warnings.warn('`conjComplex` has been depricated. Use conjugate_DGCV instead')
    return _conjComplexVFDF(arg)

def _conjComplexVFDF(arg):
    """
    Computes the complex conjugate of a complex vector field or differential form (i.e., VFClass/DFClass instance).

    Parameters:
    -----------
    arg : VFClass/DFClass
        A complex vector field instance whose complex conjugate is to be computed.
    
    Returns:
    --------
    VFClass/DFClass
        A new vector field instance representing the complex conjugate of the input vector field.

    Raises:
    -------
    Exception
        If *arg* is not an instance of VFClass.

    Example:
    --------
    >>> from DGCV import complexVarProc, conjComplexVF
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> vf = assembleFromCompVFC((z1, z2, 0), (BARz1, BARz2, 0), [z1, z2, z3])
    >>> conjComplexVF(vf)
    VFClass instance representing the complex conjugate of *vf*
    """
    # Ensure the argument is a VFClass instance
    if isinstance(arg, VFClass):
        # Return the complex conjugate of the vector field
        if arg.DGCVType=='complex':
            if arg._varSpace_type == 'complex':
                arg=allToReal(arg)
                return allToSym(VFClass(arg.varSpace, [conjugate(j) for j in arg.coeffs], DGCVType='complex',simplifyKW=arg.simplifyKW))
        return VFClass(arg.varSpace, [conjugate(j) for j in arg.coeffs], DGCVType=arg.DGCVType,simplifyKW=arg.simplifyKW)
    elif isinstance(arg, DFClass):
        # Return the complex conjugate of the vector field
        if arg.DGCVType=='complex':
            if arg._varSpace_type == 'complex':
                arg=allToReal(arg)
                return allToSym(DFClass(arg.varSpace, {a:conjugate(b) for a,b in arg.DFClassDataDict.items()}, arg.degree, DGCVType='complex',simplifyKW=arg.simplifyKW))
        return DFClass(arg.varSpace, {a:conjugate(b) for a,b in arg.DFClassDataDict.items()}, arg.degree, DGCVType=arg.DGCVType,simplifyKW=arg.simplifyKW)
    else:
        raise Exception('Expected the input to be of type VFClass or DFClass.')
    
def realPartOfVF(arg1,*args):
    """
    Computes the real part of a complex vector field (i.e., VFClass instance) *arg1* expressed in terms 
    of complex variables in *arg2*.

    The real part is obtained by adding the vector field to its complex conjugate.

    Parameters:
    -----------
    arg1 : VFClass
        A complex vector field instance whose real part is to be computed.
    arg2 : list or tuple
        A list or tuple containing Symbol objects that were initialized as complex variables via complexVarProc.
    
    Returns:
    --------
    VFClass
        A new vector field instance representing the real part of the input complex vector field.

    Raises:
    -------
    Exception
        If *arg1* is not an instance of VFClass, or if variables in *arg2* were not initialized via complexVarProc.

    Example:
    --------
    >>> from DGCV import complexVarProc, realPartOfVF, assembleFromCompVFC
    >>> from sympy import simplify
    >>> complexVarProc('z', 'x', 'y', 3)
    >>> vf = assembleFromCompVFC((z1, z2, 0), (BARz1, BARz2, 0), [z1, z2, z3])
    >>> print(simplify(realPartOfVF(vf, [z1, z2, z3]).simplify_format('real'))) # using simplify, it is easy to see that it is real!
    2*x1*D_x1 + 2*x2*D_x2 + 2*y1*D_y1 + 2*y2*D_y2
    """
    # Ensure the argument is a VFClass instance
    if isinstance(arg1, VFClass):
        # Return the real part of the vector field by adding it to its complex conjugate
        return addVF(arg1, _conjComplexVFDF(arg1))
    else:
        raise Exception('Expected the input to be of type VFClass.')


############## finite algebra creation and tools

def createFiniteAlg(obj, label, basis_labels=None, grading=None, format_sparse=False, process_matrix_rep=False):
    """
    Registers an algebra object and its basis elements in the caller's global namespace,
    and adds them to variable_registry for tracking in the Variable Management Framework.

    Parameters
    ----------
    obj : FAClass or structure data
        The algebra object (an instance of FAClass) or the structure data used to create one.
    label : str
        The label used to reference the algebra object in the global namespace.
    basis_labels : list, optional
        A list of custom labels for the basis elements of the algebra. If not provided, default labels will be generated.
    grading : list, optional
        A list specifying the grading of the algebra.
    format_sparse : bool, optional
        Whether to use sparse arrays when creating the FAClass object.
    process_matrix_rep : bool, optional
        Whether to compute and store the matrix representation of the algebra.
    """
    def validate_structure_data(data, process_matrix_rep=False):
        """
        Validates the structure data and converts it to a list of lists of lists.

        Parameters
        ----------
        data : list or array-like
            The structure data to be validated. Can be:
            - A 3D list of structure constants (list of lists of lists)
            - A list of VFClass objects
            - A list of square matrices

        process_matrix_rep : bool, optional
            If True, the function will interpret the data as a list of square matrices
            and pass them to algebraDataFromMatRep to compute the structure data.

        Returns
        -------
        list
            The validated structure data as a list of lists of lists.

        Raises
        ------
        ValueError
            If the structure data is invalid or cannot be processed into a valid 3D list.
        """
        # Case 1: If process_matrix_rep is True, handle matrix representation
        if process_matrix_rep:
            if all(isinstance(Matrix(obj), Matrix) and Matrix(obj).shape[0] == Matrix(obj).shape[1] for obj in data):
                return algebraDataFromMatRep(data)
            else:
                raise ValueError("Matrix representation requires a list of square matrices.")

        # Case 2: If the input is a list of VFClass objects, handle vector fields
        if all(isinstance(obj, VFClass) for obj in data):
            return algebraDataFromVF(data)

        # Case 3: Validate and return the data as a list of lists of lists
        try:
            # Ensure the data is a 3D list-like structure
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                if len(data) == len(data[0]) == len(data[0][0]):
                    return data  # Return as a validated 3D list
                else:
                    raise ValueError("Structure data must have 3D shape (x, x, x).")
            else:
                raise ValueError("Structure data format must be a 3D list of lists.")
        except Exception as e:
            raise ValueError(f"Invalid structure data format: {type(data)} - {e}")

    # validate_label    
    label=validate_label(label)

    clearVar(label,report=False)

    # Validate or create the FAClass object
    if isinstance(obj, FAClass):
        # Create or validate basis labels
        if basis_labels is None:
            basis_labels = [validate_label(f'{label}_{i+1}') for i in range(obj.dimension)]
        validate_label_list(basis_labels)
        passkey = retrieve_passkey()  # Get the passkey for secure initialization
        algebra_obj = FAClass(
            structure_data=obj.structureData,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey  # Pass the secure key
        )
    else:
        structure_data = validate_structure_data(obj, process_matrix_rep=process_matrix_rep)
        # Create or validate basis labels
        if basis_labels is None:
            basis_labels = [validate_label(f'{label}_{i+1}') for i in range(len(structure_data))]
        validate_label_list(basis_labels)
        passkey = retrieve_passkey()  # Get the passkey for secure initialization
        algebra_obj = FAClass(
            structure_data=structure_data,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey  # Pass the secure key
        )
    
    # Ensure that the algebra object and its basis are fully initialized
    assert algebra_obj.basis is not None, "Algebra object basis elements must be initialized."

    # Register the algebra object in _cached_caller_globals
    _cached_caller_globals.update({label: algebra_obj})

    # Register the basis elements in _cached_caller_globals
    _cached_caller_globals.update(zip(basis_labels, algebra_obj.basis))

    # Define family names and values (basis labels and their corresponding objects)
    family_names = tuple(basis_labels)
    family_values = tuple([_cached_caller_globals[j] for j in basis_labels])

    # Define family relatives (mapping basis elements to their properties)
    family_relatives = {
        a: {
            'coefficients': _cached_caller_globals[a].coeffs,  # Coefficients
            'grading': grading[i] if grading else None,  # Grading, if applicable
            'index': i,  # Index in the algebra
            'is_sparse': format_sparse,  # Whether it's sparse
            'algebra_label': label,  # Reference to the parent algebra
        }
        for i, a in enumerate(basis_labels)
    }

    variable_registry = get_variable_registry()
    variable_registry['finite_algebra_systems'][label] = {
        'family_type': 'algebra',
        'family_names': family_names,
        'family_values': family_values,
        'family_relatives': family_relatives,
        'dimension': algebra_obj.dimension,
        'grading': grading,
        'basis_labels': basis_labels,
        'structure_data': algebra_obj.structureData
    }

    # Optionally return the algebra object
    return algebra_obj

def algebraDataFromVF(vector_fields):
    """
    Create the structure data array for a Lie algebra from a list of vector fields in *vector_fields*.

    This function computes the Lie algebra structure constants from a list of vector fields
    (instances of VFClass) defined on the same variable space. The returned structure data
    can be used to initialize an FAClass instance.

    Parameters
    ----------
    vector_fields : list
        A list of VFClass instances, all defined on the same variable space with respect to the same basis.

    Returns
    -------
    list
        A 3D array-like list of lists of lists representing the Lie algebra structure data.

    Raises
    ------
    Exception
        If the vector fields do not span a Lie algebra or are not defined on a common basis.

    Notes
    -----
    This function dynamically chooses its approach to solve for the structure constants:
    - For smaller dimensional algebras, it substitutes pseudo-arbitrary values for the variables in `varSpaceLoc`
      based on a power function to create a system of linear equations.
    - For larger systems, where `len(varSpaceLoc)` raised to `len(vector_fields)` exceeds a threshold (default is 10,000),
      random rational numbers are used for substitution to avoid performance issues caused by large numbers.
    """
    # Define the product threshold for switching to random sampling
    product_threshold = 1

    # Check if all vector fields are defined on the same variable space
    if len(set([vf.varSpace for vf in vector_fields])) != 1:
        raise Exception('algebraDataFromVF requires vector fields defined with respect to a common basis.')
    
    complexHandling = any(vf.DGCVType == 'complex' for vf in vector_fields)
    if complexHandling:
        vector_fields = [allToReal(j) for j in vector_fields]
    varSpaceLoc = vector_fields[0].varSpace
    
    # Create temporary variables for solving structure constants
    tempVarLabel = 'T' + retrieve_public_key()
    variableProcedure(tempVarLabel, len(vector_fields), _tempVar=retrieve_passkey())
    combiVFLoc = addVF(*[_cached_caller_globals[tempVarLabel][j] * vector_fields[j] 
                         for j in range(len(_cached_caller_globals[tempVarLabel]))])
    
    def computeBracket(j, k):
        """
        Compute and return the Lie bracket [vf_j, vf_k] and structure constants.

        Parameters
        ----------
        j : int
            Index of the first vector field.
        k : int
            Index of the second vector field.

        Returns
        -------
        list
            Structure constants for the Lie bracket of vf_j and vf_k.
        """
        if k <= j:
            return [0] * len(_cached_caller_globals[tempVarLabel])
        
        # Compute the Lie bracket
        bracket = VF_bracket(vector_fields[j], vector_fields[k]) - combiVFLoc
        
        if complexHandling:
            bracket = [allToReal(expr) for expr in bracket.coeffs]
        else:
            bracket = bracket.coeffs

        # Check if len(varSpaceLoc) ** len(vector_fields) exceeds the threshold
        if len(varSpaceLoc) ** len(vector_fields) <= product_threshold:
            # Use the current system of pseudo-arbitrary substitutions
            bracketVals = list(set(
                sum([[expr.subs([(varSpaceLoc[i], Rational((i + 1) ** sampling_index, 32)) 
                                 for i in range(len(varSpaceLoc))]) 
                      for expr in bracket] for sampling_index in range(len(vector_fields))], [])
            ))
        else:
            # Use random sampling system for larger cases
            random_rational = lambda: Rational(randint(1, 1000), randint(1001, 2000))
            bracketVals = list(set(
                sum([[expr.subs([(varSpaceLoc[i], random_rational()) 
                                 for i in range(len(varSpaceLoc))]) 
                      for expr in bracket] for _ in range(len(vector_fields))], [])
            ))
        
        # Solve the system of equations
        solutions = list(linsolve(bracketVals, _cached_caller_globals[tempVarLabel]))
        
        if len(solutions) == 1:
            # Extract the solution and substitute into all temporary variables
            sol_values = solutions[0]
            
            # Substitute back into the original bracket
            substituted_constants = [expr.subs(zip(_cached_caller_globals[tempVarLabel], sol_values)) 
                                      for expr in _cached_caller_globals[tempVarLabel]]
            
            return substituted_constants
        else:
            raise Exception(f'Fields at positions {j} and {k} are not closed under Lie brackets.')

    # Precompute all necessary Lie brackets and store as 3D list
    structure_data = [[[0 for _ in vector_fields] for _ in vector_fields] for _ in vector_fields]
    
    for j in range(len(vector_fields)):
        for k in range(j + 1, len(vector_fields)):
            structure_data[j][k] = computeBracket(j, k)
            structure_data[k][j] = [-elem for elem in structure_data[j][k]]
    
    # Clean up temporary variables
    clearVar(*listVar(temporary_only=True),report=False)

    return structure_data

def algebraDataFromMatRep(mat_list):
    """
    Create the structure data array for a Lie algebra from a list of matrices in *mat_list*.

    This function computes the Lie algebra structure constants from a matrix representation of a Lie algebra.
    The returned structure data can be used to initialize an FAClass instance.

    Parameters
    ----------
    mat_list : list
        A list of square matrices of the same size representing the Lie algebra.

    Returns
    -------
    list
        A 3D list of lists of lists representing the Lie algebra structure data.

    Raises
    ------
    Exception
        If the matrices do not span a Lie algebra, or if the matrices are not square and of the same size.
    """
    if isinstance(mat_list, list):
        mListLoc = [Matrix(j) for j in mat_list]  # Convert input to sympy Matrix objects
        shapeLoc = mListLoc[0].shape[0]

        # Ensure all matrices are square and of the same size
        if all(j.shape == (shapeLoc, shapeLoc) for j in mListLoc):
            # Temporary variables for solving the commutators
            tempVarLabel = 'T' + retrieve_public_key()
            variableProcedure(tempVarLabel, len(mat_list), _tempVar=retrieve_passkey())

            # Create a symbolic matrix to solve for commutators
            combiMatLoc = sum([_cached_caller_globals[tempVarLabel][j] * mListLoc[j] 
                            for j in range(len(_cached_caller_globals[tempVarLabel]))], zeros(shapeLoc, shapeLoc))

            def pairValue(j, k):
                """
                Compute the commutator [m_j, m_k] and match with the combination matrix.

                Parameters
                ----------
                j : int
                    Index of the first matrix in the commutator.
                k : int
                    Index of the second matrix in the commutator.

                Returns
                -------
                list
                    The coefficients representing the structure constants.
                """
                bracketVals = list(set((mListLoc[j] * mListLoc[k] - mListLoc[k] * mListLoc[j] - combiMatLoc).vec()))
                
                solLoc = list(linsolve(bracketVals, _cached_caller_globals[tempVarLabel]))
                
                if len(solLoc) == 1:
                    return [expr.subs(zip(_cached_caller_globals[tempVarLabel], solLoc[0])) for expr in _cached_caller_globals[tempVarLabel]]
                else:
                    raise Exception(f'Unable to determine if matrices are closed under commutators. '
                                    f'Problem matrices are in positions {j} and {k}.')

            # Assemble the structure data array from commutators and store as 3D list
            structure_data = [[pairValue(j, k) for j in range(len(_cached_caller_globals[tempVarLabel]))] 
                              for k in range(len(_cached_caller_globals[tempVarLabel]))]
            
            # Clear all temporary variables
            clearVar(*listVar(temporary_only=True),report=False)
            
            return structure_data
        else:
            raise Exception('algebraDataFromMatRep expects a list of square matrices of the same size.')
    else:
        raise Exception('algebraDataFromMatRep expects a list of square matrices.')

def killingForm(arg1,list_processing=False):
    if arg1.__class__.__name__ == 'FAClass':
        # Convert the structure data to a mutable array
        if not arg1.is_lie_algebra():
            raise Exception('killingForm expects argument to be a Lie algebra instance of the FAClass')
        if list_processing:
            aRepLoc=arg1.structureData
            return [[trace_matrix(multiply_matrices(aRepLoc[j],aRepLoc[k])) for k in range(arg1.dimension)] for j in range(arg1.dimension)]
        else:
            aRepLoc=adjointRepresentation(arg1)
            return Matrix(arg1.dimension,arg1.dimension, lambda j,k:(aRepLoc[j]*aRepLoc[k]).trace())
    else:
        raise Exception('killingForm expected to receive an FAClass instance.')

def adjointRepresentation(arg1,list_format=False):
    if arg1.__class__.__name__ == 'FAClass':
        # Convert the structure data to a mutable array
        if not arg1.is_lie_algebra():
            warnings.warn('Caution: The algebra passed to adjointRepresentation is not a Lie algebra.')
        if list_format:
            return arg1.structureData
        return [Matrix(j) for j in arg1.structureData]
    else:
        raise Exception('adjointRepresentation expected to receive an FAClass instance.')



############## variable management
def listVar(standard_only=False, complex_only=False, algebras_only=False, temporary_only=False, obscure_only=False, protected_only=False):
    """
    This function lists all parent labels for objects tracked within the DGCV Variable Management Framework (VMF). In particular strings that are keys in DGCV's internal `standard_variable_systems`, `complex_variable_systems`, and 'finite_algebra_systems' dictionaries. It also accepts optional keywords to filter the results, showing only standard, complex, temporary, or protected variable system labels.

    Parameters
    ----------
    standard_only : bool, optional
        If True, only standard variable system labels will be listed.
    complex_only : bool, optional
        If True, only complex variable system labels will be listed.
    algebras_only : bool, optional
        If True, only finite algebra system labels will be listed.
    temporary_only : bool, optional
        If True, only variable system labels marked as temporary will be listed.
    protected_only : bool, optional
        If True, only variable system labels marked as protected will be listed.

    Returns
    -------
    list
        A list of variable system labels matching the provided filters.

    Notes
    -----
    - If no filters are specified, the function returns all labels from both 
      `standard_variable_systems` and `complex_variable_systems`.
    - If multiple filters are specified, the function combines them, displaying 
      labels that meet all the selected criteria.
    
    Examples
    --------
    >>> from DGCV import variableProcedure, varWithVF, complexVarProc, createFiniteAlg, listVar
    >>> variableProcedure('a', 3)
    >>> varWithVF('b')
    >>> complexVarProc('z', 'x', 'y', 2)
    >>> createFiniteAlg([x1*D_x1, x1*D_x1-x2*D_x2, x2*D_x2], 'sl2')
    >>> listVar()
    ['sl2', 'b', 'z', 'a']

    >>> listVar(standard_only=True)
    ['b', 'a']

    >>> listVar(complex_only=True)
    """
    variable_registry = get_variable_registry()

    
    # Collect all labels from standard and complex variable systems
    standard_labels = set(variable_registry['standard_variable_systems'].keys())
    complex_labels = set(variable_registry['complex_variable_systems'].keys())
    algebra_labels = set(variable_registry['finite_algebra_systems'].keys())

    # Combine standard and complex labels if no specific filters are applied
    all_labels = standard_labels | complex_labels | algebra_labels

    # Apply filters
    if standard_only:
        all_labels = standard_labels
    if complex_only:
        all_labels = complex_labels
    if algebras_only:
        all_labels = algebra_labels
    if temporary_only:
        all_labels = all_labels & variable_registry['temporary_variables']
    if obscure_only:
        all_labels = all_labels & variable_registry['obscure_variables']
    if protected_only:
        all_labels = all_labels & variable_registry['protected_variables']

    # Return the filtered list of labels
    return list(all_labels)

def clearVar(*labels,report=True):
    """
    Clears variables from the registry and global namespace. Because sometimes, we all need a fresh start.

    This function takes one or more variable system labels (strings) and clears all
    associated variables, vector fields, differential forms, and metadata from the 
    DGCV system. Variable system refers to object systems created by the DGCV 
    variable creation functions `variableProcedure`, `varWithVF`, and 
    `complexVarProc`. Use `listVar()` to retriev a list of existed variable system 
    labels. The function handles both standard and complex variable systems, 
    ensuring that all related objects are removed from the caller's globals() 
    namespace, `variable_registry`, and the conversion dictionaries.

    Parameters
    ----------
    *labels : str
        One or more string labels representing variable systems (either 
        standard or complex). These labels will be removed along with all 
        associated components.

    Functionality
    -------------
    - For standard variable systems:
        1. All family members associated with the variable label will be 
           removed from the caller's globals() namespace.
        2. If the variable system has associated differential forms (DFClass) 
           or vector fields (VFClass), these objects will also be removed.
        3. The label will be removed from `temporary_variables`, if applicable.
        4. Finally, the label will be deleted from `standard_variable_systems` 
           in `variable_registry`.

    - For complex variable systems:
        1. For each complex variable system:
            - Labels for the real and imaginary parts will be removed 
              from `variable_registry['protected_variables']`.
            - If the system is a tuple, the parent labels for holomorphic, 
              antiholomorphic, real, and imaginary variable tuples will be 
              removed from the caller's globals() namespace.
            - The `variable_relatives` dictionary will be traversed to remove 
              all associated variables, vector fields, and differential forms 
              from the caller's globals() namespace.
            - The function will also clean up the corresponding entries in 
              `conversion_dictionaries`, depending on the `complex_positioning` 
              (holomorphic, antiholomorphic, real, or imaginary).
        2. The complex variable label will be removed from `temporary_variables`, 
           if applicable.
        3. Finally, the label will be deleted from `complex_variable_systems` 
           in `variable_registry`.

    Notes
    -----
    - This function ensures that variables and their associated metadata 
      are comprehensively cleared from the DGCV system, preventing any 
      dangling references in the internal data structures.
    - Use with `listVar` to expediantly clear everything, e.g., `clearVar(*listVar())`.

    Examples
    --------
    >>> clearVar('x') # removes any DGCV variable system labeled as x, such as 
                      # (x, D_x, d_x), (x=(x1, x2), x1, x2, D_x1, d_x1,...), etc.
    >>> clearVar('z', 'y', 'w')

    This will remove all variables, vector fields, and differential forms 
    associated with the labels 'x', 'z', 'y', and 'w', along with any 
    conversion dictionary entries.

    """
    # Access variable_registry from _cached_caller_globals
    variable_registry = get_variable_registry()
    

    # Lists to track cleared labels for printing summary later
    cleared_standard = []
    cleared_complex = []
    cleared_algebras = []

    for label in labels:
        # Handle standard variables
        if label in variable_registry['standard_variable_systems']:
            system_dict = variable_registry['standard_variable_systems'][label]
            family_names = system_dict['family_names']
            
            # Remove all family members and parent from _cached_caller_globals
            for var in family_names:
                if var in _cached_caller_globals:
                    del _cached_caller_globals[var]
            if label in _cached_caller_globals:
                del _cached_caller_globals[label]
            
            # Remove differential system if present
            if system_dict.get('differential_system', None):
                for var in family_names:
                    if f'D_{var}' in _cached_caller_globals:
                        del _cached_caller_globals[f'D_{var}']
                    if f'd_{var}' in _cached_caller_globals:
                        del _cached_caller_globals[f'd_{var}']
                if f'D_{label}' in _cached_caller_globals:
                    del _cached_caller_globals[f'D_{label}']
                if f'd_{label}' in _cached_caller_globals:
                    del _cached_caller_globals[f'd_{label}']
            
            # Remove temporary marker if present
            if system_dict.get('tempVar', None):
                variable_registry['temporary_variables'].discard(label)

            # Remove obscure marker if present
            if system_dict.get('obsVar', None):
                variable_registry['obscure_variables'].discard(label)
            
            # Remove from variable_registry
            del variable_registry['standard_variable_systems'][label]
            cleared_standard.append(label)  # Add to cleared list
        
        # Handle complex variables
        elif label in variable_registry['complex_variable_systems']:
            system_dict = variable_registry['complex_variable_systems'][label]
            family_houses = system_dict['family_houses']
            
            # Remove string labels for the real and imaginary variable parents from protected_variables
            real_parent = family_houses[-2]
            imag_parent = family_houses[-1]
            if real_parent in variable_registry['protected_variables']:
                variable_registry['protected_variables'].discard(real_parent)
            if imag_parent in variable_registry['protected_variables']:
                variable_registry['protected_variables'].discard(imag_parent)
            
            # Remove tuple parent labels if family_type is 'tuple'
            if system_dict['family_type'] == 'tuple' or system_dict['family_type'] == 'multi_index': #patch until multi_index is fully implemented
                for house in family_houses:
                    if house in _cached_caller_globals:
                        del _cached_caller_globals[house]
            
            # Remove variables and their associated DFClass, VFClass from _cached_caller_globals
            variable_relatives = system_dict['variable_relatives']
            for var_label, var_data in variable_relatives.items():
                if var_label in _cached_caller_globals:
                    del _cached_caller_globals[var_label]
                if var_data.get('DFClass') and f"D_{var_label}" in _cached_caller_globals:
                    del _cached_caller_globals[f"D_{var_label}"]
                if var_data.get('VFClass') and f"d_{var_label}" in _cached_caller_globals:
                    del _cached_caller_globals[f"d_{var_label}"]
            
            # Clean up conversion dictionaries
            conversion_dictionaries = variable_registry['conversion_dictionaries']
            for var_label, var_data in variable_relatives.items():
                position = var_data.get('complex_positioning')
                value_pointer = var_data.get('variable_value')
                
                if position == 'holomorphic':
                    conversion_dictionaries['conjugation'].pop(value_pointer, None)
                    conversion_dictionaries['holToReal'].pop(value_pointer, None)
                    conversion_dictionaries['symToReal'].pop(value_pointer, None)
                
                elif position == 'antiholomorphic':
                    conversion_dictionaries['symToHol'].pop(value_pointer, None)
                    conversion_dictionaries['symToReal'].pop(value_pointer, None)
                
                elif position in ['real', 'imaginary']:
                    conversion_dictionaries['realToHol'].pop(value_pointer, None)
                    conversion_dictionaries['realToSym'].pop(value_pointer, None)
                    conversion_dictionaries['find_parents'].pop(value_pointer, None)
            
            # Remove from temporary_variables and complex_variable_systems
            variable_registry['temporary_variables'].discard(label)
            del variable_registry['complex_variable_systems'][label]
            cleared_complex.append(label)  # Add to cleared list

        # Handle finite algebra systems
        elif label in variable_registry['finite_algebra_systems']:
            family_relatives = variable_registry['finite_algebra_systems'][label].get('family_relatives', {})

            # Remove all related family relatives from _cached_caller_globals
            for relative_label in family_relatives:
                if relative_label in _cached_caller_globals:
                    del _cached_caller_globals[relative_label]

            # Now delete the parent label from finite_algebra_systems
            del variable_registry['finite_algebra_systems'][label]
            cleared_algebras.append(label)  # Add to cleared list

    # Summarize the cleared variable systems
    if cleared_standard and report:
        print(f"Cleared standard variable systems from the DGCV variable management framework: {', '.join(cleared_standard)}")
    if cleared_complex and report:
        print(f"Cleared complex variable systems from the DGCV variable management framework: {', '.join(cleared_complex)}")
    if cleared_algebras and report:
        print(f"Cleared finite algebra systems from the DGCV variable management framework: {', '.join(cleared_algebras)}")

def DGCV_snapshot(style='chalkboard_green', use_latex=False):
    """
    Generate a comprehensive snapshot of DGCV's variable management framework (VMF), including variables,
    algebras, coordinate systems, vector fields, and differential forms.

    Parameters
    ----------
    style : str, optional
        The style options to apply to the summary table. Default theme is 'chalkboard_green'. Use the DGCV function `get_DGCV_themes()` to display a list of other available themes.
    use_latex : bool, optional
        If True, the table will format text using LaTeX for better 
        mathematical display. Default is False.

    Returns
    -------
    DataFrame
        A formatted snapshot displaying the initialized variables, algebras, coordinate systems, 
        vector fields, and differential forms.
    """
    
    variable_registry = get_variable_registry()
    
    # Helper functions to escape Greek letters in LaTeX
    def check_greek(var_name):
        if var_name in greek_letters:
            return True
        else:
            return False
        
    def convert_to_greek(var_name):
        for name, greek in greek_letters.items():
            if var_name.startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    # Wrap content in $$
    def wrap_in_dollars(content):
        return f"${content}$"

    # String formatting for the var names column
    def format_variable_name(var_name, system_type, use_latex=False):
        if system_type == 'standard':
            family_type = variable_registry['standard_variable_systems'][var_name].get('family_type', 'single')
            if family_type == 'multi_index': # patch until multi_index is fully integrated!!!
                family_type = 'tuple'
            family_names = variable_registry['standard_variable_systems'][var_name]['family_names']
            initial_index = variable_registry['standard_variable_systems'][var_name].get('initial_index', 1)
        else:  # system_type == 'complex'
            family_type = variable_registry['complex_variable_systems'][var_name].get('family_type', 'single')
            family_names = variable_registry['complex_variable_systems'][var_name]['family_names'][0]  # First tuple entry
            initial_index = variable_registry['complex_variable_systems'][var_name].get('initial_index', 1)

        # Check if the tuple is empty
        if family_type == 'tuple' and len(family_names) == 0:
            content = f"{var_name} = (empty tuple)"
        elif family_type == 'tuple':
            first_index = initial_index
            last_index = initial_index + len(family_names) - 1
            if use_latex:
                # LaTeX formatting with subscripts and proper index starting from initial_index
                content = f"{convert_to_greek(var_name)} = \\left( {convert_to_greek(var_name)}_{{{first_index}}}, \\ldots, {convert_to_greek(var_name)}_{{{last_index}}} \\right)"
            else:
                # Standard formatting with ellipsis
                content = f"{var_name} = ({family_names[0]}, ..., {family_names[-1]})"
        else:
            if use_latex:
                # Single variable case, apply LaTeX formatting
                content = f"{convert_to_greek(var_name)}"
            else:
                # Single variable case without LaTeX
                content = var_name

        # Wrap content in $$ for LaTeX if needed
        if use_latex:
            content = wrap_in_dollars(content)

        return content
    
    # Helper function to get tuple length or return 1 if not a tuple
    def tupleProcLoc(var_name, system_type):
        if system_type == 'standard':
            return len(variable_registry['standard_variable_systems'][var_name]['family_names'])
        elif system_type == 'complex':
            return len(variable_registry['complex_variable_systems'][var_name]['family_names'][0])
    
    # Retrieve and sort complex and standard variable lists from variable_registry
    complex_vars = sorted(variable_registry['complex_variable_systems'].keys())
    standard_vars = sorted(variable_registry['standard_variable_systems'].keys())
    
    # Retrieve starting index for a given variable
    def retrieveStart(var_name, system_type):
        if system_type == 'standard':
            return variable_registry['standard_variable_systems'][var_name].get('initial_index', 1)
        elif system_type == 'complex':
            return variable_registry['complex_variable_systems'][var_name].get('initial_index', 1)



    # Modular function to generate LaTeX-compatible object strings for VF/DF
    def build_object_string(obj_type, var_name, start_index, tuple_len, system_type, use_latex=False):
        # Access variable_registry using the original variable name (no LaTeX conversion here)
        original_var_name = var_name

        if system_type == 'standard':
            # Check if the standard variable has a differential system
            if variable_registry['standard_variable_systems'][original_var_name].get('differential_system', None) is None:
                content = '----'  # No associated vector fields or differential forms
            elif tuple_len == 1:
                if use_latex:
                    if obj_type == 'd':  # DF case: Use 'd X'
                        content = f"d {convert_to_greek(original_var_name)}"
                    else:  # VF case: Use partial derivative notation
                        content = f"\\frac{{\\partial}}{{\\partial {convert_to_greek(original_var_name)}}}"
                else:
                    content = f"{obj_type}_{original_var_name}"  # Non-LaTeX formatting
            else:
                if use_latex:
                    if obj_type == 'd':  # DF case: Use 'd X'
                        content = f"d {convert_to_greek(original_var_name)}_{{{start_index}}}, \\ldots, d {convert_to_greek(original_var_name)}_{{{int(start_index) + tuple_len - 1}}}"
                    else:  # VF case: Use partial derivative notation
                        content = f"\\frac{{\\partial}}{{\\partial {convert_to_greek(original_var_name)}_{{{start_index}}}}}, \\ldots, \\frac{{\\partial}}{{\\partial {convert_to_greek(original_var_name)}_{{{int(start_index) + tuple_len - 1}}}}}"
                else:
                    content = f"{obj_type}_{original_var_name}{start_index},...,{obj_type}_{original_var_name}{int(start_index) + tuple_len - 1}"
        else:
            # Complex variable system
            if tuple_len == 1:
                if use_latex:
                    if obj_type == 'd':  # DF case: Use 'd X'
                        content = f"d {convert_to_greek(original_var_name)}"
                    else:  # VF case: Use partial derivative notation
                        content = f"\\frac{{\\partial}}{{\\partial {convert_to_greek(original_var_name)}}}"
                else:
                    content = f"{obj_type}_{original_var_name}"
            else:
                if use_latex:
                    if obj_type == 'd':  # DF case: Use 'd X'
                        content = f"d {convert_to_greek(original_var_name)}_{{{start_index}}}, \\ldots, d {convert_to_greek(original_var_name)}_{{{int(start_index) + tuple_len - 1}}}"
                    else:  # VF case: Use partial derivative notation
                        content = f"\\frac{{\\partial}}{{\\partial {convert_to_greek(original_var_name)}_{{{start_index}}}}}, \\ldots, \\frac{{\\partial}}{{\\partial {convert_to_greek(original_var_name)}_{{{int(start_index) + tuple_len - 1}}}}}"
                else:
                    content = f"{obj_type}_{original_var_name}{start_index},...,{obj_type}_{original_var_name}{int(start_index) + tuple_len - 1}"
        
        # Wrap content in $$ for LaTeX if needed, but skip if content is "----"
        if use_latex and content != '----':
            content = wrap_in_dollars(content)
        
        return content



    # Modular function to generate LaTeX-compatible object strings for VF/DF
    def build_object_string_for_complex(obj_type, part_names, family_names, start_index, use_latex=False):
        parts = []
        for part_name, part in zip(part_names, family_names):
            # Ensure start_index is an integer for tuple cases, otherwise handle single variables without subscripts
            if start_index is None or start_index == '':
                is_single = True  # Handle single variables
            else:
                is_single = False  # Handle tuple variables
                start_index = int(start_index)  # Ensure start_index is treated as an integer

            if use_latex:
                # Only remove "BAR" if it's at the start of the part_name (for antiholomorphic variables)
                if part_name.startswith('BAR'):
                    base_var = part_name.replace('BAR', '', 1)  # Remove only the first occurrence of 'BAR'
                    base_var = convert_to_greek(base_var)       # Convert to Greek letter if applicable
                    if is_single:
                        part_str_latex = f"\\overline{{{base_var}}}"  # Single case, fully wrap in \overline{}
                    else:
                        part_str_latex = [f"\\overline{{{base_var}_{{{i}}}}}" for i in range(start_index, start_index + len(part))]
                else:
                    # Standard case with Greek letter conversion
                    if is_single:
                        part_str_latex = f"{convert_to_greek(part_name)}"  # Treat single variables as a string, not a list
                    else:
                        part_str_latex = [f"{convert_to_greek(part_name)}_{{{i}}}" for i in range(start_index, start_index + len(part))]

                if obj_type == 'd':  # Differential Forms (DF)
                    if is_single:
                        part_str = f"d {part_str_latex}"  # No list wrapping for single variables
                    else:
                        part_str = f"d {part_str_latex[0]}, \\ldots, d {part_str_latex[-1]}"  # Use ellipsis for tuples
                else:  # Vector Fields (VF)
                    if is_single:
                        part_str = f"\\frac{{\\partial}}{{\\partial {part_str_latex}}}"  # No list wrapping for single variables
                    else:
                        part_str = f"\\frac{{\\partial}}{{\\partial {part_str_latex[0]}}}, \\ldots, \\frac{{\\partial}}{{\\partial {part_str_latex[-1]}}}"  # Use ellipsis for tuples
                part_str = wrap_in_dollars(part_str)  # Ensure it is wrapped in $$
            else:
                # Non-LaTeX handling
                if is_single:
                    part_str = f"{obj_type}_{convert_to_greek(part_name)}"  # No subscript for single variables
                else:
                    part_str = f"{obj_type}_{convert_to_greek(part_name)}{start_index},...,{obj_type}_{convert_to_greek(part_name)}{int(start_index) + len(part) - 1}"  # Use ellipsis for tuples
            parts.append(part_str)
        
        # Join all parts together
        return ', '.join(parts)


    # Prepare the data for the summary table
    data = [['Number of Var.', 'Real Part', 'Imaginary Part', 'Vector Fields', 'Differential Forms']]  # Top row of with sub-headers


    complex_vars = sorted(variable_registry.get('complex_variable_systems', {}).keys())
    standard_vars = sorted(variable_registry.get('standard_variable_systems', {}).keys())
    finite_algebra_vars = sorted(variable_registry.get('finite_algebra_systems', {}).keys())


    # Process complex variables
    for var_name in complex_vars:
        family_type = variable_registry['complex_variable_systems'][var_name].get('family_type', 'single')
        family_names = variable_registry['complex_variable_systems'][var_name]['family_names']
        initial_index = variable_registry['complex_variable_systems'][var_name].get('initial_index', 1)

        tuple_len = tupleProcLoc(var_name, 'complex')
        start_index = retrieveStart(var_name, 'complex')
        
        # Access family_houses and family_names (holomorphic, antiholomorphic, real, and imaginary parts)
        family_houses = variable_registry['complex_variable_systems'][var_name]['family_houses']
        family_names = variable_registry['complex_variable_systems'][var_name]['family_names']
        
        # Determine if the variable is single or tuple based on family_type
        is_single = variable_registry['complex_variable_systems'][var_name].get('family_type') == 'single'
        
        if use_latex:
            # Vector fields and differential forms for all parts with LaTeX formatting
            vf_str = build_object_string_for_complex('D', family_houses, family_names, start_index, use_latex)
            df_str = build_object_string_for_complex('d', family_houses, family_names, start_index, use_latex)
            
            # Real and Imaginary part formatting with LaTeX
            real_part = f"{convert_to_greek(family_houses[2])} = \\left( {convert_to_greek(family_houses[2])}_{{{start_index}}}, \\ldots, {convert_to_greek(family_houses[2])}_{{{int(start_index) + len(family_names[2]) - 1}}} \\right)" if not is_single else f"{convert_to_greek(family_houses[2])}"
            real_part = wrap_in_dollars(real_part)
            
            imaginary_part = f"{convert_to_greek(family_houses[3])} = \\left( {convert_to_greek(family_houses[3])}_{{{start_index}}}, \\ldots, {convert_to_greek(family_houses[3])}_{{{int(start_index) + len(family_names[3]) - 1}}} \\right)" if not is_single else f"{convert_to_greek(family_houses[3])}"
            imaginary_part = wrap_in_dollars(imaginary_part)
        else:
            # Non-LaTeX formatting for Vector fields and Differential forms
            vf_str = build_object_string_for_complex('D', family_houses, family_names, start_index)
            df_str = build_object_string_for_complex('d', family_houses, family_names, start_index)

            # Applying the ", ... ," tuples format for Real and Imaginary parts
            if len(family_names[2]) > 1:
                real_part = f"{family_houses[2]} = ({family_names[2][0]}, ..., {family_names[2][-1]})"
            else:
                real_part = f"{family_houses[2]} = ({family_names[2][0]})"

            if len(family_names[3]) > 1:
                imaginary_part = f"{family_houses[3]} = ({family_names[3][0]}, ..., {family_names[3][-1]})"
            else:
                imaginary_part = f"{family_houses[3]} = ({family_names[3][0]})"

        # Append to data
        data.append([
            tuple_len,  # Number of variables
            real_part,  # Real part
            imaginary_part,  # Imaginary part
            vf_str,  # Vector fields
            df_str   # Differential forms
        ])


    # Process standard variables
    for var_name in standard_vars:
        family_type = variable_registry['standard_variable_systems'][var_name].get('family_type', 'single')
        family_names = variable_registry['standard_variable_systems'][var_name]['family_names']
        initial_index = variable_registry['standard_variable_systems'][var_name].get('initial_index', 1)


        tuple_len = tupleProcLoc(var_name, 'standard')
        start_index = retrieveStart(var_name, 'standard')
        data.append([
            tuple_len,  # Number of variables
            '----',  # Real part not applicable here
            '----',  # Imaginary part not applicable here
            build_object_string('D', var_name, start_index, tuple_len, 'standard', use_latex),  # Vector fields
            build_object_string('d', var_name, start_index, tuple_len, 'standard', use_latex)  # Differential forms
        ])
    
    # Process finite algebra systems. Plan: place everything in the first column

    # Helper function to process basis labels
    def process_basis_label(label):
        # Use regex to split the name and numeric suffix
        match = re.match(r'(.*?)(\d+)?$', label)
        basis_elem_name = match.group(1)  # Alphabetic part
        basis_elem_number = match.group(2)  # Numeric part, if any
        
        # Remove trailing underscores from the name
        basis_elem_name = basis_elem_name.rstrip('_')
        
        # Build the LaTeX string
        if basis_elem_number:
            return f"{convert_to_greek(basis_elem_name)}_{{{basis_elem_number}}}"
        else:
            return f"{convert_to_greek(basis_elem_name)}"

    index = []
    for var_name in finite_algebra_vars:
        system_data = variable_registry['finite_algebra_systems'][var_name]
        algebra_family_names = system_data.get('family_names', [])
        initial_index = system_data.get('initial_index', 1)

        # Format algebra label and basis (existing logic for parent label)
        if use_latex:
            # Use regex to extract the numeric part only if it's at the end of the string
            match = re.match(r'(.*?)(\d+)?$', var_name)  # Updated regex to handle numbers at the end
            if match:
                algebra_name = match.group(1)  # Alphabetic and mixed part
                algebra_number = match.group(2)  # Numeric part, if any
                
                # Check if the algebra_name consists of only lowercase Latin letters
                if check_greek(algebra_name):
                    algebra_label = f"{convert_to_greek(algebra_name)}"
                elif re.fullmatch(r'[a-z]+', algebra_name):
                    # If it's all lowercase Latin letters, wrap in \mathfrak{}
                    algebra_label = f"\\mathfrak{{{algebra_name}}}"
                else:
                    algebra_label = algebra_name
                
                # Add the subscript if a number exists
                if algebra_number:
                    algebra_label += f"_{{{algebra_number}}}"
            
            # Build the basis string using the first and last labels, formatted
            if len(algebra_family_names) > 1:
                first_basis_label = process_basis_label(algebra_family_names[0])
                last_basis_label = process_basis_label(algebra_family_names[-1])
                basis_str = f"{first_basis_label}, \\ldots , {last_basis_label}"
            else:
                basis_str = process_basis_label(algebra_family_names[0]) if algebra_family_names else ""
            
            formatted_str = f"Algebra: {wrap_in_dollars(algebra_label)} \\\\ Basis: {wrap_in_dollars(basis_str)}"
        
        else:
            # Non-LaTeX case
            algebra_label = var_name
            if len(algebra_family_names) > 1:
                basis_str = f"{algebra_family_names[0]}, ..., {algebra_family_names[-1]}"
            else:
                basis_str = algebra_family_names[0] if algebra_family_names else ""
            
            formatted_str = f"Algebra: {algebra_label}<br>Basis: {basis_str}"

        # Add a single row to the data part for each finite_algebra_systems element
        data.append([
            len(algebra_family_names),  # Number of variables
            '----',  # Leave "Real Part" empty
            '----',  # Leave "Imaginary Part" empty
            '----',  # Leave "Vector Fields" empty
            '----'   # Leave "Differential Forms" empty
        ])

        # Add the formatted_str to the index for this algebra
        index.append((formatted_str, ''))


    # Generate index (row labels) with a MultiIndex to display 'Variable Name(s)' at the top
    index = MultiIndex.from_tuples(
        [('Coordinate/Algebra Name(s)', '')] + 
        [(format_variable_name(var_name, 'complex', use_latex), '') for var_name in complex_vars] + 
        [(format_variable_name(var_name, 'standard', use_latex), '') for var_name in standard_vars] +
        index  # Add the finite algebra index entries here
    )
    # Define your columns (these are the original column headers)
    column_headers = ['', '', '', '', ''] # blank placeholders

    # Create DataFrame
    table = DataFrame(data=data, index=index, columns=column_headers)
    table_header='Initialized Coordinate Systems and Algebras'

    table.columns=MultiIndex.from_product([[table_header], table.columns])

    # Fetch the styles from the style guide
    table_styles = get_style(style)

    # Apply the styles to the table
    table = table.style.set_table_styles(table_styles)

    # Add the caption (subheader) after setting the styles
    table = table.set_caption("(summarizes objects within the scope of DGCV's variable management framework)")


    return table

def variableSummary(*args, **kwargs):
    warnings.warn(
        "variableSummary() is deprecated and will be removed in a future version. "
        "Please use DGCV_snapshot() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return DGCV_snapshot(*args, **kwargs)
