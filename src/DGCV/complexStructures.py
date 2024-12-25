"""
DGCV: Differential Geometry with Complex Variables

This module provides tools uniquely relevant for complex differential geometry within the DGCV package. It includes Dolbeault operators (Del and DelBar) and a class for constructing and analyzing Kähler structures.

Key Functions:
    - Del(): Applies the holomorphic Dolbeault operator ∂ to a differential form or scalar.
    - DelBar(): Applies the antiholomorphic Dolbeault operator ∂̅ to a differential form or scalar.

Key Classes:
    - KahlerStructure: Represents a Kähler structure, with properties and attributes to compute many of their invariants. 

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy

License:
    MIT License
"""



############## dependencies

from sympy import sympify
from .config import get_variable_registry, _cached_caller_globals
from .vectorFieldsAndDifferentialForms import makeZeroForm
from .classesAndVariables import *
from .RiemannianGeometry import *

############## Dolbeault operators

def Del(arg1):
    """
    Applies the holomorphic Dolbeault operator ∂ (Del) to a differential form or scalar.
    
    The Del operator takes a differential form or scalar and returns the result of applying the 
    exterior derivative with respect to the holomorphic coordinates.
    
    Parameters:
    -----------
    arg1 : DFClass or sympy.Expr
        A differential form (DFClass) with DGCVType='complex', or a scalar (interpreted as a zero-form).
    
    Returns:
    --------
    DFClass
        The result of applying the Del operator, which is a differential form.
    
    Raises:
    -------
    TypeError
        If the input is not a DFClass or scalar (sympy.Expr), or if the DFClass has DGCVType='standard'.
    """
    variable_registry = get_variable_registry()
    # Ensure arg1 is a DFClass or convert it into a zero-form
    if not isinstance(arg1, DFClass) and isinstance(arg1, (int,float,sympy.Expr)):
        arg1=allToSym(arg1)
        varSpace=tuple([j for j in sympify(arg1).free_symbols if variable_registry['conversion_dictionaries']['symToReal']])
        intermediateDF = makeZeroForm(arg1,varSpace=varSpace, DGCVType='complex',default_var_format='complex')
        arg1 = makeZeroForm(arg1,intermediateDF.compVarSpace, DGCVType='complex',default_var_format='complex')

    elif isinstance(arg1,DFClass):
        if arg1.DGCVType == 'standard':
            raise TypeError('`Del` only operates on DFClass differential forms with DGCVType=\'complex\' (and scalars like sympy.Expr, which it interprets as 0-forms). Tip: set \'complex=True\' in the `createVariables` variable creation function to initialize a coframe with `DGCVType=\'complex\'.`')
        else:
            arg1=realToSym(symToReal(arg1))
    else:
        raise TypeError('`Del` only operate on differential forms, DFClass (and scalars like sympy.Expr, which it interprets as 0-forms).')

    # Helper function to compute the exterior derivative of a zero-form
    def DelOfZeroForm(arg1):
        HVSpace = arg1.holVarSpace
        sparseDataLoc = {(j,): diff(allToSym(arg1.coeffsInKFormBasis[0]), HVSpace[j]) for j in range(len(HVSpace))}        
        return DFClass(arg1.varSpace, sparseDataLoc, 1, DGCVType='complex')

    # Handle zero-forms
    if arg1.degree == 0:
        return DelOfZeroForm(arg1)

    # Handle higher-degree forms
    minDataLoc = arg1.DFClassDataMinimal
    coeffsTo1Forms = [DelOfZeroForm(makeZeroForm(j[1], varSpace=arg1.varSpace, DGCVType='complex')) for j in minDataLoc]
    # Construct the corresponding basis k-forms
    basisOfCoeffs = [
        DFClass(
            arg1.varSpace, 
            {tuple(j[0]):1}, 
            arg1.degree, 
            DGCVType='complex'
        ) for j in minDataLoc
    ]
    # Multiply the one-forms by the basis and sum them
    return addDF(*[coeffsTo1Forms[j] * basisOfCoeffs[j] for j in range(len(minDataLoc))])

def DelBar(arg1):
    """
    Applies the antiholomorphic Dolbeault operator ∂̅ (DelBar) to a differential form or scalar.
    
    The DelBar operator takes a differential form or scalar and returns the result of applying the 
    exterior derivative with respect to the antiholomorphic coordinates.
    
    Parameters:
    -----------
    arg1 : DFClass or sympy.Expr
        A differential form (DFClass) with DGCVType='complex', or a scalar (interpreted as a zero-form).
    
    Returns:
    --------
    DFClass
        The result of applying the DelBar operator, which is a differential form.
    
    Raises:
    -------
    TypeError
        If the input is not a DFClass or scalar (sympy.Expr), or if the DFClass has DGCVType='standard'.
    """
    variable_registry = get_variable_registry()
    # Ensure arg1 is a DFClass or convert it into a zero-form
    if not isinstance(arg1, DFClass) and isinstance(arg1, (int,float,sympy.Expr)):
        arg1=allToSym(arg1)
        varSpace=tuple([j for j in sympify(arg1).free_symbols if variable_registry['conversion_dictionaries']['symToReal']])
        intermediateDF = makeZeroForm(arg1,varSpace=varSpace, DGCVType='complex',default_var_format='complex')
        arg1 = makeZeroForm(arg1,intermediateDF.compVarSpace, DGCVType='complex',default_var_format='complex')

    if isinstance(arg1,DFClass):
        if arg1.DGCVType == 'standard':
            raise TypeError('`DelBar` only operates on DFClass differential forms with DGCVType=\'complex\' (and scalars like sympy.Expr, which it interprets as 0-forms). Tip: set \'complex=True\' in the `createVariables` variable creation function to initialize a coframe with `DGCVType=\'complex\'.`')
        else:
            arg1=allToSym(symToReal(arg1))
    else:
        raise TypeError('`DelBar` only operate on differential forms, DFClass (and scalars like sympy.Expr, which it interprets as 0-forms).')

    # Helper function to compute the exterior derivative of a zero-form
    def DelBarOfZeroForm(arg1):
        AHVspace = arg1.antiholVarSpace
        CDim = len(AHVspace)
        sparseDataLoc = {(j+CDim,): diff(allToSym(arg1.coeffsInKFormBasis[0]), AHVspace[j]) for j in range(CDim)}        
        return DFClass(arg1.varSpace, sparseDataLoc, 1, DGCVType='complex')

    # Handle zero-forms
    if arg1.degree == 0:
        return DelBarOfZeroForm(arg1)

    # Handle higher-degree forms
    minDataLoc = arg1.DFClassDataMinimal
    coeffsTo1Forms = [DelBarOfZeroForm(makeZeroForm(j[1], varSpace=arg1.varSpace, DGCVType='complex')) for j in minDataLoc]
    # Construct the corresponding basis k-forms
    basisOfCoeffs = [
        DFClass(
            arg1.varSpace, 
            {tuple(j[0]):1}, 
            arg1.degree, 
            DGCVType='complex'
        ) for j in minDataLoc
    ]
    
    # Multiply the one-forms by the basis and sum them
    return addDF(*[coeffsTo1Forms[j] * basisOfCoeffs[j] for j in range(len(minDataLoc))])

############## Kahler geometry

class KahlerStructure(Basic):
    """
    Represents a Kähler structure, including its metric, , and Bochner tensor.
    
    The Kähler structure is defined by a symplectic form (Kähler form) and a coordinate space. The class 
    provides methods to compute various geometric objects associated with Kähler manifolds, including the 
    its metric, Holomorphic Riemann and Ricci Curvatures, and the Bochner tensor.
    
    Parameters:
    -----------
    varSpace : tuple of sympy.Symbol
        A tuple of variables from DGCV's complex coordinate systems representing the coordinate space of the manifold.
    kahlerForm : DFClass
        A Kähler form representing the symplectic structure.
    
    Attributes:
    -----------
    metric : metricClass
        The Riemannian metric associated with the Kähler structure.
    holRiemann : TFClass
        The (0,4) Riemann curvature tensor with the complex structure operator hooked into it second and fourth positions.
    holRicci : TFClass
        The Ricci curvature tensor with the complex structure operator hooked into its second position
    Bochner : TFClass
        The Bochner tensor in \( T^{1,0}M \otimes T^{0,1}M \otimes T^{1,0}M \otimes T^{0,1}M \).
    
    Raises:
    -------
    TypeError
        If the variable space includes a mixture of real and holomorphic coordinates.
    """

    def __new__(cls, varSpace, kahlerForm):
        # Call Basic.__new__ with only the positional arguments
        obj = Basic.__new__(cls, varSpace, kahlerForm)
        return obj

    def __init__(self, varSpace, kahlerForm):
        variable_registry=get_variable_registry()
        if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in varSpace):
            self._varSpace_type= 'real'
        elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in varSpace):
            self._varSpace_type = 'complex'
        else:
            raise TypeError('The variable space given to the Kahler structure initialer has a mixture of real and holomorphic coordinates, which is not a supported data format.')
        variableSpaces=KahlerStructure.validate(varSpace)
        self.realVarSpace=variableSpaces[0]
        self.compVarSpace=variableSpaces[1]+variableSpaces[2]
        self.holVarSpace=variableSpaces[1]
        self.antiholVarSpace=variableSpaces[2]
        if self._varSpace_type == 'real':
            self.varSpace=self.realVarSpace
        else:
            self.varSpace=self.compVarSpace
        self.kahlerForm=changeDFBasis(kahlerForm,self.varSpace)
        VFBasis=[]
        for var in self.varSpace:
            varStr = str(var)
            for parent in variable_registry['complex_variable_systems']:
                if varStr in variable_registry['complex_variable_systems'][parent]['variable_relatives']:
                    VFBasis = VFBasis + [variable_registry['complex_variable_systems'][parent]['variable_relatives'][varStr]['VFClass']]
        self.coor_frame = VFBasis
        if self._varSpace_type == 'real':
            self.coor_frame_real = self.coor_frame
            VFBasisHol=[]
            for var in self.compVarSpace:
                varStr = str(var)
                for parent in variable_registry['complex_variable_systems']:
                    if varStr in variable_registry['complex_variable_systems'][parent]['variable_relatives']:
                        VFBasisHol = VFBasisHol + [variable_registry['complex_variable_systems'][parent]['variable_relatives'][varStr]['VFClass']]
            self.coor_frame_complex = VFBasisHol
        else:
            self.coor_frame_complex = self.coor_frame
            VFBasisReal=[]
            for var in self.compVarSpace:
                varStr = str(var)
                for parent in variable_registry['complex_variable_systems']:
                    if varStr in variable_registry['complex_variable_systems'][parent]['variable_relatives']:
                        VFBasisReal = VFBasisReal + [variable_registry['complex_variable_systems'][parent]['variable_relatives'][varStr]['VFClass']]
            self.coor_frame_real = VFBasisReal

                    
        self._metric = None
        self._is_closed = None
        self._holRiemann = None # holomorphic Riemann tensor
        self._holRicci = None # holomorphic Ricci tensor
        self._Bochner = None # Bochner tensor

    @staticmethod
    def validate(varSpace): # validates and organizes given varspace
        """
        Validates and organizes a given variable space into real, holomorphic, and antiholomorphic components.
        
        This method checks the input variable space and categorizes the variables into real, holomorphic, 
        and antiholomorphic components, based on the relationships between the variables.
        
        Parameters:
        -----------
        varSpace : tuple of sympy.Symbol
            A tuple of complex variables representing the coordinate space.
        
        Returns:
        --------
        list of tuples
            A list containing the real and imaginary variable spaces, holomorphic variable space, and 
            antiholomorphic variable space.
        """
        variable_registry=get_variable_registry()
        CVS=variable_registry['complex_variable_systems']
        if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in varSpace):
            _varSpace_type= 'real'
        elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in varSpace):
            _varSpace_type = 'complex'

        exhaust1 = list(varSpace)
        populate = {'holVarDict':dict(),'antiholVarDict':dict(),'realVarDict':dict(),'imVarDict':dict()}
        if _varSpace_type == 'real':
            for var in varSpace:
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
            for var in varSpace:
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
        _realVarSpace = tuple(populate['realVarDict'].keys())
        _holVarSpace = tuple(populate['holVarDict'].keys())
        _antiholVarSpace = tuple(populate['antiholVarDict'].keys())
        _imVarSpace = tuple(populate['imVarDict'].keys())
        
        return [_realVarSpace+_imVarSpace,_holVarSpace,_antiholVarSpace]

    @property
    def metric(self):
        if self._metric == None:
            if self._is_closed==False:
                warnings.warn('The provided symplectic form does not define a Kahler structure, so the associated metric tensor may not actually describe a metric.')
  
            coeffData={(j,k): self.kahlerForm(self.coor_frame[j],complex_struct_op(self.coor_frame[k])) for j in range(len(self.coor_frame)) for k in range(j,len(self.coor_frame))}
            self._metric = metricClass(STFClass(self.varSpace,coeffData,2,DGCVType='complex'))
        return self._metric
    
    @property
    def holRiemann(self):
        """
        The holomorphic Riemann curvature tensor, defined by hooking the complex structure operator into the second and fourth positions of the Kahler structure's metric's (0,4)-type Riemann curvature tensor.
        """
        if self._holRiemann == None:
            
            dim = len(self.coor_frame)
            R = self.metric.RiemannCurvature

            VFBasis = self.coor_frame
            
            def entry_rule(a,b,c,d):
                return simplify(R(a,complex_struct_op(b),c,complex_struct_op(d)))

            coeffData={(j, k, l, m): entry_rule(VFBasis[j],VFBasis[k],VFBasis[l],VFBasis[m]) for j in range(dim) for k in range(j,dim) for l in range(dim) for m in range(dim)}
            self._holRiemann = TFClass(self.varSpace,coeffData,4,DGCVType='complex')
        return self._holRiemann

    @property
    def holRicci(self):
        """
        The holomorphic Ricci curvature tensor, defined by hooking the complex structure operator into the second position of the Kahler structure's metric's Ricci curvature tensor.
        """
        if self._holRicci == None:
            
            dim = len(self.coor_frame)
            Ric = self.metric.RicciTensor

            VFBasis = self.coor_frame
            
            def entry_rule(a,b):
                return simplify(Ric(a,complex_struct_op(b)))

            coeffData={(j, k): entry_rule(VFBasis[j],VFBasis[k]) for j in range(dim) for k in range(j,dim)}
            self._holRicci = TFClass(self.varSpace,coeffData,2,DGCVType='complex')
        return self._holRicci


    @property
    def Bochner(self):
        """
        The Bochner curvature tensor in $T^{1,0}M\otimes T^{0,1}M\otimes T^{1,0}M\otimes T^{0,1}M$.
        """
        if self._Bochner == None:
            
            VFBasis = self.coor_frame_complex
            dim = len(VFBasis)
            compDim = int(Rational(dim,2))

            g = self.metric.SymTensorField
            R = self.metric.RiemannCurvature
            Ric = self.metric.RicciTensor
            S = self.metric.scalarCurvature

            def entry_rule(j,h,l,k):
                term1 = R(j,h,l,k)
                term2 = Rational(1,compDim+2)*((g(j,k))*(Ric(l,h))+(g(l,k))*(Ric(j,h))+(g(l,h))*(Ric(j,k))+(g(j,h))*(Ric(l,k)))
                term3 = S*Rational(1,2*(compDim+1)*(compDim+2))*(g(l,h)*g(j,k)+g(j,h)*g(l,k))
                return simplify(term1+term2-term3)

            coeffData={(j, k, l, m): entry_rule(VFBasis[j],VFBasis[k],VFBasis[l],VFBasis[m]) for j in range(compDim) for k in range(compDim,dim) for l in range(compDim) for m in range(compDim, dim)}
            self._Bochner = TFClass(self.varSpace,coeffData,4,DGCVType='complex',simplifyKW={'simplify_rule': None, 'simplify_ignore_list': None, 'preferred_basis_element': (0,compDim,0,compDim)})

        return self._Bochner
    
    @property
    def is_closed(self):
        if self._is_closed == None:
            self._is_closed = self.kahlerForm.is_zero()
        return self._is_closed
