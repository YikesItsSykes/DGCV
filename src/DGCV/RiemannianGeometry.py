"""
DGCV: Differential Geometry with Complex Variables

This module provides tools for Riemannian geometry within the DGCV package. It includes 
functions and classes for defining and manipulating metrics, Christoffel symbols, curvature tensors, 
and Levi-Civita connections.

Key Classes:
    - metricClass: Represents a Riemannian metric and provides methods to compute Christoffel symbols, 
      Riemann curvature, Ricci curvature, scalar curvature, and Weyl curvature.
    - LeviCivitaConnectionClass: Defines a Levi-Civita connection based on a set of Christoffel symbols 
      of the second kind.
    
Key Functions:
    - metric_from_matrix(): Creates a metricClass object from a given coordinate space and matrix representation 
      of the metric.

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy

License:
    MIT License
"""



from sympy import Matrix,ImmutableSparseNDimArray
from .classesAndVariables import *



# Reimannian cetric class
class metricClass(Basic):
    
    def __new__(cls, STF):
        # Call Basic.__new__ with only the positional arguments
        obj = Basic.__new__(cls, STF)
        return obj

    def __init__(self, STF):
        if not isinstance(STF,STFClass):
            raise TypeError('The `metric` class initializer expects a symmtric tensor field class (STFClass) object in its first argument.')
        if STF.degree != 2:
            raise TypeError('The `metric` class initializer expects a symmtric tensor field class (STFClass) object of degree 2 in its first argument.')            
        self.varSpace = STF.varSpace
        self.degree = STF.degree
        self.simplifyKW = STF.simplifyKW
        self.DGCVType = STF.DGCVType
        self.SymTensorField = STF
        self._varSpace_type = STF._varSpace_type

        self.MetricDataDict = STF.STFClassDataDict
        self.MetricDataMinimal = STF.STFClassDataMinimal
        self.coeffsInKFormBasis = STF.coeffsInKFormBasis
        self.kFormBasisGenerators = STF.kFormBasisGenerators
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._coeff_dicts = None
        self._coeffArray = None
        self._matrixRep = None
        self._matrixRepInv = None
        self._matrixRep_real = None
        self._matrixRep_hol = None
        self._matrixRep_sym = None
        self._CFFK = None # Christoffel cache
        self._CFSK = None # Christoffel cache
        self._RCT13 = None # Riemann Curvature cache
        self._RCT04 = None # Riemann Curvature cache
        self._Ricci = None # Ricci tensor cache
        self._SCT = None # Scalar curvature tensor cache
        self._tracelessRicci = None # traceless Ricci curvature cache
        self._Weyl = None # Weyl curvature cache
        self._Einstein = None # Einstein curvature cache


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
                    populate['realCoeffDataDict'] = [self.varSpace,self.MetricClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,{0:self.MetricClassDataDict[0]}]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.MetricClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,{0:self.MetricClassDataDict[0]}]
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
                    return [tuple(sorted(arg[0])),permSign(arg[0])*arg[1]]

                otherDict=dict()
                for j in self.MetricDataMinimal:
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
                    populate['realCoeffDataDict'] = [self.varSpace,self.MetricClassDataDict]
                    populate['compCoeffDataDict'] = [self._holVarSpace+self._antiholVarSpace,otherDict]
                else:
                    populate['compCoeffDataDict'] = [self.varSpace,self.MetricClassDataDict]
                    populate['realCoeffDataDict'] = [self._realVarSpace+self._imVarSpace,otherDict]

            self._coeff_dicts = populate

            return populate
        else:
            return self._coeff_dicts

    @property
    def coeffArray(self):
        def entry_rule(indexTuple):
            sortedTuple = tuple(sorted(indexTuple))
            if sortedTuple in self.MetricClassDataDict:
                return self.MetricClassDataDict[sortedTuple]
            else:
                return 0

        def generate_indices(shape):
            """Recursively generates all index tuples for an arbitrary dimensional array."""
            if len(shape) == 1:
                return [(i,) for i in range(shape[0])]
            else:
                return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]

        shape = (len(self.varSpace),) * self.degree
        sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}

        # Convert the sparse array to a list of lists (if 2D)
        return ImmutableSparseNDimArray(sparse_data, shape)


    @property
    def matrixRep(self):
        if self._matrixRep==None:
            self._matrixRep = Matrix(self.SymTensorField.coeffArray)
            return self._matrixRep
        else:
            return self._matrixRep

    @property
    def matrixRepInv(self):
        if self._matrixRepInv == None:
            self._matrixRepInv = self.matrixRep.inv()
            return self._matrixRepInv
        else:
            return self._matrixRepInv

    @property 
    def matrixRep_real(self):
        if self._matrixRep_real == None:
            self._matrixRep_real = simplify(self.matrixRep.applyfunc(allToReal))
            return self._matrixRep_real
        else:
            return self._matrixRep_real

    @property 
    def matrixRep_hol(self):
        if self._matrixRep_hol == None:
            self._matrixRep_hol = simplify(self.matrixRep.applyfunc(allToHol))
            return self._matrixRep_hol
        else:
            return self._matrixRep_hol

    @property 
    def matrixRep_sym(self):
        if self._matrixRep_sym == None:
            self._matrixRep_sym = simplify(self.matrixRep.applyfunc(allToSym))
            return self._matrixRep_sym
        else:
            return self._matrixRep_sym

    @property
    def Christoffel_symbols_of_the_second_kind(self):
        """
        the last index is the raised one...
        """
        if self._CFSK==None:
            mat = self.matrixRep
            matInv = self.matrixRepInv
            vs=self.varSpace
            dim=len(vs)
            def entry_rule(indexTuple):
                return simplify(sum([Rational(1,2)*matInv[indexTuple[2],j]*(diff(mat[j,indexTuple[1]],vs[indexTuple[0]])+diff(mat[indexTuple[0],j],vs[indexTuple[1]])-diff(mat[indexTuple[0],indexTuple[1]],vs[j])) for j in range(dim)]))
            
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (dim,)*3
            sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
            self._CFSK = ImmutableSparseNDimArray(sparse_data, shape)
        return self._CFSK
    
    @property
    def Christoffel_symbols_of_the_first_kind(self):
        if self._CFFK==None:
            mat = self.matrixRep
            vs=self.varSpace
            dim=len(vs)
            def entry_rule(indexTuple):
                return Rational(1,2)*(diff(mat[indexTuple[2],indexTuple[0]],vs[indexTuple[1]])+diff(mat[indexTuple[2],indexTuple[1]],vs[indexTuple[0]])-diff(mat[indexTuple[0],indexTuple[1]],vs[indexTuple[2]]))
            
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (dim,)*3
            sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}

            self._CFFK=ImmutableSparseNDimArray(sparse_data, shape)
        return self._CFFK

    @property
    def RiemannCurvature_1_3_type(self):
        """"
        This is the (1,3)-tensor version of the Riemann tensor. The last index is the raised one...
        """
        if self._RCT13==None:
            Gamma=self.Christoffel_symbols_of_the_second_kind
            vs=self.varSpace
            dim=len(vs)
            def entry_rule(indexTuple):
                # term1=diff(Gamma[indexTuple[2],indexTuple[0],indexTuple[3]],vs[indexTuple[1]])-diff(Gamma[indexTuple[1],indexTuple[0],indexTuple[3]],vs[indexTuple[2]])
                # term2=sum([Gamma[indexTuple[2],indexTuple[0],p]*Gamma[indexTuple[1],p,indexTuple[3]]-Gamma[indexTuple[1],indexTuple[0],p]*Gamma[indexTuple[2],p,indexTuple[3]] for p in range(dim)])
                # return simplify(term1-term2)
                r=indexTuple[0]
                j=indexTuple[1]
                k=indexTuple[2]
                i=indexTuple[3]
                term=diff(Gamma[j,r,i],vs[k])-diff(Gamma[k,r,i],vs[j])+sum([Gamma[k,s,i]*Gamma[j,r,s] for s in range(len(vs))])-sum([Gamma[j,s,i]*Gamma[k,r,s] for s in range(len(vs))])
                return simplify(term)
            
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (dim,)*4
            sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
            self._RCT13 = ImmutableSparseNDimArray(sparse_data, shape)
        return self._RCT13

    @property
    def RiemannCurvature(self):
        """
        This is the (0,4)-tensor version of the Riemann tensor. The last index is the raised one...
        """
        if self._RCT04==None:
            mat = self.matrixRep
            RCT = self.RiemannCurvature_1_3_type
            dim=len(self.varSpace)
            def entry_rule(indexTuple):
                return simplify(-sum([mat[indexTuple[0],p]*RCT[indexTuple[1],indexTuple[2],indexTuple[3],p] for p in range(dim)]))
            
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (dim,)*4
            sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
            self._RCT04 = TFClass(self.varSpace, sparse_data, 4, DGCVType=self.DGCVType)
        return self._RCT04
    
    @property
    def RicciTensor(self):
        
        if self._Ricci==None:
            mat = self.matrixRepInv
            RCT = self.RiemannCurvature.coeffArray
            dim=len(self.varSpace)
            def entry_rule(indexTuple):
                return simplify(sum([mat[p,q]*RCT[indexTuple[0],p,indexTuple[1],q] for p in range(dim) for q in range(dim)]))
            
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (dim,)*2
            sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
            self._Ricci = STFClass(self.varSpace,sparse_data,self.degree,DGCVType=self.DGCVType,simplifyKW=self.simplifyKW)
        return self._Ricci

    @property
    def scalarCurvature(self):
        
        if self._SCT==None:
            mat = self.matrixRepInv
            RCT = self.RicciTensor.coeffArray
            dim=len(self.varSpace)
            self._SCT = simplify(sum([mat[p,q]*RCT[p,q] for p in range(dim) for q in range(dim)]))
        return self._SCT

    @property
    def tracelessRicci(self):
        
        if self._tracelessRicci==None:
            Ric = self.RicciTensor
            SC = self.scalarCurvature
            self._tracelessRicci = Ric-Rational(1,len(self.varSpace))*SC*self.SymTensorField
        return self._Ricci

    @property
    def WeylCurvature(self):
        """
        The Weyl curvature tensor.
        """
        if self._Weyl==None:
            g = self.matrixRep
            RCT = self.RiemannCurvature.coeffArray
            R = self.scalarCurvature
            TR = self.tracelessRicci.coeffArray
            dim=len(self.varSpace)
            def entry_rule(iT):
                if dim<3:
                    return 0
                else:
                    term1 = RCT[iT[0],iT[1],iT[2],iT[3]]-Rational(1,(dim-1)*(dim))*R*(g[iT[0],iT[2]]*g[iT[1],iT[3]]-g[iT[0],iT[3]]*g[iT[1],iT[2]])
                    term2 = Rational(1,dim-2)*(TR[iT[0],iT[2]]*g[iT[1],iT[3]]-TR[iT[1],iT[2]]*g[iT[0],iT[3]]+TR[iT[1],iT[3]]*g[iT[0],iT[2]]-TR[iT[0],iT[3]]*g[iT[1],iT[2]])
                    return simplify(term1-term2)
            
            def generate_indices(shape):
                """Recursively generates all index tuples for an arbitrary dimensional array."""
                if len(shape) == 1:
                    return [(i,) for i in range(shape[0])]
                else:
                    return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
            shape = (dim,)*4
            sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
            self._Weyl = TFClass(self.varSpace, sparse_data, 4, DGCVType=self.DGCVType)
        return self._Weyl

    @property
    def Einstein_tensor(self):
        
        if self._tracelessRicci==None:
            Ric = self.RicciTensor
            SC = self.scalarCurvature
            self._tracelessRicci = Ric-Rational(1,2)*SC*self.SymTensorField
        return self._Ricci

    def sectionalCurvature(self,vf1,vf2):
        RCT = self.RiemannCurvature
        value = self.RiemannCurvature(vf1,vf2,vf1,vf2)/(self.SymTensorField(vf1,vf1)*self.SymTensorField(vf2,vf2)-(self.SymTensorField(vf1,vf2))**2)
        return value    

    def __repr__(self):
        return self._generate_representation()

    def __str__(self):
        return self._generate_representation()

    def _sympystr(self, printer):
        """
        Overrides SymPy's default string formatting method.
        This ensures that custom __repr__ is used when printing metric class objects.
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
            return str(self.MetricDataMinimal[0][1])

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
        metric
            A new DFClass instance with updated simplification settings.
        """
        if format_type not in {None, 'holomorphic','real', 'symbolic_conjugate'}:
            warnings.warn('simplify_format() recieved an unsupported first argument. Try None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\' instead.')
        return metricClass(self.SymTensorField.simplify_format(format_type=format_type,skipVar=skipVar))

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
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.MetricDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='holomorphic':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToHol(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.MetricDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='real':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToReal(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.MetricDataDict.items()}
        elif self.simplifyKW['simplify_rule']=='symbolic_conjugate':
            # Simplify each element in the coeffs list
            simplified_coeffs = {a:simplify(allToSym(b,skipVar=self.simplifyKW['simplify_ignore_list']), **kwargs) for a,b in self.MetricDataDict.items()}
        else:
            warnings.warn('_eval_simplify recieved an unsupported STFClass.simplifyKW[\'simplify_rule\']. It is recommend to only set the simplifyKW[\'simplify_rule\'] attribute to None, \'holomorphic\', \'real\',  or \'symbolic_conjugate\'.')
            simplified_coeffs = {a:simplify(b, **kwargs) for a,b in self.MetricDataDict.items()}

        # Return a new instance of DFClass with simplified coeffs

        # Return a new instance of DFClass with simplified coeffs
        return metricClass(STFClass(self.varSpace,simplified_coeffs,self.degree,DGCVType=self.DGCVType,simplifyKW=self.simplifyKW))

    def subs(self,subsData):
        return metricClass(self.SymTensorField.subs(subsData))

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
                return f"\overline{{{label_part}_{{{number_part}}}}}" if number_part else label_part
        else:
            # Return LaTeX for non-BAR variables
            return latex(var)

    def _repr_latex_(self):
        if self.degree == 0:
            return f"${self._process_var_label(self.coeffsInKFormBasis[0])}$"
        
        terms = []
        varLabels = [[self.varSpace[k] for k in j[0]] for j in self.MetricDataMinimal]
        for coeff, var_list in zip(self.coeffsInKFormBasis, varLabels):
            if coeff == 0:
                continue
            latex_vars = [f"d {self._process_var_label(var)}" for var in var_list]
            differential_form = " \\odot ".join(latex_vars)
            latex_coeff = self._format_coeff(coeff)
            term = f"{latex_coeff}{differential_form}"
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

class LeviCivitaConnectionClass(Basic):

    def __new__(cls, varSpace, Christoffel_symbols_of_the_second_kind, variable_handling_default='standard'):
        # Call Basic.__new__ with only the positional arguments
        obj = Basic.__new__(cls, varSpace,Christoffel_symbols_of_the_second_kind)
        return obj

    from sympy.tensor.array import ImmutableSparseNDimArray

    def __init__(self, varSpace, Christoffel_symbols_of_the_second_kind, variable_handling_default='standard'):
        # Attempt to convert the input to an ImmutableSparseNDimArray
        if not isinstance(Christoffel_symbols_of_the_second_kind, ImmutableSparseNDimArray):
            try:
                Christoffel_symbols_of_the_second_kind = ImmutableSparseNDimArray(Christoffel_symbols_of_the_second_kind)
            except Exception as e:
                raise TypeError(
                    'The `LeviCivitaConnection` class initializer expects an ImmutableSparseNDimArray '
                    'or array-like data that can be converted to one. '
                    f'Failed to convert the input: {e}'
                )

        # Check the shape of the Christoffel symbols
        shapeCheck = Christoffel_symbols_of_the_second_kind.shape
        if len(shapeCheck) != 3 or len(set(shapeCheck)) != 1:
            raise TypeError(
                'The `LeviCivitaConnection` class initializer was given array data for the '
                '`Christoffel_symbols_of_the_second_kind` of invalid shape.'
            )

        # Check that the coordinate space matches the Christoffel symbols' size
        if len(varSpace) != shapeCheck[0]:
            raise TypeError(
                'The `LeviCivitaConnection` class initializer was given array data for the '
                '`Christoffel_symbols_of_the_second_kind` of invalid size relative to the provided coordinate space. '
                'Number of coordinates must match the Christoffel symbols\' index range.'
            )

        # Handle variable type based on the default handling mode
        if variable_handling_default == 'complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in varSpace):
                self._varSpace_type = 'real'
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in varSpace):
                self._varSpace_type = 'complex'
            else:
                raise KeyError(
                    'To initialize a `LeviCivitaConnectionClass` instance with variable_handling_default=\'complex\', '
                    '`varSpace` must contain only variables from DGCV\'s complex variables systems, and all variables in '
                    '`varSpace` must be simultaneously among the real and imaginary types, or simultaneously among the '
                    'holomorphic and antiholomorphic types. Use `complexVarProc` to easily create DGCV complex variable systems.'
                )
        else:
            self._varSpace_type = 'standard'

        # Assign attributes
        self.Christoffel_symbols = Christoffel_symbols_of_the_second_kind
        self.varSpace = varSpace
        
    def __call__(self,vf1,vf2):
        if not all([isinstance(vf1,VFClass), isinstance(vf2,VFClass)]):
            raise TypeError('`LeviCivitaConnectionClass` only operates on pairs of VFClass objects.')
        dimLoc=len(self.varSpace)
        def _coeff(VF1,VF2,l):
            term1=sum([diff(VF2.coeffs[l],self.varSpace[j])*VF1.coeffs[j] for j in range(dimLoc)])
            term2=sum([self.Christoffel_symbols[j,k,l]*VF1.coeffs[j]*VF2.coeffs[k] for j in range(dimLoc) for k in range(dimLoc)])
            return term1+term2

        if self._varSpace_type=='standard':
            vf1=changeVFBasis(vf1,self.varSpace)
            vf2=changeVFBasis(vf2,self.varSpace)
            newCoeffs=[_coeff(vf1,vf2,l) for l in range(dimLoc)]
            return VFClass(self.varSpace,newCoeffs)
        elif self._varSpace_type=='real':
            vf1=changeVFBasis(allToReal(vf1),self.varSpace)
            vf2=changeVFBasis(allToReal(vf2),self.varSpace)
            newCoeffs=[_coeff(vf1,vf2,l) for l in range(dimLoc)]
            return VFClass(self.varSpace,newCoeffs,DGCVType='complex')
        elif self._varSpace_type=='complex':
            vf1=changeVFBasis(allToSym(vf1),self.varSpace)
            vf2=changeVFBasis(allToSym(vf2),self.varSpace)
            newCoeffs=[_coeff(vf1,vf2,l) for l in range(dimLoc)]
            return VFClass(self.varSpace,newCoeffs,DGCVType='complex')


def metric_from_matrix(coordinates,matrix):

    if isinstance(matrix,list):
        if all(isinstance(j,list) for j in matrix):
            if all(len(j)==len(matrix[0]) for j in matrix):
                pass
            else:
                raise TypeError('The second argument `metric_from_matrix` should be a list of lists in the shape of a square 2-d. array.')
        else:
            raise TypeError('The second argument `metric_from_matrix` should be a list of lists in the shape of a square 2-d. array.')
    else:
        raise TypeError('The second argument `metric_from_matrix` should be a list of lists in the shape of a square 2-d. array.')

    def entry_rule(iT): #define tensor field value for a particular index tuple
        return matrix[iT[0]][iT[1]]
    deg = 2

    def generate_nondecreasing_indices(shape, min_val=0):
        """
        Recursively generates all index tuples of nondecreasing integer lists for an arbitrary dimensional array.
        
        Parameters
        ----------
        shape : tuple
            A tuple where each element defines the dimension (length) of each axis.
        min_val : int, optional
            The minimum value allowed for the current index, ensuring nondecreasing order.

        Returns
        -------
        list of tuples
            All possible index tuples with nondecreasing integer values.
        """
        if len(shape) == 1:
            # Generate the indices for the last dimension, constrained by min_val
            return [(i,) for i in range(min_val, shape[0])]
        else:
            # Recursively generate indices for the next dimensions, ensuring nondecreasing order
            return [(i,) + t for i in range(min_val, shape[0]) for t in generate_nondecreasing_indices(shape[1:], i)]

    shape = (len(coordinates),)*deg
    sparse_data = {indices: entry_rule(indices) for indices in generate_nondecreasing_indices(shape)}
    return metricClass(STFClass(coordinates, sparse_data, deg))





# import importlib
# from sympy import Symbol
# def inverse_with_precomputed(matrix):
#     # Ensure the input is an ImmutableSparseNDimArray and 2D
#     if not isinstance(matrix, ImmutableSparseNDimArray):
#         raise TypeError("Input must be a sympy.ImmutableSparseNDimArray.")
    
#     shape = matrix.shape
#     if len(shape) != 2 or shape[0] != shape[1]:
#         raise ValueError("Input must be a 2D square matrix.")
    
#     dim = shape[0]

#     try:
#         # Dynamically import the correct module for the matrix dimension
#         module_name = f'DGCV.assets.inversions.precomputed_{dim}x{dim}'
#         precomputed_inverses = importlib.import_module(module_name)
#     except ImportError:
#         raise ImportError(f"Precomputed inverses for {dim}x{dim} matrices not found.")
    
#     # Retrieve the precomputed inverse formula
#     inverse_formula = precomputed_inverses.inverse_formulas.get(dim)
    
#     if not inverse_formula:
#         raise ValueError(f"No precomputed formula found for {dim}x{dim} matrices.")
    
#     # Create a dictionary for substitution (map symbolic labels to matrix entries)
#     subs_dict = {}
    
#     # Substitute values for the symbolic variables in the adjugate and determinant
#     for i in range(dim):
#         for j in range(dim):
#             subs_dict[Symbol(f'l{i+1}l{j+1}')] = matrix[(i, j)]  # Use tuple indexing

#     # Substitute the symbolic variables in the adjugate and determinant
#     adjugate_substituted = [[entry.subs(subs_dict) for entry in row] for row in inverse_formula['adjugate']]
#     det_substituted = inverse_formula['det'].subs(subs_dict)

#     # Compute the inverse using the adjugate and determinant
#     if det_substituted == 0:
#         raise ValueError("Matrix is singular and cannot be inverted.")
    
#     # Divide adjugate by determinant to get the inverse
#     inverse_matrix = [[entry / det_substituted for entry in row] for row in adjugate_substituted]
    
#     # Clean up memory
#     del precomputed_inverses
#     import gc
#     gc.collect()

#     # Convert result back to an ImmutableSparseNDimArray
#     result_shape = (dim, dim)
#     inverse_sparse = ImmutableSparseNDimArray(inverse_matrix, result_shape)
    
#     return Matrix(inverse_sparse)
