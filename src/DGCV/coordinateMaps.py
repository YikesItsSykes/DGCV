
from .classesAndVariables import allToReal, allToSym, VF_coeffs, addVF, STFClass, DFClass, TFClass
from .RiemannianGeometry import metricClass
from .complexStructures import KahlerStructure
from .vectorFieldsAndDifferentialForms import get_VF, get_DF
from .config import get_variable_registry
from sympy import Basic, simplify, re, im

class coordinate_map(Basic):
    def __new__(cls, coordinates1, coordinates2, coordinate_formulas, holomorphic = None):
        obj = Basic.__new__(cls, coordinates1, coordinates2, coordinate_formulas)
        return obj

    def __init__(self, coordinates1, coordinates2, coordinate_formulas, holomorphic = None):
        if not all(isinstance(j,(list,tuple)) for j in [coordinates1,coordinates2,coordinate_formulas]):
            raise TypeError('`coordinate_map` needs all arguments given in the form of lists or tuples.')
        if len(coordinates2) != len(coordinate_formulas):
            raise TypeError('`coordinate_map` recieved incompatible initialization data. The number of coordinates provided for the range (i.e., in the second argument position) must match the number of components in the provided formulas for the mapping\'s components (i.e., in the third argument position).')
        
        vr = get_variable_registry()

        self.domain = tuple(coordinates1)
        self.range = tuple(coordinates2)
        if holomorphic == True:
            def get_real_parts(varList):
                reals = []
                ims = []
                for var in varList:
                    varStr = str(var)
                    for parent in vr['complex_variable_systems']:
                        if varStr in vr['complex_variable_systems'][parent]['variable_relatives']:
                            realParts = vr['complex_variable_systems'][parent]['variable_relatives'][varStr]['complex_family'][2:]
                            reals = reals + [realParts[0]]
                            ims = ims + [realParts[1]]
                return reals+ims
            if all(j in vr['conversion_dictionaries']['symToReal'] for j in self.domain+self.range):
                self.domain = get_real_parts(self.domain)
                self.range = get_real_parts(self.range)
                coordinate_formulas = [simplify(re(allToReal(j))) for j in coordinate_formulas]+[simplify(im(allToReal(j))) for j in coordinate_formulas]
            else:
                raise TypeError('When setting `holomorphic=True`, `coordinate_map` expects the variables given for the domain and range to be all holomorphic parts of DGCV complex variable systems. It will infer the appropriate action of antiholomophic/real/imaginary parts using holomorphicity.')
        self.domain_varSpace_type,self.domain_frame,self.domain_coframe=coordinate_map.validate_coordinates(self.domain)
        self.range_varSpace_type,self.range_frame,self.range_coframe=coordinate_map.validate_coordinates(self.range)
        self._varSpace_type = self.domain_varSpace_type
        self.coordinate_formulas = list(coordinate_formulas)
        self._JacobianMatrix = None

    @property
    def JacobianMatrix(self):
        if self._JacobianMatrix == None:
            self._JacobianMatrix = [[simplify(j(k)) for j in self.domain_frame] for k in self.coordinate_formulas]
        return self._JacobianMatrix

    @staticmethod
    def validate_coordinates(varSpace):

        if len(varSpace) != len(set(varSpace)):
            raise TypeError('`coordinate_map` was a list of variables for coordinates (either for the domain or range) that have repeated values.)')
        vr = get_variable_registry()
        def checkReal(var):
            return var in vr['conversion_dictionaries']['realToSym']
        def checkHol(var):
            return var in vr['conversion_dictionaries']['symToReal']
        def checkStandard(var):
            varStr=str(var)
            for parent in vr['standard_variable_systems']:
                if varStr in vr['standard_variable_systems'][parent]['variable_relatives']:
                    return True
            return False
        def checkRegistered(var):
            varStr=str(var)
            for parent in vr['complex_variable_systems']:
                if varStr in vr['complex_variable_systems'][parent]['variable_relatives']:
                    return True
            return checkStandard(var)
        
        if any(checkReal(var) for var in varSpace):
            if all(checkReal(var) for var in varSpace):
                return 'real',[allToReal(j) for j in get_VF(*varSpace)],[allToReal(j) for j in get_DF(*varSpace)]
            else:
                raise TypeError('`coordinate_map` was given coordinates (either for the domain or range) that are all associated with complex variable systems but given with non-uniform variable format between \'real coordinates\' (i.e., real and imaginary parts of holomorphic coordinates) and \'holomorphic coordinates\'. Coordinates should be provided in a uniform format, be it real or holomorphic. (The format may vary between domain and range.)')
        elif all(checkHol(var) for var in varSpace):
            return 'complex',[allToSym(j) for j in get_VF(*varSpace)],[allToSym(j) for j in get_DF(*varSpace)]
        elif all(checkStandard(var) for var in varSpace):
            return 'standard', get_VF(*varSpace), get_DF(*varSpace)
        elif not all(checkRegistered(var) for var in varSpace):
            raise TypeError('`coordinate_map` was given coordinates containing variables that were not initialized in the DGCV variable management framework (VMF). Use variable creation functions like `createVariables` to initialize variables while automatically registering them in the VMF.')
        else:

            raise TypeError('`coordinate_map` was given coordinates (either for the domain or range) that contain a mixture of variable types, i.e., some from a \'standard\' variable system and some from a \'complex\' variable system. Only one type is allowed per coordinate set (but the type can vary between domain and range).')
        

    def differential(self,vf):
        inputCoeffs = VF_coeffs(vf,self.domain)
        vf_list = []
        for j in range(len(self.range)):
            vf_list += [sum([inputCoeffs[k]*self.JacobianMatrix[j][k] for k in range(len(self.domain))])*self.range_frame[j]]
        return addVF(*vf_list)
    
    def pull_back(self,tf):
        if isinstance(tf,STFClass):
            vf_basis = [self.differential(vf) for vf in self.domain_frame]
            if set(tf.varSpace)==set(self.range):
                def entry_rule(iT): #define tensor field value for a particular index tuple
                    argumentList=[vf_basis[iT[j]] for j in range(len(iT))]
                    return tf(*argumentList)
                deg = tf.degree
            
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

                shape = (len(self.domain),)*deg
                sparse_data = {indices: entry_rule(indices) for indices in generate_nondecreasing_indices(shape)}
                return STFClass(self.domain, sparse_data, deg, DGCVType=tf.DGCVType,simplifyKW=tf.simplifyKW)

            else:
                raise Exception('`coordinate_map.pull_back` can only be applied to differential for tensor fields defined over the same coordinates initialized as `coodinate_map.range`.')
        if isinstance(tf,TFClass):
            vf_basis = [self.differential(vf) for vf in self.domain_frame]
            if set(tf.varSpace)==set(self.range):
                def entry_rule(iT): #define tensor field value for a particular index tuple
                    argumentList=[vf_basis[iT[j]] for j in range(len(iT))]
                    return tf(*argumentList)
                deg = tf.degree
            
                def generate_indices(shape):
                    """Recursively generates all index tuples for an arbitrary dimensional array."""
                    if len(shape) == 1:
                        return [(i,) for i in range(shape[0])]
                    else:
                        return [(i,) + t for i in range(shape[0]) for t in generate_indices(shape[1:])]
                shape = (dim,)*3
                sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
                
                shape = (len(self.domain),)*deg
                sparse_data = {indices: entry_rule(indices) for indices in generate_indices(shape)}
                return TFClass(self.domain, sparse_data, deg, DGCVType=tf.DGCVType,simplifyKW=tf.simplifyKW)

            else:
                raise Exception('`coordinate_map.pullack` can only be applied to differential for tensor fields defined over the same coordinates initialized as `coodinate_map.range`.')
        if isinstance(tf,DFClass):
            vf_basis = [self.differential(vf) for vf in self.domain_frame]
            if set(tf.varSpace)==set(self.range):
                def entry_rule(iT): #define tensor field value for a particular index tuple
                    argumentList=[vf_basis[iT[j]] for j in range(len(iT))]
                    return tf(*argumentList)
                deg = tf.degree
            
                def generate_nondecreasing_indices(shape, min_val=0):
                    """
                    Recursively generates all index tuples of increasing integer lists for an arbitrary dimensional array.
                    
                    Parameters
                    ----------
                    shape : tuple
                        A tuple where each element defines the dimension (length) of each axis.
                    min_val : int, optional
                        The minimum value allowed for the current index, ensuring nondecreasing order.

                    Returns
                    -------
                    list of tuples
                        All possible index tuples with strictly increasing integer values.
                    """
                    if len(shape) == 1:
                        # Generate the indices for the last dimension, constrained by min_val
                        return [(i,) for i in range(min_val, shape[0])]
                    else:
                        # Recursively generate indices for the next dimensions, ensuring strictly increasing order
                        return [(i,) + t for i in range(min_val, shape[0]) for t in generate_nondecreasing_indices(shape[1:], i + 1)]
                shape = (len(self.domain),)*deg
                sparse_data = {indices: entry_rule(indices) for indices in generate_nondecreasing_indices(shape)}
                return DFClass(self.domain, sparse_data, deg, DGCVType=tf.DGCVType,simplifyKW=tf.simplifyKW)

            else:
                raise Exception('`coordinate_map.pullack` can only be applied to differential for tensor fields defined over the same coordinates initialized as `coodinate_map.range`.')
        if isinstance(tf,metricClass):
            return metricClass(self.pull_back(tf.SymTensorField))
        
        if isinstance(tf,KahlerStructure):
            newKF = self.pull_back(tf.kahlerForm)
            newVS=newKF.varSpace
            return KahlerStructure(newVS,newKF)





                    
