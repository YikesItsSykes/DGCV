import sympy as sp

from ._safeguards import get_dgcv_category, query_dgcv_categories, retrieve_passkey
from .combinatorics import carProd
from .solvers import solve_dgcv
from .tensors import tensorProduct, vector_space_class


class homomorphism:
    def __init__(self,domain,codomain,basis_image=[]):
        classes={'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace','tensor_proxy'}
        if isinstance(codomain,(list,tuple)):
            if all(get_dgcv_category(elem) in classes for elem in codomain):
                codomain=_fast_tensor_proxy(codomain)
            else:
                raise TypeError('If providing the `codomain` parameter as a list reprenting a tensor product of vector spaces then its elements must be vector space class types (or similar, e.g., `algebra_class` etc.)')
        if isinstance(domain,(list,tuple)):
            if all(get_dgcv_category(elem) in classes for elem in domain):
                domain=_fast_tensor_proxy(domain)
            else:
                raise TypeError('If providing the `domain` parameter as a list reprenting a tensor product of vector spaces then its elements must be vector space class types (or similar, e.g., `algebra_class` etc.)')

        classes.add('tensor_proxy')
        if not all(get_dgcv_category(space) in classes for space in [domain,codomain]):
            raise TypeError('domain and codomain parameters for `homomorphism` must be vector space class types (or similar, e.g., `algebra_class` etc.)')
        # Default to the zero action in the codomain, not its endomorphisms
        if basis_image==[]:
            basis_image=[codomain.zero_element]*domain.dimension
        if not isinstance(basis_image,(list,tuple)) or len(basis_image)!=domain.dimension:
            raise TypeError(f'`basis_image` parameter should be a list of elements in the codomain. Its length must match the dimension of the domain. Recieved {basis_image} which should have length {domain.dimension}')
        newBI = []
        def _decomp(tp):
            if get_dgcv_category(codomain)=='tensor_proxy':
                pass
            if not hasattr(codomain,'tensor_representation'):
                return tuple()
            vars = [sp.Symbol(f'c{j}') for j in range(codomain.dimension)]
            tps=[var*elem for var,elem in zip(vars,codomain.tensor_representation)]
            if len(tps)==0:
                return tuple()
            sol = solve_dgcv(sum(tps,-tp),vars)
            if len(sol)==0:
                return tuple()
            return tuple(j.subs(sol[0]) for j in vars)

        all_zero = True
        for elem in basis_image:
            if get_dgcv_category(codomain)=='tensor_proxy':
                if codomain.contains(elem):
                    if isinstance(elem,(list,tuple)):
                        tp=elem[0]
                        for factor in elem[1:]:
                            tp = tp@factor
                        newBI.append(tp)
                    else:
                        newBI.append(elem)
                else:
                    raise TypeError(f'`basis_image` parameter should be a list of elements in the codomain. Its length must match the dimension of the domain. Recieved {basis_image} which should have length {domain.dimension}')
            elif getattr(elem,'dgcv_vs_id',None)==codomain.dgcv_vs_id:
                newBI.append(elem)
            elif get_dgcv_category(elem)=='tensorProduct':
                coeffs=_decomp(elem) 
                if len(coeffs)>0:
                    terms = [v*e for v,e in zip(coeffs,codomain.basis)]
                    newBI.append(sum(terms[1:],terms[0]))
                else:
                    raise TypeError(f'`basis_image` parameter should be a list of elements in the codomain. Its length must match the dimension of the domain. Recieved {basis_image} which should have length {domain.dimension}')
            else:
                raise TypeError('`basis_image` parameter should be a list of elements in the codomain. Its length must match the dimension of the domain.')
            if not (getattr(newBI[-1],'is_zero',False) or newBI[-1]==0 or (isinstance(newBI[-1],(list,tuple)) and all(getattr(vec[-1],'is_zero',False) or vec[-1]==0 for vec in newBI[-1]))):
                all_zero = False
        self.domain=domain
        self.codomain=codomain
        self._zero_map = all_zero
        if not self._zero_map:
            self.tensor_representation=sum(e1@e2.dual() for e1,e2 in zip(newBI,domain.basis) if not getattr(e1,'is_zero',False))
        else:
            self.tensor_representation=None

    def __repr__(self):
        if getattr(self, '_zero_map', False):
            return (self.codomain.zero_element)@(self.domain.zero_element.dual()).__repr__
        return self.tensor_representation.__repr__()
    def __str__(self):
        if getattr(self, '_zero_map', False):
            return "0"
        return self.tensor_representation.__str__()
    def _repr_latex_(self,**kwargs):
        if getattr(self, '_zero_map', False):
            return r"$0$"
        return self.tensor_representation._repr_latex_(**kwargs)
    def __call__(self, elem):
        if getattr(elem,'dgcv_vs_id',None)==self.domain.dgcv_vs_id:
            if getattr(self, '_zero_map', False):
                return self.codomain.zero_element
            return self.tensor_representation(elem,demote_to_VS_when_possible=True)
        return 'UNDEF'

class _fast_tensor_proxy:
    def __init__(self,vector_spaces:list|tuple):
        val_vs=_fast_tensor_proxy._validate_vs(vector_spaces)
        if isinstance(val_vs,str):
            raise TypeError(val_vs)
        self.vector_spaces=val_vs
        dim=1
        dimensions=[]
        for vs in val_vs:
            nd=vs.dimension
            dimensions.append(nd)
            dim = dim*nd
        self.dimensions=tuple(dimensions)
        self.degree=len(self.dimensions)
        self.domain=self.vector_spaces[-1].dual() if self.degree>0 else vector_space_class(0)
        self.dim=dim if len(val_vs)>0 else 0
        self.zero_element = tuple(vs.zero_element for vs in self.vector_spaces)
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='tensor_proxy'
        self.dgcv_vs_id='tensor_proxy'
        self._pre_basis_cache=None
        self._basis=None

    @staticmethod
    def _validate_vs(vs_list):
        classes={'vector_space','algebra','subalgebra'}
        wm='_fast_tensor_proxy.__init__ requires `vector_spaces` to be a list/tuple of vector space-like objects.'
        if not isinstance(vs_list,(list,tuple)):
            return wm
        new_list=[]
        for elem in vs_list:
            if get_dgcv_category(elem) in classes:
                new_list.append(elem)
            elif isinstance(elem,(list,tuple)):
                inner_list=_fast_tensor_proxy._validate_vs(elem)
                if isinstance(inner_list,str):
                    return wm
                new_list+=inner_list
            else:
                return wm
        return new_list
    def contains(self,elem,*args):
        if elem==0:
            return True
        if self.degree==0:
            if elem==0 or getattr(elem,'is_zero',False):
                return True
            return False
        elif self.degree==1:
            return self.vector_spaces[0].contains(elem,strict_types=False)
        if isinstance(elem,(list,tuple)):
            if len(elem)==self.degree and all(vs.contains(el,strict_types=False) for vs,el in zip(self.vector_spaces,elem)):
                return True
            else:
                return False
        if get_dgcv_category(elem)=='tensorProduct' and elem.max_degree==self.degree and elem.min_degree==self.degree:
            for k in elem.coeff_dict.keys():
                deg=len(k)
                for vs,vsidx,val in zip(self.vector_spaces,k[2*deg:],k[deg:2*deg]):
                    if vs.dgcv_vs_id==vsidx:
                        if query_dgcv_categories(vs,'algebra_dual'):
                            if val==0:
                                continue
                        elif val==1:
                            continue
                    return False    ### add logic branch here for subalgebra support
            return True
        return False

    @property
    def _pre_basis(self):
        if self._pre_basis_cache is None:
            self._pre_basis_cache=carProd(*[vs.basis for vs in self.vector_spaces])
        return self._pre_basis_cache


    @property
    def basis(self):
        if self._basis is None:
            def prodList(liEl):
                if len(liEl)==0:
                    return tensorProduct('_',tuple())
                tp=liEl[0]
                for el in liEl[1:]:
                    tp=tp@el
                return tp
            self._basis=tuple(prodList(elem) for elem in self._pre_basis)
        return self._basis
