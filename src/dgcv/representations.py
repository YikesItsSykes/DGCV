
from .algebras.algebras_core import FAClass
from .tensors import vectorSpace


class generalLinearGroup(FAClass):
    def __init__(self, vector_space, grading=None, format_sparse=False, process_matrix_rep=False, _label=None, _basis_labels=None, _calledFromCreator=None, _callLock=None, _print_warning=None, _child_print_warning=None, _exclude_from_VMF=None):
        if not isinstance(vector_space,(FAClass,vectorSpace)):
            raise TypeError('The `vector_space` parameter in the `generalLinearGroup` initializer must of type `FAClass` or `vectorSpace`.')
        super().__init__(vector_space, grading, format_sparse, process_matrix_rep, _label, _basis_labels, _calledFromCreator, _callLock, _print_warning, _child_print_warning, _exclude_from_VMF)
