from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._vmf._safeguards import get_dgcv_category
from .shapes import _infer_shape


class _tp_products:
    def tp(self, other, impose_shape=None, _guard_shape=True):
        """Tensor product with another tensorProduct."""
        if get_dgcv_category(other) in {
            "subalgebra_element",
            "algebra_element",
        }:
            other = other._convert_to_tp()
        if isinstance(other, expr_numeric_types()):
            return other * self
        if not isinstance(other, tensorProduct):
            raise ValueError(
                f"The other object must be a tensorProduct instance or vector space element related class. Recieved {type(other)} instead."
            )

        new_shape = _infer_shape(self, other, impose_shape=impose_shape)
        if _guard_shape:
            if not (new_shape == "general" or new_shape == "all"):
                raise TypeError(
                    f"The standard tensor product does not currently support operation on shape {new_shape}."
                )
        # Compute new coefficient dictionary
        new_coeff_dict = {}
        for key1, value1 in self.coeff_dict.items():
            for key2, value2 in other.coeff_dict.items():
                new_coeff_dict[key1 + key2] = value1 * value2

        return tensorProduct(
            new_coeff_dict,
            shape=new_shape,
            _process_shape_with_accumulation=True,
        )

    def shape_inferred_tensor_product(self, other, impose_shape=None):
        return self.tp(other, impose_shape=impose_shape, _guard_shape=False)

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        return self.tp(other)

    def __rmatmul__(self, other):
        """Overload @ operator for tensor product."""
        if isinstance(other, expr_numeric_types()):
            return tensorProduct({tuple(): other}).__matmul__(self)
        elif get_dgcv_category(other) == "tensorProduct":
            return other.__matmul__(self)
        elif get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            return other._convert_to_tp().__matmul__(self)
