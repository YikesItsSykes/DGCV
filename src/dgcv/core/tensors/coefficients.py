from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..dgcv_core.spaces.spaces import card_root
from .promotion import _promote_keys, _promotion_targets
from .shapes import _shape_filter

_key_format_message = (
    "Keys in coeff_dict must be tuples of (index, valence, card) triples, one per "
    "tensor factor, where valence is 0 or 1. Received keys: {keys}"
)


def _validate_key(key, coeff_dict, numeric_types):
    if not isinstance(key, tuple):
        raise ValueError(_key_format_message.format(keys=list(coeff_dict.keys())))
    for factor in key:
        if (
            type(factor) is not tuple
            or len(factor) != 3
            or (factor[1] != 0 and factor[1] != 1)
            or not isinstance(factor[0], numeric_types)
        ):
            raise ValueError(_key_format_message.format(keys=list(coeff_dict.keys())))


def _build(coeff_dict, shape, _amb_prom, accumulate, validate):
    if not coeff_dict:
        return {tuple(): 0}, 0, 0, ()
    numeric_types = expr_numeric_types() if validate else None
    max_degree = 0
    min_degree = -1
    seen = set()
    cards = []
    processed = dict()
    for key, value in coeff_dict.items():
        if _scalar_is_zero(value):
            continue
        if validate:
            _validate_key(key, coeff_dict, numeric_types)
        deg = len(key)
        if deg > max_degree:
            max_degree = deg
        if min_degree < 0 or deg < min_degree:
            min_degree = deg
        for factor in key:
            card = factor[2]
            if card not in seen:
                seen.add(card)
                cards.append(card)
        processed[key] = value
    targets = _promotion_targets(cards, _amb_prom) if cards else set()
    if targets:
        processed = dict(_promote_keys(processed, targets))
        vs_id = []
        promoted = set()
        for card in cards:
            card = card_root(card) if card in targets else card
            if card not in promoted:
                promoted.add(card)
                vs_id.append(card)
        vs_id = tuple(vs_id)
    else:
        vs_id = tuple(cards)
    if not processed:
        processed[tuple()] = 0
    if shape == "skew" or shape == "symmetric" or callable(shape):
        processed = _shape_filter(processed, shape=shape, accumulate=accumulate)
        if not processed:
            processed[tuple()] = 0
    return processed, max_degree, min_degree, vs_id


def _process_coeffs_dict(
    coeff_dict, shape, _amb_prom, _process_shape_with_accumulation=False
):
    return _build(
        coeff_dict,
        shape,
        _amb_prom,
        _process_shape_with_accumulation,
        True,
    )


def _process_coeffs_dict_trusted(
    coeff_dict, shape, _amb_prom, _process_shape_with_accumulation=False
):
    return _build(
        coeff_dict,
        shape,
        _amb_prom,
        _process_shape_with_accumulation,
        False,
    )
