from collections import Counter

from ..combinatorics.combinatorics import permSign
from ..dgcv_core.spaces.spaces import _vs_card


def _infer_shape(t1, t2, impose_shape=None):
    s1, s2 = getattr(t1, "shape", "general"), getattr(t2, "shape", "general")
    if not isinstance(s1, str):
        s1 = "custom"
    if not isinstance(s2, str):
        s2 = "custom"
    if isinstance(impose_shape, str):
        shapes = {"all", impose_shape}
        if s1 not in shapes or s2 not in shapes:
            raise TypeError(
                f"Attempted binary operation between `tensorProduct` elements of shapes {s1} and {s2} is not supported or disallowed."
            )
        new_shape = impose_shape
    else:
        new_shape = s1 if (s2 == "all" or s1 == s2) else s2 if s1 == "all" else "broken"
    if new_shape == "broken":
        raise TypeError(
            f"Attempted binary operation between `tensorProduct` elements of shapes {s1} and {s2} is not supported."
        )
    return new_shape


def _factor_order(factor):
    card = factor[2]
    return (factor[0], factor[1], card.uid if type(card) is _vs_card else -1)


def _shape_filter(
    cd: dict | Counter, shape: str, accumulate=False, counter_format: bool = False
):
    def skew_filter(key: tuple):
        if len(key) != len(set(key)):
            return
        sign, new_key = permSign(key, returnSorted=True, key=_factor_order)
        return tuple(new_key), sign

    def symmetric_filter(key: tuple):
        return tuple(sorted(key, key=_factor_order)), 1

    current_filter = (
        shape
        if callable(shape)
        else {"skew": skew_filter, "symmetric": symmetric_filter}.get(shape, None)
    )
    if current_filter is None:
        return cd
    out = Counter() if counter_format else dict()
    for k, v in cd.items():
        canonicallized = current_filter(k)
        if canonicallized is None:
            continue
        key, sign = canonicallized
        if accumulate:
            out[key] = out.get(key, 0) + sign * v
            continue
        if key in out:
            if sign * v != out[key]:
                raise ShapeError(
                    "Tensor coefficient dict does not match the specified shape."
                )
            continue
        else:
            out[key] = sign * v
    return out


class ShapeError(Exception):
    """tensor coefficient dict does not match the specified shape."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
