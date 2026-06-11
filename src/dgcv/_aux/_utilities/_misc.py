"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux._config


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import uuid

from .._backends._types_and_constants import imag_unit, symbol


def zip_sum(*args, init=0):
    if len(args) == 2:
        return sum([a * b for a, b in zip(*args)], init)
    if len(args) == 0:
        return
    if len(args) == 1:
        return args[0]
    return zip_sum(zip_sum(*args[:-1]), args[-1])


def linear_combination(
    elements,
    coef_label: str = None,
    separate_real_and_imag_parts=False,
    coefficient_assumptions=None,
    **kwargs,
):
    if separate_real_and_imag_parts:
        imag = imag_unit()
        if isinstance(coef_label, str):
            vl = coef_label
            vlr, vli = coef_label + "r", coef_label + "i"
        else:
            prefix = kwargs.pop("prefix", None) or "_dgcvvar_"
            prefr, prefi = prefix + "r" or "_dgcvr_", prefix + "i" or "_dgcvi_"
            suff = uuid.uuid4().hex[:6]
            vlr, vli = prefr + suff, prefi + suff
        n, ad = len(elements), {"real": True}
        vr = [symbol(f"{vlr}{x}", assumptions=ad) for x in range(n)]
        vi = [symbol(f"{vli}{x}", assumptions=ad) for x in range(n)]
        v = [a + imag * b for a, b in zip(vr, vi)]
        return zip_sum(v, elements), vr, vi
    if isinstance(coef_label, str):
        vl = coef_label
    else:
        pref = kwargs.pop("prefix", None) or "_dgcvvar_"
        vl = pref + uuid.uuid4().hex[:6]
    v = [
        symbol(f"{vl}{x}", assumptions=coefficient_assumptions)
        for x in range(len(elements))
    ]
    return zip_sum(v, elements), v
