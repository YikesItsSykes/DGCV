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

from .._backends._types_and_constants import symbol


def zip_sum(*args, init=0):
    if len(args) == 2:
        return sum([a * b for a, b in zip(*args)], init)
    if len(args) == 0:
        return
    if len(args) == 1:
        return args[0]
    return zip_sum(zip_sum(*args[:-1]), args[-1])


def linear_combination(elements, prefix: str = None):
    pref = prefix or "_dgcvvar_"
    vl = pref + uuid.uuid4().hex[:6]
    v = [symbol(f"{vl}{x}") for x in range(len(elements))]
    return zip_sum(v, elements), v
