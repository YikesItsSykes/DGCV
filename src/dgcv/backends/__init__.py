"""
package: dgcv - Differential Geometry with Complex Variables
module: backends/__init__

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
# calculus-like operations
from ._calculus import diff as diff_dgcv
from ._calculus import integrate as integrate_dgcv

# expression/router operations
from ._symbolic_router import (
    as_numer_denom,
    get_free_symbols,
    ilcm,
    im,
    ratio,
    re,
)
from ._symbolic_router import (
    cancel as cancel_dgcv,
)
from ._symbolic_router import (
    conjugate as conjugate_dgcv_sym_router,
)
from ._symbolic_router import (
    expand as expand_dgcv,
)
from ._symbolic_router import (
    factor as factor_dgcv,
)
from ._symbolic_router import (
    simplify as simplify_dgcv,
)
from ._symbolic_router import (
    subs as subs_dgcv,
)

__all__ = [
    # calculus
    "diff_dgcv",
    "integrate_dgcv",
    # router
    "as_numer_denom",
    "cancel_dgcv",
    "conjugate_dgcv_sym_router",
    "expand_dgcv",
    "factor_dgcv",
    "get_free_symbols",
    "ilcm",
    "im",
    "re",
    "ratio",
    "simplify_dgcv",
    "subs_dgcv",
]
