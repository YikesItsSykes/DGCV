import warnings

from ._config import get_dgcv_settings_registry
from ._dgcv_display import dgcv_init_printing
from .backends._sage_backend import is_sage_available


def set_dgcv_settings(theme=None,
                      format_displays=None,
                      use_latex=None,
                      version_specific_defaults=None,
                      ask_before_overwriting_objects_in_vmf=None,
                      forgo_warnings=None,default_engine=None):
    dgcvSR = get_dgcv_settings_registry()
    if theme is not None:
        dgcvSR['theme'] = theme
    if format_displays is True:
        dgcv_init_printing()
    if use_latex is not None:
        dgcvSR['use_latex'] = use_latex
    if version_specific_defaults is not None:
        dgcvSR['version_specific_defaults'] = version_specific_defaults
    if ask_before_overwriting_objects_in_vmf is not None:
        dgcvSR['ask_before_overwriting_objects_in_vmf'] = ask_before_overwriting_objects_in_vmf
    if forgo_warnings is not None:
        dgcvSR['forgo_warnings'] = forgo_warnings
    if default_engine is not None:
        engine = str(default_engine).lower()
        if engine in ("sage", "sagemath"):
            if is_sage_available():
                dgcvSR['default_symbolic_engine'] = 'sage'
            else:
                warnings.warn(
                    "SageMath backend requested via `set_dgcv_settings(default_engine='sage'), "
                    "but Sage is not available. Default engine was not updated."
                )
        elif engine in ("sympy",):
            dgcvSR['default_symbolic_engine'] = 'sympy'
        else:
            warnings.warn(
                f"Unrecognized default_engine value {default_engine!r}. "
                "Supported options are 'sympy' and 'sage'. Default engine was not updated."
            )
