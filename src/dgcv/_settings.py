
from ._config import get_dgcv_settings_registry
from ._dgcv_display import dgcv_init_printing


def set_dgcv_settings(theme=None,format_displays=None,use_latex=None,version_specific_defaults=None):
    dgcvSR = get_dgcv_settings_registry()
    if theme is not None:
        dgcvSR['theme'] = theme
    if format_displays is True:
        dgcv_init_printing()
    if use_latex is not None:
        dgcvSR['use_latex'] = use_latex
    if version_specific_defaults is not None:
        dgcvSR['version_specific_defaults'] = version_specific_defaults

