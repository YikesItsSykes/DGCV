
from ._config import get_dgcv_settings_registry
from ._dgcv_display import dgcv_init_printing


def set_dgcv_settings(theme=None,
                      format_displays=None,
                      use_latex=None,
                      version_specific_defaults=None,
                      ask_before_overwriting_objects_in_vmf=None,
                      forgo_warnings=None):
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


