"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.styles


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
import colorsys
import difflib
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union

from ._config import get_dgcv_settings_registry

__all__ = ["get_dgcv_themes", "get_style", "ThemeConfig"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------

dgcv_display_theme = "matte_slate_soft"
dgcv_custom_variables = [
    "--dgcv-border-width",
    "--dgcv-border-radius",
    "--dgcv-hover-transform",
    "--dgcv-table-shadow",
    "--dgcv-special-background",
    "--dgcv-text-shadow",
    "--dgcv-hover-transition",
    "--dgcv-border-image",
    "--dgcv-hover-font-weight",
]


@dataclass
class ThemeConfig:
    """
    Configuration dataclass for a single dgcv HTML theme.

    Each theme defines a palette of CSS custom properties (variables) that are
    consumed by dgcv's HTML builders. Builders access theme data exclusively
    through the CSS variables emitted by ``to_css_string`` and ``get_style`` —
    they never read ThemeConfig fields directly.

    Background/foreground pairs are designed to be used together to ensure
    readable contrast. Mixing across pairs may produce inaccessible color
    combinations:

        ``bg_surface``      /  ``text_heading``
        ``bg_primary``      /  ``text_main``
        ``bg_alt``          /  ``text_alt``       (falls back to text_main if unset)
        ``bg_hover``        /  ``text_hover``
        ``bg_action``       /  ``text_on_action``
        ``bg_action_hover`` /  ``text_on_action``
        ``bg_error``        /  ``text_on_error``
        ``bg_success``      /  ``text_on_success``

        optional pair in custom_css_vars:
        ``--dgcv-special-background``      /  ``--dgcv-special-text``


    Attributes:
        bg_primary: Main background color for table/panel body areas.
        text_main: Default body text color. Paired with ``bg_primary``.
        bg_surface: Header/caption background. Paired with ``text_heading``.
        text_heading: Text color for headers and captions. Paired with ``bg_surface``.
        bg_alt: Alternating row background. Paired with ``text_alt``.
        bg_hover: Background applied on hover. Paired with ``text_hover``.
        text_hover: Text color applied on hover. Paired with ``bg_hover``.
        bg_action: Background for action/button elements. Paired with ``text_on_action``.
        text_on_action: Text color for action elements. Paired with ``bg_action``.
        bg_action_hover: Hover background for action elements. Paired with ``text_on_action``.
        bg_error: Background for error states. Paired with ``text_on_error``.
        text_on_error: Text color for error states. Paired with ``bg_error``.
        bg_success: Background for success states. Paired with ``text_on_success``.
        text_on_success: Text color for success states. Paired with ``bg_success``.
        border_main: Primary border color, used for outer borders and header dividers.
        border_alt: Secondary border color, used for row dividers and subtle borders.
        font_family: CSS font-family string. Defaults to ``"inherit"``.
        text_alt: Alternating row text color. Paired with ``bg_alt``. Falls back
            to ``text_main`` if not set.
        custom_css_vars: Optional dict of additional CSS custom properties to
            emit alongside the standard palette. Keys may include or omit the
            leading ``--``; if omitted the ``dgcv-`` prefix is applied
            automatically. See the dgcv custom variables registry for supported
            values and their effects.
    """

    bg_primary: str
    text_main: str
    bg_surface: str
    text_heading: str
    bg_alt: str
    bg_hover: str
    text_hover: str
    bg_action: str
    text_on_action: str
    bg_action_hover: str
    bg_error: str
    text_on_error: str
    bg_success: str
    text_on_success: str
    border_main: str
    border_alt: str

    font_family: str = "inherit"
    text_alt: Optional[str] = None
    custom_css_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.text_alt is None:
            self.text_alt = self.text_main

    def to_css_dict(self, prefix: str = "dgcv-") -> Dict[str, str]:
        """
        Returns all theme values as a dict of CSS custom property names to values.

        Standard fields are emitted as ``--{prefix}{field-name}`` with underscores
        replaced by hyphens. Entries in ``custom_css_vars`` are appended after the
        standard fields; keys that already start with ``--`` are used verbatim,
        otherwise the prefix is applied.

        Args:
            prefix: CSS variable prefix. Defaults to ``"dgcv-"``.

        Returns:
            Ordered dict of CSS variable names to their string values.
        """
        css_vars = {}

        for key, value in asdict(self).items():
            if key == "custom_css_vars" or value is None:
                continue

            css_key = f"--{prefix}{key.replace('_', '-')}"
            css_vars[css_key] = str(value)

        for key, value in self.custom_css_vars.items():
            if key.startswith("--"):
                css_key = key
            else:
                css_key = f"--{prefix}{key.replace('_', '-')}"

            css_vars[css_key] = str(value)

        return css_vars

    def to_css_string(self, prefix: str = "dgcv-") -> str:
        """
        Returns all theme values as a newline-separated string of CSS declarations,
        indented for embedding directly inside a ``{ }`` block.

        Args:
            prefix: CSS variable prefix. Defaults to ``"dgcv-"``.

        Returns:
            String of the form ``--dgcv-bg-primary: #fff;\\n    --dgcv-text-main: #000;``
            suitable for insertion into a ``:root { }`` or scoped selector block.
        """
        css_dict = self.to_css_dict(prefix)
        return "\n    ".join(f"{k}: {v};" for k, v in css_dict.items())

    def registry_format(self, name: str) -> str:
        lines = [f'"{name}": ThemeConfig(']
        fields = [
            ("bg_primary", self.bg_primary),
            ("bg_surface", self.bg_surface),
            ("bg_alt", self.bg_alt),
            ("bg_hover", self.bg_hover),
            ("text_main", self.text_main),
            ("text_heading", self.text_heading),
            ("text_hover", self.text_hover),
            ("text_alt", self.text_alt),
            ("border_main", self.border_main),
            ("border_alt", self.border_alt),
            ("bg_action", self.bg_action),
            ("text_on_action", self.text_on_action),
            ("bg_action_hover", self.bg_action_hover),
            ("bg_error", self.bg_error),
            ("text_on_error", self.text_on_error),
            ("bg_success", self.bg_success),
            ("text_on_success", self.text_on_success),
            ("font_family", self.font_family),
        ]
        for k, v in fields:
            lines.append(f'    {k}="{v}",')
        if self.custom_css_vars:
            lines.append("    custom_css_vars={")
            for k, v in self.custom_css_vars.items():
                lines.append(f'        "{k}": "{v}",')
            lines.append("    },")
        lines.append("),")
        result = "\n".join(lines)
        print(result)
        return result


THEME_REGISTRY: Dict[str, ThemeConfig] = {
    "1980s_neon": ThemeConfig(
        bg_primary="#3b3b58",
        bg_surface="#9400d3",
        bg_alt="#282a36",
        bg_hover="#00ff7f",
        text_main="#ffffff",
        text_heading="#00ff00",
        text_hover="#000000",
        border_main="#ff1493",
        bg_action="#ff1493",
        text_on_action="#ffffff",
        bg_action_hover="#ff69b4",
        bg_error="#ff0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#4b0082",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(90deg, #ff1493, #9400d3, #00ff7f) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-border-width": "2px",
            "--dgcv-text-shadow": "0px 0px 8px #00ff00, 0px 0px 2px #ffffff",
            "--dgcv-table-shadow": "0 0 20px rgba(0, 255, 127, 0.7), 0 0 60px rgba(148, 0, 211, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "appalachian": ThemeConfig(
        bg_primary="#F0F8FF",
        bg_surface="#2E8B57",
        bg_alt="#B0E0E6",
        bg_hover="#5F9EA0",
        text_main="#2F4F4F",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#2E8B57",
        bg_action="#1E90FF",
        text_on_action="#ffffff",
        bg_action_hover="#00BFFF",
        bg_error="#D32F2F",
        text_on_error="#ffffff",
        bg_success="#388E3C",
        text_on_success="#ffffff",
        border_alt="#87CEFA",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(160deg, #5f9ea0 0%, #52898b 50%, #457375 100%)",
            "--dgcv-table-shadow": "0 0 12px rgba(46, 139, 87, 0.35)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.5)",
        },
    ),
    "aurora_borealis": ThemeConfig(
        bg_primary="#0a0e1a",
        bg_surface="#0d1f2d",
        bg_alt="#0d1a2a",
        bg_hover="#00e5cc",
        text_main="#c8f0e8",
        text_heading="#c8f0e8",
        text_hover="#0a0e1a",
        text_alt="#a8d8cc",
        border_main="#00b4a0",
        border_alt="#1a3a4a",
        bg_action="#7b2fff",
        text_on_action="#ffffff",
        bg_action_hover="#9d5fff",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#0a0e1a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(170deg, #0a0e1a 0%, #0d3320 35%, #0a1a3a 65%, #1a0a2e 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(0, 180, 160, 0.4), 0 0 60px rgba(123, 47, 255, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 8px rgba(0, 229, 204, 0.6)",
        },
    ),
    "autumn": ThemeConfig(
        bg_primary="#ffe0b2",
        bg_surface="#bf360c",
        bg_alt="#ffcc80",
        bg_hover="#ffb74d",
        text_main="#5d1a06",
        text_heading="#ffffff",
        text_hover="#5d1a06",
        border_main="#bf360c",
        bg_action="#e65100",
        text_on_action="#ffffff",
        bg_action_hover="#ff9800",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#33691e",
        text_on_success="#ffffff",
        border_alt="#ffb74d",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #ffe0b2 0%, #ffcc80 60%, #ffb74d 100%)",
            "--dgcv-special-text": "#5d1a06",
            "--dgcv-table-shadow": "0 0 14px rgba(191, 54, 12, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 2px rgba(93, 26, 6, 0.2)",
        },
    ),
    "back_to_the_future": ThemeConfig(
        bg_primary="#0059b3",
        bg_surface="#003366",
        bg_alt="#004080",
        bg_hover="#ff7f50",
        text_main="#f7e014",
        text_heading="#f7e014",
        text_hover="#000000",
        border_main="#c0c0c0",
        bg_action="#ff7f50",
        text_on_action="#000000",
        bg_action_hover="#ff9966",
        bg_error="#ff3333",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#000000",
        border_alt="#808080",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #c0c0c0, #ff7f50, #00ffff) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-text-shadow": "0px 0px 8px #ff7f50, 0px 0px 2px #f7e014",
            "--dgcv-table-shadow": "0 4px 16px rgba(255, 127, 80, 0.8), 0 0 40px rgba(0, 255, 255, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "blue": ThemeConfig(
        bg_primary="#24283b",
        bg_surface="#24283b",
        bg_alt="#2f3549",
        bg_hover="#2f3549",
        text_main="#a9b1d6",
        text_heading="#a9b1d6",
        text_hover="#7dcfff",
        border_main="#a9b1d6",
        bg_action="#3d59a1",
        text_on_action="#ffffff",
        bg_action_hover="#7dcfff",
        bg_error="#f7768e",
        text_on_error="#24283b",
        bg_success="#9ece6a",
        text_on_success="#24283b",
        border_alt="#414868",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(125, 207, 255, 0.5)",
        },
    ),
    "blue_gray": ThemeConfig(
        bg_primary="#2e3440",
        bg_surface="#2e3440",
        bg_alt="#3b4252",
        bg_hover="#434c5e",
        text_main="#d8dee9",
        text_heading="#eceff4",
        text_hover="#8fbcbb",
        border_main="#4c566a",
        bg_action="#5e81ac",
        text_on_action="#eceff4",
        bg_action_hover="#81a1c1",
        bg_error="#bf616a",
        text_on_error="#eceff4",
        bg_success="#a3be8c",
        text_on_success="#2e3440",
        border_alt="#434c5e",
        font_family="'Inter', sans-serif",
    ),
    "blue_plain": ThemeConfig(
        bg_primary="#ffffff",
        bg_surface="#0056b3",
        bg_alt="#f7f7f7",
        bg_hover="#cce5ff",
        text_main="#000000",
        text_heading="#ffffff",
        text_hover="#000000",
        border_main="#0056b3",
        bg_action="#007bff",
        text_on_action="#ffffff",
        bg_action_hover="#0056b3",
        bg_error="#dc3545",
        text_on_error="#ffffff",
        bg_success="#28a745",
        text_on_success="#ffffff",
        border_alt="#e9ecef",
        font_family="inherit",
    ),
    "blueprint": ThemeConfig(
        bg_primary="#002b4f",
        bg_surface="#003366",
        bg_alt="#003366",
        bg_hover="#336699",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#cccccc",
        bg_action="#4da6ff",
        text_on_action="#002b4f",
        bg_action_hover="#80bfff",
        bg_error="#ff4d4d",
        text_on_error="#ffffff",
        bg_success="#4dff4d",
        text_on_success="#002b4f",
        border_alt="#e6e6e6",
        font_family="Roboto Mono, monospace",
        custom_css_vars={
            "--dgcv-border-width": "2px",
        },
    ),
    "brass": ThemeConfig(
        bg_primary="#FFF8DC",
        bg_surface="#B87333",
        bg_alt="#E6D5B8",
        bg_hover="#CD7F32",
        text_main="#4B3621",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#B87333",
        bg_action="#8B4513",
        text_on_action="#ffffff",
        bg_action_hover="#A0522D",
        bg_error="#B22222",
        text_on_error="#ffffff",
        bg_success="#556B2F",
        text_on_success="#ffffff",
        border_alt="#D2B48C",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-image": "linear-gradient(135deg, #cd7f32, #ffe08a, #b87333, #ffd700, #cd7f32) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 12px rgba(205, 127, 50, 0.6), inset 0 1px 0 rgba(255, 224, 138, 0.4)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.3)",
        },
    ),
    "brushed_metal": ThemeConfig(
        bg_primary="#b0b8c1",
        bg_surface="#78909c",
        bg_alt="#9aa4ad",
        bg_hover="#cfd8dc",
        text_main="#1c2833",
        text_heading="#1c2833",
        text_hover="#1c2833",
        text_alt="#263238",
        border_main="#546e7a",
        border_alt="#90a4ae",
        bg_action="#455a64",
        text_on_action="#eceff1",
        bg_action_hover="#607d8b",
        bg_error="#c62828",
        text_on_error="#ffffff",
        bg_success="#2e7d32",
        text_on_success="#ffffff",
        font_family="'Courier New', monospace",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(180deg, #d4d9dd 0%, #9aa4ad 40%, #b8bfc5 60%, #cfd4d8 80%, #a8b0b8 100%)",
            "--dgcv-table-shadow": "0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.3)",
            "--dgcv-hover-transform": "none",
            "--dgcv-hover-transition": "background-color 0.15s ease",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.4)",
        },
    ),
    "chalkboard_black": ThemeConfig(
        bg_primary="#3c3c3c",
        bg_surface="#2b2b2b",
        bg_alt="#454545",
        bg_hover="#1c1c1c",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#8b4513",
        bg_action="#9e9e9e",
        text_on_action="#000000",
        bg_action_hover="#bdbdbd",
        bg_error="#e53935",
        text_on_error="#ffffff",
        bg_success="#43a047",
        text_on_success="#ffffff",
        border_alt="#616161",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(255,255,255,0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "chalkboard_blue": ThemeConfig(
        bg_primary="#305e91",
        bg_surface="#2c528c",
        bg_alt="#457bc1",
        bg_hover="#193a71",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#8b4513",
        bg_action="#64b5f6",
        text_on_action="#0d47a1",
        bg_action_hover="#90caf9",
        bg_error="#ef5350",
        text_on_error="#ffffff",
        bg_success="#66bb6a",
        text_on_success="#ffffff",
        border_alt="#2196f3",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(255,255,255,0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "chalkboard_green": ThemeConfig(
        bg_primary="#3c6e47",
        bg_surface="#355e3b",
        bg_alt="#4a7c59",
        bg_hover="#2c5a33",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#8b4513",
        bg_action="#81c784",
        text_on_action="#1b5e20",
        bg_action_hover="#a5d6a7",
        bg_error="#e57373",
        text_on_error="#ffffff",
        bg_success="#69f0ae",
        text_on_success="#000000",
        border_alt="#66bb6a",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(255,255,255,0.25)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "chalkboard_purple": ThemeConfig(
        bg_primary="#673ab7",
        bg_surface="#7e57c2",
        bg_alt="#9575cd",
        bg_hover="#512da8",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#8b4513",
        bg_action="#b388ff",
        text_on_action="#311b92",
        bg_action_hover="#d1c4e9",
        bg_error="#ff5252",
        text_on_error="#ffffff",
        bg_success="#69f0ae",
        text_on_success="#000000",
        border_alt="#b39ddb",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(255,255,255,0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "chalkboard_red": ThemeConfig(
        bg_primary="#822626",
        bg_surface="#731919",
        bg_alt="#6e1e1e",
        bg_hover="#4a1010",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#8b4513",
        bg_action="#e57373",
        text_on_action="#4a1010",
        bg_action_hover="#ef9a9a",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#81c784",
        text_on_success="#000000",
        border_alt="#9a3030",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(255,255,255,0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "chalkboard_teal": ThemeConfig(
        bg_primary="#004d40",
        bg_surface="#00897b",
        bg_alt="#00695c",
        bg_hover="#004d40",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#8b4513",
        bg_action="#4db6ac",
        text_on_action="#004d40",
        bg_action_hover="#80cbc4",
        bg_error="#e53935",
        text_on_error="#ffffff",
        bg_success="#43a047",
        text_on_success="#ffffff",
        border_alt="#26a69a",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(255,255,255,0.25)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "chalkboard_yellow": ThemeConfig(
        bg_primary="#f2c849",
        bg_surface="#d4a017",
        bg_alt="#e8c04a",
        bg_hover="#cfa524",
        text_main="#000000",
        text_heading="#000000",
        text_hover="#000000",
        border_main="#8b4513",
        bg_action="#f57f17",
        text_on_action="#ffffff",
        bg_action_hover="#fbc02d",
        bg_error="#d32f2f",
        text_on_error="#ffffff",
        bg_success="#388e3c",
        text_on_success="#ffffff",
        border_alt="#f9a825",
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-text-shadow": "0 0 3px rgba(0,0,0,0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "coffee_shop": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#3e2723",
        bg_alt="#f5f5dc",
        bg_hover="#deb887",
        text_main="#3e2723",
        text_heading="#ffffff",
        text_hover="#3e2723",
        border_main="#8b4513",
        bg_action="#5d4037",
        text_on_action="#ffffff",
        bg_action_hover="#8d6e63",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#33691e",
        text_on_success="#ffffff",
        border_alt="#eaddc5",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-radius": "8px",
            "--dgcv-table-shadow": "0 0 10px rgba(62, 39, 35, 0.2)",
            "--dgcv-text-shadow": "none",
        },
    ),
    "dark_blue": ThemeConfig(
        bg_primary="#002b4f",
        bg_surface="#001f3f",
        bg_alt="#004080",
        bg_hover="#0059b3",
        text_main="#7fdbff",
        text_heading="#7fdbff",
        text_hover="#7fdbff",
        border_main="#7fdbff",
        bg_action="#0074d9",
        text_on_action="#ffffff",
        bg_action_hover="#39cccc",
        bg_error="#ff4136",
        text_on_error="#ffffff",
        bg_success="#2ecc40",
        text_on_success="#001f3f",
        border_alt="#0059b3",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dark_high_contrast": ThemeConfig(
        bg_primary="#1e1e1e",
        bg_surface="#000000",
        bg_alt="#333333",
        bg_hover="#4d4d4d",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#ffffff",
        bg_action="#ffffff",
        text_on_action="#000000",
        bg_action_hover="#cccccc",
        bg_error="#ff0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#666666",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dark_high_contrast_bright": ThemeConfig(
        bg_primary="#1e1e1e",
        bg_surface="#000000",
        bg_alt="#4d4d4d",
        bg_hover="#9370db",
        text_main="#ff69b4",
        text_heading="#00ffff",
        text_hover="#ffffff",
        border_main="#ffff00",
        bg_action="#00ffff",
        text_on_action="#000000",
        bg_action_hover="#ff00ff",
        bg_error="#ff0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#ff00ff",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 0 15px rgba(255, 105, 180, 0.5), 0 0 30px rgba(0, 255, 255, 0.3)",
            "--dgcv-text-shadow": "0 0 6px currentColor",
        },
    ),
    "dark_modern": ThemeConfig(
        bg_primary="#1c1c1c",
        bg_surface="#2c2c2c",
        bg_alt="#3a3a3a",
        bg_hover="#484848",
        text_main="#f5f5f5",
        text_heading="#f5f5f5",
        text_hover="#7fdbff",
        border_main="#444444",
        bg_action="#0074D9",
        text_on_action="#ffffff",
        bg_action_hover="#39CCCC",
        bg_error="#FF4136",
        text_on_error="#ffffff",
        bg_success="#2ECC40",
        text_on_success="#1c1c1c",
        border_alt="#555555",
    ),
    "dark_purple": ThemeConfig(
        bg_primary="#3b1a4f",
        bg_surface="#2c003e",
        bg_alt="#503a66",
        bg_hover="#4a3a57",
        text_main="#d3d3d3",
        text_heading="#e6e6e6",
        text_hover="#d3d3d3",
        border_main="#8c0099",
        bg_action="#8a2be2",
        text_on_action="#ffffff",
        bg_action_hover="#9932cc",
        bg_error="#ff1493",
        text_on_error="#ffffff",
        bg_success="#32cd32",
        text_on_success="#000000",
        border_alt="#68427d",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dessert": ThemeConfig(
        bg_primary="#ffebcd",
        bg_surface="#8b4513",
        bg_alt="#fffacd",
        bg_hover="#ffd700",
        text_main="#8b4513",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#d2b48c",
        bg_action="#d2691e",
        text_on_action="#ffffff",
        bg_action_hover="#a0522d",
        bg_error="#cd5c5c",
        text_on_error="#ffffff",
        bg_success="#6b8e23",
        text_on_success="#ffffff",
        border_alt="#ffe4b5",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #d2b48c, #ffd700, #d2b48c) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 4px 12px rgba(139, 69, 19, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "dusk_gradient": ThemeConfig(
        bg_primary="#0a080f",
        bg_surface="#130f18",
        bg_alt="#0e0b14",
        bg_hover="#cc7733",
        text_main="#ecddd4",
        text_heading="#d4a088",
        text_hover="#0a080f",
        text_alt="#c8b8cc",
        border_main="#6633aa",
        border_alt="#1e1428",
        bg_action="#6633aa",
        text_on_action="#ecddd4",
        bg_action_hover="#cc7733",
        bg_error="#cc2244",
        text_on_error="#ecddd4",
        bg_success="#2a7755",
        text_on_success="#ecddd4",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a080f 0%, #180a20 15%, #2a1040 28%, #4a1a30 40%, #7a2a20 52%, #aa5522 62%, #cc7733 72%, #ddaa55 80%, #180a20 90%, #0a080f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(102, 51, 170, 0.5), 0 0 60px rgba(204, 119, 51, 0.2), inset 0 1px 0 rgba(220, 170, 120, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(180, 100, 60, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #6633aa, #aa3366, #cc7733, #ddaa55, #cc7733, #6633aa) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "fire_gradient": ThemeConfig(
        bg_primary="#0f0700",
        bg_surface="#1a0c00",
        bg_alt="#120900",
        bg_hover="#ff6600",
        text_main="#f5ddc8",
        text_heading="#ffaa55",
        text_hover="#0f0700",
        text_alt="#e0c0a0",
        border_main="#aa3300",
        border_alt="#2a1200",
        bg_action="#aa3300",
        text_on_action="#f5ddc8",
        bg_action_hover="#ff6600",
        bg_error="#ff2200",
        text_on_error="#0f0700",
        bg_success="#336644",
        text_on_success="#f5ddc8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0700 0%, #2a0a00 12%, #4a1500 24%, #6a2200 35%, #aa3300 48%, #cc5500 58%, #ff6600 68%, #ffaa00 78%, #ffcc44 86%, #0f0700 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(170, 51, 0, 0.6), 0 0 60px rgba(255, 102, 0, 0.2), inset 0 1px 0 rgba(255, 200, 100, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(255, 120, 0, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #ff2200, #ff6600, #ffaa00, #ffcc44, #ff6600, #ff2200) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "foggy_lights": ThemeConfig(
        bg_primary="#f8f9fa",
        bg_surface="#f8f9fa",
        bg_alt="#e9ecef",
        bg_hover="#b0c4de",
        text_main="#495057",
        text_heading="#343a40",
        text_hover="#212529",
        border_main="#778899",
        bg_action="#4682B4",
        text_on_action="#ffffff",
        bg_action_hover="#5F9EA0",
        bg_error="#DC143C",
        text_on_error="#ffffff",
        bg_success="#2E8B57",
        text_on_success="#ffffff",
        border_alt="#CED4DA",
        font_family="Verdana, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "forest_floor": ThemeConfig(
        bg_primary="#1b4332",
        bg_surface="#1b4332",
        bg_alt="#2d6a4f",
        bg_hover="#b7e4c7",
        text_main="#d8f3dc",
        text_heading="#d8f3dc",
        text_hover="#081c15",
        border_main="#6b705c",
        bg_action="#40916c",
        text_on_action="#ffffff",
        bg_action_hover="#52b788",
        bg_error="#d90429",
        text_on_error="#ffffff",
        bg_success="#74c69d",
        text_on_success="#081c15",
        border_alt="#40916c",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(170deg, #1b4332 0%, #2d6a4f 50%, #1b4332 100%)",
            "--dgcv-table-shadow": "0 0 16px rgba(27, 67, 50, 0.6)",
            "--dgcv-text-shadow": "0 1px 2px rgba(0,0,0,0.4)",
        },
    ),
    "gothic": ThemeConfig(
        bg_primary="#1c1c1c",
        bg_surface="#2c0033",
        bg_alt="#330033",
        bg_hover="#660000",
        text_main="#e6e6e6",
        text_heading="#a80000",
        text_hover="#e6e6e6",
        border_main="#a80000",
        bg_action="#800000",
        text_on_action="#ffffff",
        bg_action_hover="#4d0000",
        bg_error="#ff0000",
        text_on_error="#000000",
        bg_success="#2e8b57",
        text_on_success="#ffffff",
        border_alt="#4a004a",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-image": "linear-gradient(180deg, #a80000, #4a004a, #a80000) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 20px rgba(168, 0, 0, 0.5), inset 0 0 10px rgba(74, 0, 74, 0.3)",
            "--dgcv-text-shadow": "0 0 6px rgba(168, 0, 0, 0.6)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "graffiti": ThemeConfig(
        bg_primary="#ff4500",
        bg_surface="#1e90ff",
        bg_alt="#ffa500",
        bg_hover="#39FF14",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#000000",
        border_main="#ffa500",
        bg_action="#ff00ff",
        text_on_action="#ffffff",
        bg_action_hover="#da70d6",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#ff8c00",
        font_family="Permanent Marker, cursive",
        custom_css_vars={
            "--dgcv-border-width": "3px",
            "--dgcv-border-image": "linear-gradient(90deg, #ff4500, #ff00ff, #39FF14, #1e90ff, #ff4500) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 20px rgba(30, 144, 255, 0.7), 0 0 40px rgba(255, 0, 255, 0.4)",
            "--dgcv-text-shadow": "2px 2px 0px rgba(0,0,0,0.5)",
            "--dgcv-hover-transform": "rotate(0.5deg) scale(1.002)",
        },
    ),
    "graph_paper": ThemeConfig(
        bg_primary="#f2faff",
        bg_surface="#ffffff",
        bg_alt="#ffffff",
        bg_hover="#e6f7ff",
        text_main="#000000",
        text_heading="#000000",
        text_hover="#000000",
        border_main="#cccccc",
        bg_action="#0066cc",
        text_on_action="#ffffff",
        bg_action_hover="#3385ff",
        bg_error="#cc0000",
        text_on_error="#ffffff",
        bg_success="#009900",
        text_on_success="#ffffff",
        border_alt="#e6e6e6",
        font_family="Roboto, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "gruvbox_colorful": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#689d6a",
        bg_alt="#ebdbb2",
        bg_hover="#d3869b",
        text_main="#9d0006",
        text_heading="#fbf1c7",
        text_hover="#282828",
        text_alt="#076678",
        border_main="#427b58",
        bg_action="#d65d0e",
        text_on_action="#fbf1c7",
        bg_action_hover="#fe8019",
        bg_error="#cc241d",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        border_alt="#d5c4a1",
        font_family="Comic Sans MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.1)",
        },
    ),
    "gruvbox_dark": ThemeConfig(
        bg_primary="#282828",
        bg_surface="#1d2021",
        bg_alt="#32302f",
        bg_hover="#3c3836",
        text_main="#ebdbb2",
        text_alt="#83a598",
        text_heading="#fabd2f",
        text_hover="#8ec07c",
        border_main="#504945",
        bg_action="#458588",
        text_on_action="#ebdbb2",
        bg_action_hover="#83a598",
        bg_error="#cc241d",
        text_on_error="#ebdbb2",
        bg_success="#98971a",
        text_on_success="#1d2021",
        border_alt="#665c54",
        font_family="monospace",
    ),
    "gruvbox_light": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#ebdbb2",
        bg_alt="#f2e5bc",
        bg_hover="#bdae93",
        text_main="#3c3836",
        text_heading="#9d0006",
        text_hover="#076678",
        border_main="#d5c4a1",
        bg_action="#076678",
        text_on_action="#fbf1c7",
        bg_action_hover="#458588",
        bg_error="#9d0006",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        border_alt="#e8d8b0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.05)",
        },
    ),
    "gruvbox_reds": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#af3a03",
        bg_alt="#d5c4a1",
        bg_hover="#fabd2f",
        text_main="#076678",
        text_heading="#fbf1c7",
        text_hover="#282828",
        text_alt="#9d0006",
        border_main="#d65d0e",
        bg_action="#8f3f71",
        text_on_action="#fbf1c7",
        bg_action_hover="#b16286",
        bg_error="#cc241d",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        border_alt="#bdae93",
        font_family="Comic Sans MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(215, 153, 33, 0.4)",
        },
    ),
    "lunar": ThemeConfig(
        bg_primary="#dfe7ec",
        bg_surface="#394b59",
        bg_alt="#f5f8fa",
        bg_hover="#cbd6e2",
        text_main="#333333",
        text_heading="#b0c4de",
        text_hover="#333333",
        border_main="#394b59",
        bg_action="#5c7080",
        text_on_action="#ffffff",
        bg_action_hover="#738694",
        bg_error="#db3737",
        text_on_error="#ffffff",
        bg_success="#0f9960",
        text_on_success="#ffffff",
        border_alt="#e1e8ed",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "matte_amber": ThemeConfig(
        bg_primary="#0f0c07",
        bg_surface="#181208",
        bg_alt="#120e06",
        bg_hover="#cc8822",
        text_main="#f0e4c8",
        text_heading="#e8c070",
        text_hover="#0f0c07",
        text_alt="#d8cc99",
        border_main="#7a5510",
        border_alt="#251a08",
        bg_action="#7a5510",
        text_on_action="#f0e4c8",
        bg_action_hover="#cc8822",
        bg_error="#cc3322",
        text_on_error="#f0e4c8",
        bg_success="#336644",
        text_on_success="#f0e4c8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0c07 0%, #201508 50%, #0f0c07 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(240, 200, 100, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_blue": ThemeConfig(
        bg_primary="#080c14",
        bg_surface="#0c1220",
        bg_alt="#0a0f1a",
        bg_hover="#4477cc",
        text_main="#d8e4f8",
        text_heading="#99bbee",
        text_hover="#080c14",
        text_alt="#b0c8e8",
        border_main="#1e3a6e",
        border_alt="#111a2e",
        bg_action="#1e3a6e",
        text_on_action="#d8e4f8",
        bg_action_hover="#4477cc",
        bg_error="#cc3344",
        text_on_error="#ffffff",
        bg_success="#2a8855",
        text_on_success="#d8e4f8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #080c14 0%, #0c1a2e 50%, #080c14 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(100, 160, 255, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_cyan": ThemeConfig(
        bg_primary="#070f0f",
        bg_surface="#0a1818",
        bg_alt="#081212",
        bg_hover="#1aaa99",
        text_main="#cceee8",
        text_heading="#55ccbb",
        text_hover="#070f0f",
        text_alt="#99ddd4",
        border_main="#136655",
        border_alt="#0f2020",
        bg_action="#136655",
        text_on_action="#cceee8",
        bg_action_hover="#1aaa99",
        bg_error="#cc3344",
        text_on_error="#cceee8",
        bg_success="#1aaa99",
        text_on_success="#070f0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0f 0%, #0c2020 50%, #070f0f 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(85, 220, 200, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_green": ThemeConfig(
        bg_primary="#070f0a",
        bg_surface="#0a1810",
        bg_alt="#081209",
        bg_hover="#22aa55",
        text_main="#d0eedd",
        text_heading="#66cc88",
        text_hover="#070f0a",
        text_alt="#a8d8bb",
        border_main="#1a5530",
        border_alt="#102018",
        bg_action="#1a5530",
        text_on_action="#d0eedd",
        bg_action_hover="#22aa55",
        bg_error="#cc3322",
        text_on_error="#d0eedd",
        bg_success="#22aa55",
        text_on_success="#070f0a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0a 0%, #0c2015 50%, #070f0a 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(100, 220, 140, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_red": ThemeConfig(
        bg_primary="#0f0907",
        bg_surface="#180f0a",
        bg_alt="#120b08",
        bg_hover="#cc4422",
        text_main="#f0ddd8",
        text_heading="#e8a090",
        text_hover="#0f0907",
        text_alt="#d8c0b8",
        border_main="#7a2a1a",
        border_alt="#2a1510",
        bg_action="#7a2a1a",
        text_on_action="#f0ddd8",
        bg_action_hover="#cc4422",
        bg_error="#ff4422",
        text_on_error="#0f0907",
        bg_success="#336644",
        text_on_success="#f0ddd8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0907 0%, #200e08 50%, #0f0907 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(240, 160, 120, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_slate": ThemeConfig(
        bg_primary="#090a0c",
        bg_surface="#111318",
        bg_alt="#0c0e12",
        bg_hover="#5577aa",
        text_main="#d8dde8",
        text_heading="#99aabf",
        text_hover="#090a0c",
        text_alt="#b0bcc8",
        border_main="#2a3344",
        border_alt="#181c24",
        bg_action="#2a3344",
        text_on_action="#d8dde8",
        bg_action_hover="#5577aa",
        bg_error="#cc3344",
        text_on_error="#d8dde8",
        bg_success="#2a6644",
        text_on_success="#d8dde8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #090a0c 0%, #111820 50%, #090a0c 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(150, 180, 220, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_slate_soft": ThemeConfig(
        bg_primary="#111214",
        bg_surface="#18191d",
        bg_alt="#141518",
        bg_hover="#5577aa",
        text_main="#c8cdd6",
        text_heading="#c8cdd6",
        text_hover="#111214",
        text_alt="#8892a0",
        border_main="#2a2d34",
        border_alt="#1e2026",
        bg_action="#2a2d34",
        text_on_action="#c8cdd6",
        bg_action_hover="#5577aa",
        bg_error="#cc3344",
        text_on_error="#c8cdd6",
        bg_success="#2a6644",
        text_on_success="#c8cdd6",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #18191d 0%, #1c1e24 50%, #18191d 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(150, 180, 220, 0.04)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_violet": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#100e18",
        bg_alt="#0d0b14",
        bg_hover="#9966ff",
        text_main="#e0d8f0",
        text_heading="#c8b8f0",
        text_hover="#0a0a0f",
        text_alt="#c0b0e8",
        border_main="#5533aa",
        border_alt="#1a1428",
        bg_action="#5533aa",
        text_on_action="#ffffff",
        bg_action_hover="#7744cc",
        bg_error="#cc2244",
        text_on_error="#ffffff",
        bg_success="#44aa77",
        text_on_success="#0a0a0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a0a0f 0%, #120a20 40%, #1a1030 70%, #0a0a0f 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(180, 140, 255, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matisse": ThemeConfig(
        bg_primary="#b2d5eb",
        bg_surface="#ee9e84",
        bg_alt="#41b199",
        bg_hover="#df7d8c",
        text_main="#d45659",
        text_heading="#004d00",
        text_alt="#ffd6cc",
        text_hover="#b2d5eb",
        border_main="#ee9e84",
        bg_action="#d45659",
        text_on_action="#ffffff",
        bg_action_hover="#c0392b",
        bg_error="#900C3F",
        text_on_error="#ffffff",
        bg_success="#27ae60",
        text_on_success="#ffffff",
        border_alt="#318c78",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 4px 12px rgba(238, 158, 132, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "matisse_light": ThemeConfig(
        bg_primary="#e6ffe6",
        bg_surface="#ffcc00",
        bg_alt="#ffd6cc",
        bg_hover="#ffe680",
        text_main="#004d00",
        text_heading="#004d00",
        text_hover="#004d00",
        border_main="#004d00",
        bg_action="#008000",
        text_on_action="#ffffff",
        bg_action_hover="#009900",
        bg_error="#ff3333",
        text_on_error="#ffffff",
        bg_success="#00cc00",
        text_on_success="#ffffff",
        border_alt="#ffb399",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 2px 8px rgba(255, 204, 0, 0.3)",
        },
    ),
    "mist": ThemeConfig(
        bg_primary="#e0f7fa",
        bg_surface="#00796b",
        bg_alt="#b2ebf2",
        bg_hover="#a7c4c7",
        text_main="#004d40",
        text_heading="#ffffff",
        text_hover="#004d40",
        border_main="#b2dfdb",
        bg_action="#00838f",
        text_on_action="#ffffff",
        bg_action_hover="#00bcd4",
        bg_error="#c62828",
        text_on_error="#ffffff",
        bg_success="#2e7d32",
        text_on_success="#ffffff",
        border_alt="#80deea",
        font_family="Cormorant, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #e0f7fa 0%, #b2ebf2 60%, #e0f7fa 100%)",
            "--dgcv-special-text": "#004d40",
            "--dgcv-table-shadow": "0 0 20px rgba(0, 121, 107, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "monet": ThemeConfig(
        bg_primary="#f7f9e4",
        bg_surface="#2a5d67",
        bg_alt="#c6f3d8",
        bg_hover="#d0ece7",
        text_main="#2a5d67",
        text_heading="#ffffff",
        text_hover="#2a5d67",
        border_main="#2a5d67",
        bg_action="#1a3d44",
        text_on_action="#ffffff",
        bg_action_hover="#3b7a86",
        bg_error="#d9534f",
        text_on_error="#ffffff",
        bg_success="#5cb85c",
        text_on_success="#ffffff",
        border_alt="#b2dfcc",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 2px 8px rgba(42, 93, 103, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "neutral_dark": ThemeConfig(
        bg_primary="#121212",
        bg_surface="#121212",
        bg_alt="#1e1e1e",
        bg_hover="#252525",
        text_main="#e0e0e0",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#333333",
        bg_action="#0288d1",
        text_on_action="#ffffff",
        bg_action_hover="#03a9f4",
        bg_error="#cf6679",
        text_on_error="#000000",
        bg_success="#81c784",
        text_on_success="#000000",
        border_alt="#424242",
        font_family="system-ui, -apple-system, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "newspaper": ThemeConfig(
        bg_primary="#f4f4f4",
        bg_surface="#fafafa",
        bg_alt="#fafafa",
        bg_hover="#e0e0e0",
        text_main="#000000",
        text_heading="#000000",
        text_hover="#000000",
        border_main="#333333",
        bg_action="#000000",
        text_on_action="#ffffff",
        bg_action_hover="#4d4d4d",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#006400",
        text_on_success="#ffffff",
        border_alt="#808080",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0px 2px 5px rgba(0, 0, 0, 0.1)",
        },
    ),
    "ocean": ThemeConfig(
        bg_primary="#4682b4",
        bg_surface="#1e90ff",
        bg_alt="#87cefa",
        bg_hover="#1e90ff",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#1e90ff",
        bg_action="#000080",
        text_on_action="#ffffff",
        bg_action_hover="#0000cd",
        bg_error="#dc143c",
        text_on_error="#ffffff",
        bg_success="#20b2aa",
        text_on_success="#ffffff",
        border_alt="#5f9ea0",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(90deg, #00bfff, #1e90ff, #4682b4) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-text-shadow": "0px 0px 4px rgba(0, 191, 255, 0.6)",
            "--dgcv-table-shadow": "0 4px 12px rgba(30, 144, 255, 0.6)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "ocean_gradient": ThemeConfig(
        bg_primary="#020d1a",
        bg_surface="#041424",
        bg_alt="#041020",
        bg_hover="#00e5ff",
        text_main="#b0eaf5",
        text_heading="#b0eaf5",
        text_hover="#020d1a",
        text_alt="#80d0e8",
        border_main="#0077a8",
        border_alt="#0a2a3a",
        bg_action="#006994",
        text_on_action="#ffffff",
        bg_action_hover="#0090c8",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#020d1a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "radial-gradient(ellipse at top, #0a2a4a 0%, #041424 40%, #020810 100%)",
            "--dgcv-table-shadow": "0 0 30px rgba(0, 119, 168, 0.5), 0 0 80px rgba(0, 229, 255, 0.15)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(0, 229, 255, 0.5)",
        },
    ),
    "oil": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#12101a",
        bg_alt="#0d0d14",
        bg_hover="#cc44ff",
        text_main="#e8e0f0",
        text_heading="#e8e0f0",
        text_hover="#0a0a0f",
        text_alt="#d0c8e8",
        border_main="#6644aa",
        border_alt="#1a1428",
        bg_action="#2244cc",
        text_on_action="#ffffff",
        bg_action_hover="#4466ff",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#0a0a0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 15%, #0a1a3a 30%, #0a2a1a 45%, #1a2a0a 55%, #2a1a0a 65%, #2a0a1a 75%, #1a0a2e 88%, #0a0a0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(102, 68, 170, 0.6), 0 0 60px rgba(34, 68, 204, 0.3), inset 0 1px 0 rgba(200, 150, 255, 0.15)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(180, 120, 255, 0.7)",
            "--dgcv-border-image": "linear-gradient(135deg, #ff00ff, #0044ff, #00ffcc, #ffaa00, #ff00ff) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_amber": ThemeConfig(
        bg_primary="#0f0c07",
        bg_surface="#181208",
        bg_alt="#120e06",
        bg_hover="#cc8822",
        text_main="#f0e4c8",
        text_heading="#e8c070",
        text_hover="#0f0c07",
        text_alt="#d8cc99",
        border_main="#7a5510",
        border_alt="#251a08",
        bg_action="#7a5510",
        text_on_action="#f0e4c8",
        bg_action_hover="#cc8822",
        bg_error="#cc3322",
        text_on_error="#f0e4c8",
        bg_success="#336644",
        text_on_success="#f0e4c8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0c07 0%, #201508 50%, #0f0c07 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(122, 85, 16, 0.6), inset 0 1px 0 rgba(240, 200, 100, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(204, 136, 34, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #cc8822, #7a5510, #cc8822) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_blue": ThemeConfig(
        bg_primary="#080c14",
        bg_surface="#0c1220",
        bg_alt="#0a0f1a",
        bg_hover="#4477cc",
        text_main="#d8e4f8",
        text_heading="#99bbee",
        text_hover="#080c14",
        text_alt="#b0c8e8",
        border_main="#1e3a6e",
        border_alt="#111a2e",
        bg_action="#1e3a6e",
        text_on_action="#d8e4f8",
        bg_action_hover="#4477cc",
        bg_error="#cc3344",
        text_on_error="#ffffff",
        bg_success="#2a8855",
        text_on_success="#d8e4f8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #080c14 0%, #0c1a2e 50%, #080c14 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(30, 58, 110, 0.6), inset 0 1px 0 rgba(100, 160, 255, 0.1)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(68, 119, 204, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #4477cc, #1e3a6e, #4477cc) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_cyan": ThemeConfig(
        bg_primary="#070f0f",
        bg_surface="#0a1818",
        bg_alt="#081212",
        bg_hover="#1aaa99",
        text_main="#cceee8",
        text_heading="#55ccbb",
        text_hover="#070f0f",
        text_alt="#99ddd4",
        border_main="#136655",
        border_alt="#0f2020",
        bg_action="#136655",
        text_on_action="#cceee8",
        bg_action_hover="#1aaa99",
        bg_error="#cc3344",
        text_on_error="#cceee8",
        bg_success="#1aaa99",
        text_on_success="#070f0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0f 0%, #0c2020 50%, #070f0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(19, 102, 85, 0.6), inset 0 1px 0 rgba(85, 220, 200, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(26, 170, 153, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #1aaa99, #136655, #1aaa99) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_green": ThemeConfig(
        bg_primary="#070f0a",
        bg_surface="#0a1810",
        bg_alt="#081209",
        bg_hover="#22aa55",
        text_main="#d0eedd",
        text_heading="#66cc88",
        text_hover="#070f0a",
        text_alt="#a8d8bb",
        border_main="#1a5530",
        border_alt="#102018",
        bg_action="#1a5530",
        text_on_action="#d0eedd",
        bg_action_hover="#22aa55",
        bg_error="#cc3322",
        text_on_error="#d0eedd",
        bg_success="#22aa55",
        text_on_success="#070f0a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0a 0%, #0c2015 50%, #070f0a 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(26, 85, 48, 0.6), inset 0 1px 0 rgba(100, 220, 140, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(34, 170, 85, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #22aa55, #1a5530, #22aa55) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_red": ThemeConfig(
        bg_primary="#0f0907",
        bg_surface="#180f0a",
        bg_alt="#120b08",
        bg_hover="#cc4422",
        text_main="#f0ddd8",
        text_heading="#e8a090",
        text_hover="#0f0907",
        text_alt="#d8c0b8",
        border_main="#7a2a1a",
        border_alt="#2a1510",
        bg_action="#7a2a1a",
        text_on_action="#f0ddd8",
        bg_action_hover="#cc4422",
        bg_error="#ff4422",
        text_on_error="#0f0907",
        bg_success="#336644",
        text_on_success="#f0ddd8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0907 0%, #200e08 50%, #0f0907 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(122, 42, 26, 0.6), inset 0 1px 0 rgba(240, 160, 120, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(204, 68, 34, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #cc4422, #7a2a1a, #cc4422) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_slate": ThemeConfig(
        bg_primary="#090a0c",
        bg_surface="#111318",
        bg_alt="#0c0e12",
        bg_hover="#5577aa",
        text_main="#d8dde8",
        text_heading="#99aabf",
        text_hover="#090a0c",
        text_alt="#b0bcc8",
        border_main="#2a3344",
        border_alt="#181c24",
        bg_action="#2a3344",
        text_on_action="#d8dde8",
        bg_action_hover="#5577aa",
        bg_error="#cc3344",
        text_on_error="#d8dde8",
        bg_success="#2a6644",
        text_on_success="#d8dde8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #090a0c 0%, #111820 50%, #090a0c 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(42, 51, 68, 0.7), inset 0 1px 0 rgba(150, 180, 220, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(85, 119, 170, 0.4)",
            "--dgcv-border-image": "linear-gradient(135deg, #5577aa, #2a3344, #5577aa) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_violet": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#100e18",
        bg_alt="#0d0b14",
        bg_hover="#9966ff",
        text_main="#e0d8f0",
        text_heading="#c8b8f0",
        text_hover="#0a0a0f",
        text_alt="#c0b0e8",
        border_main="#5533aa",
        border_alt="#1a1428",
        bg_action="#5533aa",
        text_on_action="#ffffff",
        bg_action_hover="#7744cc",
        bg_error="#cc2244",
        text_on_error="#ffffff",
        bg_success="#44aa77",
        text_on_success="#0a0a0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a0a0f 0%, #120a20 40%, #1a1030 70%, #0a0a0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(85, 51, 170, 0.5), inset 0 1px 0 rgba(180, 140, 255, 0.1)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(153, 102, 255, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #9966ff, #5533aa, #9966ff) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "outer_space": ThemeConfig(
        bg_primary="#1d1d1d",
        bg_surface="#2b2d42",
        bg_alt="#333366",
        bg_hover="#4b0082",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#6c63ff",
        bg_action="#6c63ff",
        text_on_action="#ffffff",
        bg_action_hover="#8c82ff",
        bg_error="#ff4d4d",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#1d1d1d",
        border_alt="#404080",
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "radial-gradient(ellipse at top, #2b2d42 0%, #1d1d1d 60%, #0a0a14 100%)",
            "--dgcv-table-shadow": "0 4px 20px rgba(75, 0, 130, 0.8), 0 0 60px rgba(108, 99, 255, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(108, 99, 255, 0.8)",
        },
    ),
    "paper_graphite": ThemeConfig(
        bg_primary="#f5f4f2",
        bg_surface="#eceae6",
        bg_alt="#f0eeeb",
        bg_hover="#6b7280",
        text_main="#1c1c1e",
        text_heading="#2c2c30",
        text_hover="#f5f4f2",
        text_alt="#4a4a50",
        border_main="#b0aaa0",
        border_alt="#d8d4ce",
        bg_action="#4a4a50",
        text_on_action="#f5f4f2",
        bg_action_hover="#6b7280",
        bg_error="#cc3344",
        text_on_error="#f5f4f2",
        bg_success="#3a6644",
        text_on_success="#f5f4f2",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #eceae6 0%, #e4e0da 50%, #eceae6 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_ivory": ThemeConfig(
        bg_primary="#faf7f0",
        bg_surface="#f0ebe0",
        bg_alt="#f5f1e8",
        bg_hover="#7a6a50",
        text_main="#1e1a14",
        text_heading="#2e2618",
        text_hover="#faf7f0",
        text_alt="#4a4030",
        border_main="#c8b89a",
        border_alt="#e0d8c8",
        bg_action="#5a4a32",
        text_on_action="#faf7f0",
        bg_action_hover="#7a6a50",
        bg_error="#cc3344",
        text_on_error="#faf7f0",
        bg_success="#3a6644",
        text_on_success="#faf7f0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #f0ebe0 0%, #e8e0d0 50%, #f0ebe0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_sage": ThemeConfig(
        bg_primary="#f2f5f0",
        bg_surface="#e4ece0",
        bg_alt="#eaf0e6",
        bg_hover="#4a6a50",
        text_main="#141c12",
        text_heading="#1e2e1a",
        text_hover="#f2f5f0",
        text_alt="#384a34",
        border_main="#90aa88",
        border_alt="#c8d8c0",
        bg_action="#304a30",
        text_on_action="#f2f5f0",
        bg_action_hover="#4a6a50",
        bg_error="#cc3344",
        text_on_error="#f2f5f0",
        bg_success="#3a6644",
        text_on_success="#f2f5f0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #e4ece0 0%, #d8e8d0 50%, #e4ece0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_slate": ThemeConfig(
        bg_primary="#f2f4f7",
        bg_surface="#e4e8f0",
        bg_alt="#eaecf2",
        bg_hover="#4a6080",
        text_main="#141820",
        text_heading="#202838",
        text_hover="#f2f4f7",
        text_alt="#384458",
        border_main="#9aaac0",
        border_alt="#ccd4e0",
        bg_action="#304460",
        text_on_action="#f2f4f7",
        bg_action_hover="#4a6080",
        bg_error="#cc3344",
        text_on_error="#f2f4f7",
        bg_success="#3a6644",
        text_on_success="#f2f4f7",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #e4e8f0 0%, #d8dfec 50%, #e4e8f0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_white": ThemeConfig(
        bg_primary="#fafafa",
        bg_surface="#f0f0f0",
        bg_alt="#f5f5f5",
        bg_hover="#444444",
        text_main="#111111",
        text_heading="#222222",
        text_hover="#fafafa",
        text_alt="#555555",
        border_main="#cccccc",
        border_alt="#e4e4e4",
        bg_action="#333333",
        text_on_action="#fafafa",
        bg_action_hover="#444444",
        bg_error="#cc3344",
        text_on_error="#fafafa",
        bg_success="#3a6644",
        text_on_success="#fafafa",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 50%, #f0f0f0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "parchment": ThemeConfig(
        bg_primary="#f3e2c7",
        bg_surface="#5c4327",
        bg_alt="#f9f2e7",
        bg_hover="#e8d4af",
        text_main="#5c4327",
        text_heading="#ffffff",
        text_hover="#4a301f",
        border_main="#5c4327",
        bg_action="#8b5a2b",
        text_on_action="#ffffff",
        bg_action_hover="#a0522d",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        border_alt="#ebd4b3",
        font_family="Dancing Script, cursive",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #f3e2c7 0%, #ecdec0 50%, #f3e2c7 100%)",
            "--dgcv-special-text": "#5c4327",
            "--dgcv-table-shadow": "0 0 15px rgba(92, 67, 39, 0.5), inset 0 0 30px rgba(92, 67, 39, 0.05)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.4)",
        },
    ),
    "picasso_blue": ThemeConfig(
        bg_primary="#eaf2f8",
        bg_surface="#154360",
        bg_alt="#d6eaf8",
        bg_hover="#aed6f1",
        text_main="#154360",
        text_heading="#ffffff",
        text_hover="#154360",
        border_main="#154360",
        bg_action="#1f618d",
        text_on_action="#ffffff",
        bg_action_hover="#2980b9",
        bg_error="#c0392b",
        text_on_error="#ffffff",
        bg_success="#27ae60",
        text_on_success="#ffffff",
        border_alt="#85c1e9",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 2px 10px rgba(21, 67, 96, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "presentation": ThemeConfig(
        bg_primary="#ffffff",
        bg_surface="#007acc",
        bg_alt="#e0f4ff",
        bg_hover="#cce5ff",
        text_main="#000000",
        text_heading="#ffffff",
        text_hover="#000000",
        border_main="#007acc",
        bg_action="#005c99",
        text_on_action="#ffffff",
        bg_action_hover="#004080",
        bg_error="#cc0000",
        text_on_error="#ffffff",
        bg_success="#009933",
        text_on_success="#ffffff",
        border_alt="#99ccff",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "purple_black_grad": ThemeConfig(
        bg_primary="#161312",
        bg_surface="#272020",
        bg_alt="#322a29",
        bg_hover="#5632cc",
        text_main="#f3f1f0",
        text_heading="#a1ffce",
        text_hover="#ffffff",
        text_alt="#f4f0ef",
        border_main="#895f5b",
        border_alt="#978280",
        bg_action="#5632cc",
        text_on_action="#ffffff",
        bg_action_hover="#4428a3",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00ff87",
        text_on_success="#000000",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-hover-transform": "rotate(1deg) scale(1.01)",
            "--dgcv-hover-transition": "all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
            "--dgcv-special-background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "--dgcv-table-shadow": "0 10px 30px rgba(0,0,0,0.5)",
        },
    ),
    "purple_green_grad": ThemeConfig(
        bg_primary="#141216",
        bg_surface="#242027",
        bg_alt="#2f2932",
        bg_hover="#b5cc32",
        text_main="#f2f0f3",
        text_heading="#a1ffce",
        text_hover="#ffffff",
        text_alt="#f3eff4",
        border_main="#7a5b89",
        border_alt="#908097",
        bg_action="#b5cc32",
        text_on_action="#141216",
        bg_action_hover="#91a328",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00ff87",
        text_on_success="#000000",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-hover-transform": "rotate(1deg) scale(1.01)",
            "--dgcv-hover-transition": "all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
            "--dgcv-special-background": "linear-gradient(135deg, #2d4a1e 0%, #4a3d6b 50%, #1e2d4a 100%)",
            "--dgcv-table-shadow": "0 10px 30px rgba(0,0,0,0.5)",
        },
    ),
    "purples": ThemeConfig(
        bg_primary="#2D2B55",
        bg_surface="#3F3F7A",
        bg_alt="#4A4A8A",
        bg_hover="#6A5ACD",
        text_main="#F5F5F5",
        text_heading="#E6E6FA",
        text_hover="#F5F5F5",
        border_main="#6A5ACD",
        bg_action="#8A2BE2",
        text_on_action="#ffffff",
        bg_action_hover="#9370DB",
        bg_error="#FF1493",
        text_on_error="#ffffff",
        bg_success="#00FA9A",
        text_on_success="#2D2B55",
        border_alt="#7B68EE",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg, #2D2B55 0%, #3F3F7A 50%, #2D2B55 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(106, 90, 205, 0.7), 0 0 60px rgba(138, 43, 226, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(106, 90, 205, 0.5)",
        },
    ),
    "radial_blues": ThemeConfig(
        bg_primary="#1a1a2e",
        bg_surface="#2c2c54",
        bg_alt="#0f3460",
        bg_hover="#22a6b3",
        text_main="#e0e0e0",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#ffffff",
        bg_action="#e94560",
        text_on_action="#ffffff",
        bg_action_hover="#ff6b81",
        bg_error="#ff4757",
        text_on_error="#ffffff",
        bg_success="#2ed573",
        text_on_success="#1a1a2e",
        border_alt="#16213e",
        font_family="Trebuchet MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "radial-gradient(circle, #1a1a2e, #0f3460)",
            "--dgcv-table-shadow": "0 0 10px rgba(34, 166, 179, 0.8)",
        },
    ),
    "rain_forest": ThemeConfig(
        bg_primary="#90ee90",
        bg_surface="#228b22",
        bg_alt="#98fb98",
        bg_hover="#006400",
        text_main="#2f4f4f",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#228b22",
        bg_action="#008000",
        text_on_action="#ffffff",
        bg_action_hover="#006400",
        bg_error="#b22222",
        text_on_error="#ffffff",
        bg_success="#32cd32",
        text_on_success="#000000",
        border_alt="#3cb371",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #006400, #32cd32, #228b22) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-special-background": "linear-gradient(170deg, #90ee90 0%, #98fb98 50%, #90ee90 100%)",
            "--dgcv-special-text": "#2f4f4f",
            "--dgcv-text-shadow": "0px 0px 4px rgba(0, 100, 0, 0.4)",
            "--dgcv-table-shadow": "0 4px 12px rgba(0, 100, 0, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "rembrandt": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#523d2e",
        bg_alt="#f7e6d5",
        bg_hover="#d4b996",
        text_main="#523d2e",
        text_heading="#ffffff",
        text_hover="#523d2e",
        border_main="#6c4f3d",
        bg_action="#38291f",
        text_on_action="#ffffff",
        bg_action_hover="#6c4f3d",
        bg_error="#a52a2a",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        border_alt="#ebd4b3",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #fffaf0 0%, #f7e6d5 60%, #fffaf0 100%)",
            "--dgcv-special-text": "#523d2e",
            "--dgcv-table-shadow": "0 4px 16px rgba(82, 61, 46, 0.3), inset 0 0 40px rgba(82, 61, 46, 0.05)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.5)",
        },
    ),
    "safari": ThemeConfig(
        bg_primary="#f3e6d4",
        bg_surface="#8b6e4e",
        bg_alt="#e4d7c5",
        bg_hover="#d1b998",
        text_main="#4b4b4b",
        text_heading="#f7d94c",
        text_hover="#4b4b4b",
        border_main="#8b6e4e",
        bg_action="#6b8e23",
        text_on_action="#ffffff",
        bg_action_hover="#556b2f",
        bg_error="#b22222",
        text_on_error="#ffffff",
        bg_success="#228b22",
        text_on_success="#ffffff",
        border_alt="#d2b48c",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 2px 8px rgba(139, 110, 78, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "sakura": ThemeConfig(
        bg_primary="#fff0f5",
        bg_surface="#965357",
        bg_alt="#ffeef8",
        bg_hover="#f7cac9",
        text_main="#8a2a2b",
        text_heading="#ffffff",
        text_hover="#8a2a2b",
        border_main="#f7cac9",
        bg_action="#e64980",
        text_on_action="#ffffff",
        bg_action_hover="#f06595",
        bg_error="#c92a2a",
        text_on_error="#ffffff",
        bg_success="#2b8a3e",
        text_on_success="#ffffff",
        border_alt="#ffb3c6",
        font_family="Palatino, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(170deg, #965357 0%, #916063 60%, #965357 100%)",
            "--dgcv-special-text": "#ffffff",
            "--dgcv-table-shadow": "0 0 16px rgba(214, 51, 108, 0.25)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.6)",
        },
    ),
    "sci_fi": ThemeConfig(
        bg_primary="#000000",
        bg_surface="#001f3f",
        bg_alt="#011627",
        bg_hover="#7fdbff",
        text_main="#7fdbff",
        text_heading="#7fdbff",
        text_hover="#001f3f",
        border_main="#7fdbff",
        bg_action="#0074d9",
        text_on_action="#ffffff",
        bg_action_hover="#39cccc",
        bg_error="#ff4136",
        text_on_error="#ffffff",
        bg_success="#2ecc40",
        text_on_success="#000000",
        border_alt="#004080",
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 20px rgba(127, 219, 255, 0.9), 0 0 60px rgba(127, 219, 255, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(127, 219, 255, 0.8)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "sci_fi_green": ThemeConfig(
        bg_primary="#000000",
        bg_surface="#001f00",
        bg_alt="#012200",
        bg_hover="#39FF14",
        text_main="#39FF14",
        text_heading="#39FF14",
        text_hover="#001f00",
        border_main="#39FF14",
        bg_action="#00FF00",
        text_on_action="#000000",
        bg_action_hover="#32CD32",
        bg_error="#FF0000",
        text_on_error="#ffffff",
        bg_success="#00FA9A",
        text_on_success="#000000",
        border_alt="#003300",
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 20px rgba(57, 255, 20, 0.9), 0 0 60px rgba(57, 255, 20, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(57, 255, 20, 0.9)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "sci_fi_magenta": ThemeConfig(
        bg_primary="#1A001F",
        bg_surface="#300033",
        bg_alt="#3A0040",
        bg_hover="#FF00FF",
        text_main="#FF00FF",
        text_heading="#FF00FF",
        text_hover="#300033",
        border_main="#FF00FF",
        bg_action="#8A2BE2",
        text_on_action="#ffffff",
        bg_action_hover="#9400D3",
        bg_error="#FF4500",
        text_on_error="#ffffff",
        bg_success="#32CD32",
        text_on_success="#000000",
        border_alt="#4B0082",
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 20px rgba(255, 0, 255, 0.9), 0 0 60px rgba(255, 0, 255, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(255, 0, 255, 0.9)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "sci_fi_orange": ThemeConfig(
        bg_primary="#1A0D00",
        bg_surface="#331900",
        bg_alt="#442600",
        bg_hover="#FFA500",
        text_main="#FFA500",
        text_heading="#FFA500",
        text_hover="#331900",
        border_main="#FFA500",
        bg_action="#FF4500",
        text_on_action="#000000",
        bg_action_hover="#FF6347",
        bg_error="#DC143C",
        text_on_error="#ffffff",
        bg_success="#32CD32",
        text_on_success="#000000",
        border_alt="#663300",
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 20px rgba(255, 165, 0, 0.9), 0 0 60px rgba(255, 165, 0, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(255, 165, 0, 0.9)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "slate_and_copper": ThemeConfig(
        bg_primary="#2f3e46",
        bg_surface="#2f3e46",
        bg_alt="#354f52",
        bg_hover="#b87333",
        text_main="#cad2c5",
        text_heading="#cad2c5",
        text_hover="#2f3e46",
        border_main="#b87333",
        bg_action="#b87333",
        text_on_action="#ffffff",
        bg_action_hover="#cd7f32",
        bg_error="#e07a5f",
        text_on_error="#ffffff",
        bg_success="#81b29a",
        text_on_success="#2f3e46",
        border_alt="#52796f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 2px 8px rgba(184, 115, 51, 0.3)",
        },
    ),
    "sourdough": ThemeConfig(
        bg_primary="#f4e1d2",
        bg_surface="#8b5a2b",
        bg_alt="#f9e8dc",
        bg_hover="#d2a679",
        text_main="#5c4033",
        text_heading="#ffffff",
        text_hover="#5c4033",
        border_main="#d2a679",
        bg_action="#a0522d",
        text_on_action="#ffffff",
        bg_action_hover="#cd853f",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        border_alt="#deb887",
        font_family="Verdana, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(170deg, #f4e1d2 0%, #f9e8dc 60%, #f4e1d2 100%)",
            "--dgcv-special-text": "#5c4033",
            "--dgcv-table-shadow": "0 0 10px rgba(139, 90, 43, 0.2)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.3)",
        },
    ),
    "starry_night": ThemeConfig(
        bg_primary="#2a2a72",
        bg_surface="#344e86",
        bg_alt="#009ffd",
        bg_hover="#f7d84b",
        text_main="#ffffff",
        text_heading="#f7d84b",
        text_hover="#2a2a72",
        border_main="#f7d84b",
        bg_action="#009ffd",
        text_on_action="#ffffff",
        bg_action_hover="#33b5fe",
        bg_error="#ff4d4d",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#2a2a72",
        border_alt="#007acc",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(to bottom, #2a2a72, #009ffd)",
            "--dgcv-table-shadow": "0 0 20px rgba(247, 216, 75, 0.8), 0 0 60px rgba(0, 159, 253, 0.3)",
            "--dgcv-text-shadow": "0 0 6px rgba(247, 216, 75, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "sunset_gradient": ThemeConfig(
        bg_primary="#1a0a0a",
        bg_surface="#2d1a0d",
        bg_alt="#221010",
        bg_hover="#ff9a3c",
        text_main="#fde8d8",
        text_heading="#fde8d8",
        text_hover="#1a0a0a",
        text_alt="#f5d0b8",
        border_main="#c45e2a",
        border_alt="#3a1a10",
        bg_action="#c2185b",
        text_on_action="#ffffff",
        bg_action_hover="#e91e63",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#000000",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(160deg, #ff6b35 0%, #f7431c 20%, #c2185b 50%, #7b1fa2 80%, #4a148c 100%)",
            "--dgcv-table-shadow": "0 4px 24px rgba(194, 24, 91, 0.5), 0 0 60px rgba(255, 107, 53, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.25s ease",
            "--dgcv-text-shadow": "0 1px 4px rgba(0,0,0,0.5)",
        },
    ),
    "teals": ThemeConfig(
        bg_primary="#EDF6F9",
        bg_surface="#005F73",
        bg_alt="#CAF0F8",
        bg_hover="#90E0EF",
        text_main="#023047",
        text_heading="#ffffff",
        text_hover="#023047",
        border_main="#00A8E8",
        bg_action="#0A9396",
        text_on_action="#ffffff",
        bg_action_hover="#94D2BD",
        bg_error="#E63946",
        text_on_error="#ffffff",
        bg_success="#52B788",
        text_on_success="#ffffff",
        border_alt="#ADE8F4",
        font_family="Helvetica, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "3px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 168, 232, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "too_much": ThemeConfig(
        bg_primary="#000000",
        bg_surface="#1a1a1a",
        bg_alt="#333333",
        bg_hover="#FF4500",
        text_main="#FFFFFF",
        text_heading="#FF4500",
        text_hover="#FFFFFF",
        border_main="#FF4500",
        bg_action="#FF4500",
        text_on_action="#FFFFFF",
        bg_action_hover="#FF6347",
        bg_error="#FF0000",
        text_on_error="#FFFFFF",
        bg_success="#39FF14",
        text_on_success="#000000",
        border_alt="#555555",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "3px",
            "--dgcv-border-image": "linear-gradient(90deg, #FF4500, #FF0000, #FF6347, #FF4500) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 20px rgba(255, 69, 0, 0.9), 0 0 60px rgba(255, 0, 0, 0.4)",
            "--dgcv-hover-transform": "scale(1.002) rotate(0.3deg)",
            "--dgcv-text-shadow": "0 0 8px rgba(255, 69, 0, 0.8)",
            "--dgcv-hover-transition": "all 0.15s ease",
        },
    ),
    "turtle_shell": ThemeConfig(
        bg_primary="#f5f5f5",
        bg_surface="#556b2f",
        bg_alt="#e0e0e0",
        bg_hover="#6b8e23",
        text_main="#2f4f4f",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#556b2f",
        bg_action="#8fbc8f",
        text_on_action="#000000",
        bg_action_hover="#9acd32",
        bg_error="#cd5c5c",
        text_on_error="#ffffff",
        bg_success="#2e8b57",
        text_on_success="#ffffff",
        border_alt="#d3d3d3",
        font_family="Tahoma, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 8px rgba(85, 107, 47, 0.3)",
            "--dgcv-border-image": "linear-gradient(135deg, #556b2f, #9acd32, #556b2f) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "underwater": ThemeConfig(
        bg_primary="#87ceeb",
        bg_surface="#00ced1",
        bg_alt="#afeeee",
        bg_hover="#1e90ff",
        text_main="#006994",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#4682b4",
        bg_action="#0000cd",
        text_on_action="#ffffff",
        bg_action_hover="#4169e1",
        bg_error="#cd5c5c",
        text_on_error="#ffffff",
        bg_success="#3cb371",
        text_on_success="#ffffff",
        border_alt="#add8e6",
        font_family="Trebuchet MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #87ceeb 0%, #afeeee 50%, #87ceeb 100%)",
            "--dgcv-special-text": "#006994",
            "--dgcv-table-shadow": "0 4px 16px rgba(30, 144, 255, 0.5), 0 0 40px rgba(0, 206, 209, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 2px rgba(0, 105, 148, 0.3)",
        },
    ),
    "van_gogh": ThemeConfig(
        bg_primary="#f7f2e7",
        bg_surface="#1c6ea4",
        bg_alt="#fff7d1",
        bg_hover="#ffd700",
        text_main="#1c6ea4",
        text_heading="#ffffff",
        text_hover="#1c6ea4",
        border_main="#1c6ea4",
        bg_action="#e6b800",
        text_on_action="#1c6ea4",
        bg_action_hover="#cca300",
        bg_error="#cc3300",
        text_on_error="#ffffff",
        bg_success="#339966",
        text_on_success="#ffffff",
        border_alt="#104e76",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg, #f7f2e7 0%, #fff7d1 50%, #f7f2e7 100%)",
            "--dgcv-special-text": "#1c6ea4",
            "--dgcv-table-shadow": "0 4px 12px rgba(28, 110, 164, 0.25)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.5)",
        },
    ),
    "warm_orange_purple": ThemeConfig(
        bg_primary="#FFDAB9",
        bg_surface="#4B0082",
        bg_alt="#FFE4B5",
        bg_hover="#FF7F50",
        text_main="#4B0082",
        text_heading="#ffffff",
        text_hover="#4B0082",
        border_main="#800080",
        bg_action="#FF8C00",
        text_on_action="#ffffff",
        bg_action_hover="#FFA500",
        bg_error="#B22222",
        text_on_error="#ffffff",
        bg_success="#228B22",
        text_on_success="#ffffff",
        border_alt="#FFDEAD",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transition": "background-color 0.5s ease",
            "--dgcv-table-shadow": "0 2px 8px rgba(75, 0, 130, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "wooden_borders": ThemeConfig(
        bg_primary="transparent",
        bg_surface="transparent",
        bg_alt="transparent",
        bg_hover="transparent",
        text_main="inherit",
        text_heading="inherit",
        text_hover="inherit",
        border_main="#8b4513",
        bg_action="#a0522d",
        text_on_action="#ffffff",
        bg_action_hover="#cd853f",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        border_alt="#a0522d",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "10px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-border-image": "linear-gradient(135deg, #8b4513, #cd853f, #8b4513, #a0522d, #8b4513) 1",
            "--dgcv-border-radius": "0",
        },
    ),
}


def get_dgcv_themes(
    show_themes: Union[bool, str, List[str]] = False,
) -> Union[List[str], str]:
    theme_names = sorted(THEME_REGISTRY.keys())

    if not show_themes:
        return theme_names

    display_names = theme_names
    if isinstance(show_themes, str):
        display_names = [show_themes] if show_themes in THEME_REGISTRY else []
    elif isinstance(show_themes, list):
        display_names = [name for name in show_themes if name in THEME_REGISTRY]

    ui_bg = "#1a1a1a"
    ui_text = "#eceff4"
    ui_border = "#333333"

    html_output = [
        f"""
<div class='dgcv-gallery-container' style='background-color: {ui_bg}; color: {ui_text}; padding: 30px; font-family: system-ui, sans-serif;'>
    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {ui_border}; margin-bottom: 30px; padding-bottom: 10px;'>
        <h1 style='margin:0;'><code>dgcv</code> Theme Registry</h1>
        <span style='opacity: 0.6;'>{len(display_names)} Themes Loaded</span>
    </div>

    <style>
        .dgcv-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
            gap: 25px; 
        }}
        .theme-card {{ 
            background: #252525; 
            border: 1px solid {ui_border}; 
            border-radius: 10px; 
            padding: 15px; 
            display: flex; 
            flex-direction: column;
            transition: border-color 0.3s ease;
        }}
        .theme-card:hover {{ border-color: #555; }}
        .card-header {{ 
            margin-bottom: 12px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }}
        .swatch-grid {{ 
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            gap: 8px; 
        }}
        .swatch {{ 
            height: 40px; 
            display: flex; 
            flex-direction: column;
            align-items: center; 
            justify-content: center; 
            font-size: 10px; 
            font-weight: bold; 
            border-radius: 4px;
            text-transform: uppercase;
            overflow: hidden;
        }}
        .swatch-label {{ font-size: 9px; opacity: 0.7; margin-bottom: 2px; text-transform: none; }}
        .hover-zone {{
            grid-column: span 4;
            height: 50px;
            margin-top: 8px;
            cursor: help;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            font-size: 11px;
            letter-spacing: 1px;
            font-weight: 900;
        }}
    </style>
    <div class="dgcv-grid">
"""
    ]

    for name in display_names:
        style_id = f"card-{name.replace('_', '-')}"

        raw_css, theme_data = get_style(name, return_theme_data=True)
        font = theme_data.font_family.replace("inherit", "system")
        border_radius = int(
            theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace(
                "px", ""
            )
        )
        hover_zone_text = (
            "var(--dgcv-special-text,var(--dgcv-text-heading))"
            if "--dgcv-special-background" in raw_css
            else "var(--dgcv-text-main)"
        )
        scoped_css = raw_css.replace(":root", f".{style_id}")

        html_output.append(f"""
<style>
    {scoped_css}

    .{style_id} .hover-zone {{
        background: var(--dgcv-special-background, var(--dgcv-bg-primary));
        color: {hover_zone_text};
        box-shadow: var(--dgcv-table-shadow, none);
        border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
        border-image: var(--dgcv-border-image, none);
        border-radius: {border_radius};
        transition: var(--dgcv-hover-transition, all 0.2s ease);
        text-shadow: var(--dgcv-text-shadow, none);
    }}
    .{style_id} .hover-zone:hover {{
        background: var(--dgcv-bg-hover) !important;
        color: var(--dgcv-text-hover) !important;
        transform: var(--dgcv-hover-transform, none);
        box-shadow: var(--dgcv-table-shadow, none);
    }}

    .{style_id} .swatch-head {{
        background: var(--dgcv-bg-surface); 
        color: var(--dgcv-text-heading); 
        text-shadow: var(--dgcv-text-shadow, none);
    }}
    .{style_id} .swatch-main {{
        background: var(--dgcv-bg-primary); 
        color: var(--dgcv-text-main); 
        border: 1px solid var(--dgcv-border-main);
    }}
    .{style_id} .swatch-alt {{
        background: var(--dgcv-bg-alt); 
        color: var(--dgcv-text-alt);
    }}
    .{style_id} .swatch-hov {{
        background: var(--dgcv-bg-hover); 
        color: var(--dgcv-text-hover);
    }}
    .{style_id} .swatch-act {{
        background: var(--dgcv-bg-action); 
        color: var(--dgcv-text-on-action);
    }}
    .{style_id} .swatch-suc {{
        background: var(--dgcv-bg-success); 
        color: var(--dgcv-text-on-success);
    }}
    .{style_id} .swatch-err {{
        background: var(--dgcv-bg-error); 
        color: var(--dgcv-text-on-error);
    }}
    .{style_id} .swatch-brd {{
        background: var(--dgcv-bg-primary); 
        color: var(--dgcv-text-main);
        border: 2px solid var(--dgcv-border-alt);
    }}
</style>

<div class="theme-card {style_id}" style="font-family: var(--dgcv-font-family);">
    <div class="card-header">
        <strong style="color: #fff;">{name}</strong>
        <span style="font-family: var(--dgcv-font-family); font-size: 9px; opacity: 0.5;"><span style="font-size: 9px; opacity: 0.5;">{"fonts: " if "," in font else "font: "}</span>{font}</span>
    </div>

    <div class="swatch-grid">
        <div class="swatch swatch-head">HEAD</div>
        <div class="swatch swatch-main">MAIN</div>
        <div class="swatch swatch-alt">ALT</div>
        <div class="swatch swatch-hov">HOV</div>
        <div class="swatch swatch-act">ACT</div>
        <div class="swatch swatch-suc">SUC</div>
        <div class="swatch swatch-err">ERR</div>
        <div class="swatch swatch-brd">BRD-A</div>
        <div class="hover-zone">HOVER TEST</div>
    </div>
</div>
""")

    html_output.append("</div></div>")
    from ._config import _try_wrap_html

    return _try_wrap_html("".join(html_output))


def _set_dgcv_default_theme(theme: str):
    global dgcv_display_theme
    dgcv_display_theme = theme


def get_legible_hex(h: float, L: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, L, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def get_random_theme(vibrancy: float = 0.2) -> ThemeConfig:
    is_dark_mode = random.choice([True, False])
    base_hue = random.random()
    sat_base = max(0.1, min(1.0, vibrancy))
    sat_accent = max(0.2, min(1.0, vibrancy + 0.4))
    if is_dark_mode:
        bg_l, surface_l, alt_l = 0.08, 0.14, 0.18
        text_l = 0.95
    else:
        bg_l, surface_l, alt_l = 0.98, 0.94, 0.90
        text_l = 0.05

    bg_primary = get_legible_hex(base_hue, bg_l, sat_base * 0.4)
    bg_surface = get_legible_hex(base_hue, surface_l, sat_base * 0.5)
    bg_alt = get_legible_hex(base_hue, alt_l, sat_base * 0.5)
    accent_hue = (base_hue + random.uniform(0.3, 0.7)) % 1.0
    bg_hover = get_legible_hex(accent_hue, 0.5, sat_accent)
    text_heading = get_legible_hex(accent_hue, 0.6 if is_dark_mode else 0.3, sat_accent)

    custom_css_vars = {
        "--dgcv-border-width": "1px",
        "--dgcv-hover-transform": "scale(1.02) translateY(-2px)",
        "--dgcv-hover-transition": "all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
    }

    rarity = random.random()
    if rarity < 0.05:
        bg_primary = "linear-gradient(to bottom, #2a2a72, #009ffd)"
        custom_css_vars["--dgcv-table-shadow"] = "0 0 15px rgba(247, 216, 75, 0.9)"
        custom_css_vars["--dgcv-special-background"] = bg_primary
        text_main = "#ffffff"
        border_main = "#f7d84b"
    elif rarity < 0.10:
        bg_primary = get_legible_hex(base_hue, bg_l, sat_base * 0.4)
        custom_css_vars["--dgcv-special-background"] = (
            "radial-gradient(circle, #1a1a2e, #0f3460)"
        )
        custom_css_vars["--dgcv-table-shadow"] = "0 0 20px rgba(34, 166, 179, 0.8)"
        custom_css_vars["--dgcv-border-width"] = "2px"
        border_main = "#22a6b3"
        text_main = get_legible_hex(base_hue, text_l, 0.1)
    elif rarity < 0.15:
        bg_primary = get_legible_hex(base_hue, bg_l, sat_base * 0.4)
        custom_css_vars["--dgcv-special-background"] = (
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        )
        custom_css_vars["--dgcv-table-shadow"] = "0 10px 30px rgba(0,0,0,0.5)"
        custom_css_vars["--dgcv-hover-transform"] = "rotate(1deg) scale(1.01)"
        text_heading = "#a1ffce"
        text_main = get_legible_hex(base_hue, text_l, 0.1)
        border_main = get_legible_hex(base_hue, 0.45, sat_base)
    else:
        text_main = get_legible_hex(base_hue, text_l, 0.1)
        border_main = get_legible_hex(base_hue, 0.45, sat_base)

    out = ThemeConfig(
        bg_primary=bg_primary,
        bg_surface=bg_surface,
        bg_alt=bg_alt,
        bg_hover=bg_hover,
        text_main=text_main,
        text_heading=text_heading,
        text_hover="#ffffff" if is_dark_mode else "#000000",
        text_alt=get_legible_hex(base_hue, text_l, 0.2),
        border_main=border_main,
        border_alt=get_legible_hex(base_hue, 0.55, sat_base * 0.5),
        font_family=random.choice(
            [
                "'Courier New', monospace",
                "Georgia, serif",
                "Impact, sans-serif",
                "Verdana",
            ]
        ),
        bg_action=bg_hover,
        text_on_action="#ffffff",
        bg_action_hover=get_legible_hex(accent_hue, 0.4, sat_accent),
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00ff87",
        text_on_success="#000000",
        custom_css_vars=custom_css_vars,
    )
    if get_dgcv_settings_registry().get("DEBUG", False):
        print(out.registry_format("random"))
    return out


def get_style(theme_name: str, *args, return_theme_data: bool = False, **kwargs) -> str:
    """
    Returns a CSS ``:root { }`` block of dgcv theme variables for the given theme name.

    Looks up ``theme_name`` in the THEME_REGISTRY and emits all theme fields and
    custom variables as scoped CSS custom properties via ``ThemeConfig.to_css_string``.

    Parameters:
    -----------
    theme_name: str
        Name of the theme to look up. A light autocomplete (via the python difflib)
        will replace malformed str names to nearby themes if they exist.
        If ``theme_name`` is ``"random"``, a theme is selected at random from the
        registry. If ``theme_name`` or a close match is not found in the registry
        and is not ``"random"``, falls back silently to dgcv settings defualts.

    Returns:
    --------
        str — a ``:root { }`` CSS block with all theme variables indented and
        ready for injection into a ``<style>`` tag.
    """
    default_dark, default_light = "matte_slate_soft", "paper_graphite"  # hard coded
    aliases = {
        "chalkboard": "chalkboard_green",
        "gruv": "gruvbox_dark",
        "dark": default_dark,
        "light": default_light,
    }
    theme_name = theme_name.lower()
    if difflib.get_close_matches(theme_name, ["shuffle"], n=1, cutoff=0.6):
        theme_name = random.choice(get_dgcv_themes())
    theme_name = aliases.get(theme_name, theme_name)
    if theme_name not in THEME_REGISTRY and theme_name != "random":
        close = difflib.get_close_matches(
            theme_name, THEME_REGISTRY.keys(), n=1, cutoff=0.6
        )
        theme_name = (
            close[0]
            if close
            else aliases.get(
                get_dgcv_settings_registry().get("theme", default_dark), default_dark
            )
        )
    theme_data = (
        get_random_theme() if theme_name == "random" else THEME_REGISTRY[theme_name]
    )
    if return_theme_data:
        return f":root {{\n    {theme_data.to_css_string()}\n}}", theme_data
    return f":root {{\n    {theme_data.to_css_string()}\n}}"
