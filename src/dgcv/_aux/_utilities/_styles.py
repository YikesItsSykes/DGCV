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
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

__all__ = ["get_dgcv_themes", "get_style", "ThemeConfig"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------

dgcv_display_theme = "dark_modern"
dgcv_custom_variables = [
    "--dgcv-border-width",
    "--dgcv-border-radius",
    "--dgcv-hover-transform",
    "--dgcv-table-shadow",
    "--dgcv-table-background",
    "--dgcv-text-shadow",
    "--dgcv-hover-transition",
    "--dgcv-border-image",
    "--dgcv-hover-font-weight",
]

# safe pairs
# --dgcv-bg-surface/--dgcv-text-heading
# --bg-primary/--dccv-text-main
# --dgcv-bg-alt/--dgcv-text-alt


@dataclass
class ThemeConfig:
    bg_primary: str
    bg_surface: str
    bg_alt: str
    bg_hover: str
    text_main: str
    text_heading: str
    text_hover: str
    border_main: str
    font_family: str = "inherit"
    text_alt: Optional[str] = None
    custom_css_vars: Dict[str, str] = field(default_factory=dict)


THEME_REGISTRY: Dict[str, ThemeConfig] = {
    "appalachian": ThemeConfig(
        bg_primary="#F0F8FF",
        bg_surface="#2E8B57",
        bg_alt="#B0E0E6",
        bg_hover="#5F9EA0",
        text_main="#2F4F4F",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#2E8B57",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.01)",
            "--dgcv-table-shadow": "0 0 8px rgba(46, 139, 87, 0.3)",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.005)",
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
        font_family="'Inter', sans-serif",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 8px rgba(205, 127, 50, 0.5)",
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
        font_family="Verdana, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.002)",
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
        font_family="Arial, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.002)",
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
        font_family="monospace",
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
        font_family="system-ui, -apple-system, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.002)",
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
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 12px rgba(106, 90, 205, 0.7)",
            "--dgcv-hover-transform": "scale(1.03)",
        },
    ),
    "sakura": ThemeConfig(
        bg_primary="#fff0f5",
        bg_surface="#d6336c",
        bg_alt="#ffeef8",
        bg_hover="#f7cac9",
        text_main="#8a2a2b",
        text_heading="#ffffff",
        text_hover="#8a2a2b",
        border_main="#f7cac9",
        font_family="Palatino, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.01)",
            "--dgcv-table-shadow": "0 0 8px rgba(214, 51, 108, 0.2)",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.002)",
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
        font_family="Verdana, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.01)",
            "--dgcv-table-shadow": "0 0 8px rgba(0, 0, 0, 0.1)",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-table-background": "linear-gradient(to bottom, #2a2a72, #009ffd)",
            "--dgcv-table-shadow": "0 0 10px rgba(247, 216, 75, 0.8)",
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
        font_family="Helvetica, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "3px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 168, 232, 0.5)",
            "--dgcv-hover-transform": "scale(1.02)",
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
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 12px rgba(255, 69, 0, 0.7)",
            "--dgcv-hover-transform": "scale(1.03)",
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
        font_family="Tahoma, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.01)",
            "--dgcv-table-shadow": "0 0 8px rgba(0, 0, 0, 0.1)",
        },
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.01)",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.05)",
        },
    ),
    "gruvbox_colorful": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#458588",
        bg_alt="#ebdbb2",
        bg_hover="#fabd2f",
        text_main="#282828",
        text_heading="#ffffff",
        text_hover="#282828",
        border_main="#8ec07c",
        font_family="Comic Sans MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.02)",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.1)",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-radius": "8px",
            "--dgcv-table-shadow": "0 0 10px rgba(62, 39, 35, 0.2)",
            "--dgcv-text-shadow": "none",
        },
    ),
    "dark_purple": ThemeConfig(
        bg_primary="#1a1a2e",
        bg_surface="#2c2c54",
        bg_alt="#0f3460",
        bg_hover="#22a6b3",
        text_main="#e0e0e0",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#ffffff",
        font_family="Trebuchet MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-background": "radial-gradient(circle, #1a1a2e, #0f3460)",
            "--dgcv-table-shadow": "0 0 10px rgba(34, 166, 179, 0.8)",
        },
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
        font_family="inherit",
        custom_css_vars={},
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
        font_family="Dancing Script, cursive",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 10px rgba(92, 67, 39, 0.5)",
            "--dgcv-hover-transform": "scale(1.01)",
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
        font_family="Cormorant, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 15px rgba(0, 121, 107, 0.5)",
            "--dgcv-hover-transform": "scale(1.01)",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 10px rgba(191, 54, 12, 0.3)",
            "--dgcv-hover-transform": "scale(1.01)",
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
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 15px rgba(127, 219, 255, 0.9)",
            "--dgcv-hover-transform": "scale(1.02)",
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
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 15px rgba(57, 255, 20, 0.9)",
            "--dgcv-hover-transform": "scale(1.02)",
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
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 15px rgba(255, 0, 255, 0.9)",
            "--dgcv-hover-transform": "scale(1.02)",
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
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 15px rgba(255, 165, 0, 0.9)",
            "--dgcv-hover-transform": "scale(1.02)",
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
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 4px 10px rgba(75, 0, 130, 0.8)",
            "--dgcv-hover-transform": "scale(1.01)",
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
        font_family="Trebuchet MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 4px 10px rgba(30, 144, 255, 0.6)",
            "--dgcv-hover-transform": "scale(1.02)",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transition": "background-color 0.5s ease",
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
        font_family="Permanent Marker, cursive",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 15px rgba(30, 144, 255, 0.9)",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "10px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
            "--dgcv-border-radius": "8px",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
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
        font_family="Courier New, monospace",
        custom_css_vars={
            "--dgcv-border-width": "5px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
        },
    ),
    "1980s_neon": ThemeConfig(
        bg_primary="#3b3b58",
        bg_surface="#9400d3",
        bg_alt="#282a36",
        bg_hover="#00ff7f",
        text_main="#ffffff",
        text_heading="#00ff00",
        text_hover="#000000",
        border_main="#ff1493",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(90deg, #ff1493, #9400d3) 1",
            "--dgcv-border-radius": "12px",
            "--dgcv-text-shadow": "0px 0px 6px #00ff00",
            "--dgcv-table-shadow": "0 4px 8px rgba(0, 255, 127, 0.7)",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #c0c0c0, #ff7f50, #00ffff) 1",
            "--dgcv-border-radius": "12px",
            "--dgcv-text-shadow": "0px 0px 5px #ff7f50",
            "--dgcv-table-shadow": "0 4px 8px rgba(255, 127, 80, 0.8)",
        },
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
        font_family="Roboto Mono, monospace",
        custom_css_vars={
            "--dgcv-border-width": "2px",
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
        font_family="Roboto, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
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
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0px 2px 5px rgba(0, 0, 0, 0.1)",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #006400, #228b22) 1",
            "--dgcv-border-radius": "15px",
            "--dgcv-text-shadow": "0px 0px 4px #006400",
            "--dgcv-table-shadow": "0 4px 8px rgba(0, 100, 0, 0.7)",
            "--dgcv-hover-font-weight": "bold",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(90deg, #00bfff, #1e90ff, #4682b4) 1",
            "--dgcv-border-radius": "12px",
            "--dgcv-text-shadow": "0px 0px 4px #00bfff",
            "--dgcv-table-shadow": "0 4px 8px rgba(30, 144, 255, 0.6)",
            "--dgcv-hover-font-weight": "bold",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #d2b48c, #ffd700) 1",
            "--dgcv-border-radius": "10px",
            "--dgcv-text-shadow": "none",
            "--dgcv-table-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dark_moody": ThemeConfig(
        bg_primary="#3b1a4f",
        bg_surface="#2c003e",
        bg_alt="#503a66",
        bg_hover="#4a3a57",
        text_main="#d3d3d3",
        text_heading="#e6e6e6",
        text_hover="#d3d3d3",
        border_main="#8c0099",
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
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "Van_Gogh": ThemeConfig(
        bg_primary="#f7f2e7",
        bg_surface="#1c6ea4",
        bg_alt="#fff7d1",
        bg_hover="#ffd700",
        text_main="#1c6ea4",
        text_heading="#ffffff",
        text_hover="#1c6ea4",
        border_main="#1c6ea4",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "Monet": ThemeConfig(
        bg_primary="#f7f9e4",
        bg_surface="#2a5d67",
        bg_alt="#c6f3d8",
        bg_hover="#d0ece7",
        text_main="#2a5d67",
        text_heading="#ffffff",
        text_hover="#2a5d67",
        border_main="#2a5d67",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "Rembrandt": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#523d2e",
        bg_alt="#f7e6d5",
        bg_hover="#d4b996",
        text_main="#523d2e",
        text_heading="#ffffff",
        text_hover="#523d2e",
        border_main="#6c4f3d",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "Picasso_blue": ThemeConfig(
        bg_primary="#eaf2f8",
        bg_surface="#154360",
        bg_alt="#d6eaf8",
        bg_hover="#aed6f1",
        text_main="#154360",
        text_heading="#ffffff",
        text_hover="#154360",
        border_main="#154360",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "Matisse": ThemeConfig(
        bg_primary="#e6ffe6",
        bg_surface="#ffcc00",
        bg_alt="#ffd6cc",
        bg_hover="#ffe680",
        text_main="#004d00",
        text_heading="#004d00",
        text_hover="#004d00",
        border_main="#004d00",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
}


def get_dgcv_themes(show_themes=False) -> str:
    theme_names = sorted(THEME_REGISTRY.keys())

    if not show_themes:
        return str(theme_names)

    ui_bg = "#1a1a1a"
    ui_text = "#eceff4"
    ui_border = "#333333"

    html_output = [
        f"""
<div class='dgcv-gallery-container' style='background-color: {ui_bg}; color: {ui_text}; padding: 30px; font-family: system-ui, sans-serif;'>
    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {ui_border}; margin-bottom: 30px; padding-bottom: 10px;'>
        <h1 style='margin:0;'><code>dgcv</code> Theme Registry</h1>
        <span style='opacity: 0.6;'>{len(theme_names)} Themes Loaded</span>
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
            grid-template-columns: 1fr 1fr; 
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
            grid-column: span 2;
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

    for name in theme_names:
        theme = THEME_REGISTRY[name]

        h_trans = theme.custom_css_vars.get("--dgcv-hover-transform", "none")
        h_shad = theme.custom_css_vars.get("--dgcv-table-shadow", "none")
        h_speed = theme.custom_css_vars.get("--dgcv-hover-transition", "all 0.2s ease")
        t_shad = theme.custom_css_vars.get("--dgcv-text-shadow", "none")
        b_width = theme.custom_css_vars.get("--dgcv-border-width", "1px")

        alt_text_color = getattr(theme, "text_alt", theme.text_main) or theme.text_main

        table_bg = theme.custom_css_vars.get(
            "--dgcv-table-background", theme.bg_primary
        )
        style_id = f"card-{name.replace('_', '-')}"

        html_output.append(f"""
<style>
    .{style_id} .hover-zone {{
        background: {table_bg};
        color: {theme.text_main};
        border: {b_width} solid {theme.border_main};
        transition: {h_speed};
        text-shadow: {t_shad};
    }}
    .{style_id} .hover-zone:hover {{
        background: {theme.bg_hover} !important;
        color: {theme.text_hover} !important;
        transform: {h_trans};
        box-shadow: {h_shad};
    }}
</style>

<div class="theme-card {style_id}">
    <div class="card-header">
        <strong style="color: #fff;">{name}</strong>
        <span style="font-size: 9px; opacity: 0.5;">{theme.font_family.split(",")[0]}</span>
    </div>

    <div class="swatch-grid" style="font-family: {theme.font_family};">
        <div class="swatch" style="background: {theme.bg_surface}; color: {theme.text_heading}; text-shadow: {t_shad};">
            HEAD
        </div>
        <div class="swatch" style="background: {theme.bg_primary}; color: {theme.text_main}; border: 1px solid {theme.border_main}33;">
            MAIN
        </div>
        <div class="swatch" style="background: {theme.bg_alt}; color: {alt_text_color};">
            ALT
        </div>
        <div class="swatch" style="background: {theme.bg_hover}; color: {theme.text_hover};">
            HOV
        </div>
        <div class="hover-zone">
            HOVER TEST
        </div>
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


def get_random_theme(vibrancy: float = 0.1) -> ThemeConfig:
    is_dark_mode = random.choice([True, False])
    base_hue = random.random()

    sat_base = max(0.0, min(1.0, vibrancy))
    sat_accent = max(0.0, min(1.0, vibrancy + 0.2))

    if is_dark_mode:
        bg_l, text_l = random.uniform(0.12, 0.20), random.uniform(0.80, 0.90)
        alt_l = bg_l + 0.06
    else:
        bg_l, text_l = random.uniform(0.90, 0.96), random.uniform(0.10, 0.25)
        alt_l = bg_l - 0.06

    bg_hex = get_legible_hex(base_hue, bg_l, sat_base * 0.5)
    text_hex = get_legible_hex(base_hue, text_l, sat_base * 0.2)
    accent_hex = get_legible_hex(base_hue, 0.5, sat_accent)
    alt_bg_hex = get_legible_hex(base_hue, alt_l, sat_base * 0.5)

    fonts = [
        "Arial, sans-serif",
        "Georgia, serif",
        "Verdana, sans-serif",
        "Tahoma, sans-serif",
    ]

    return ThemeConfig(
        bg_primary=bg_hex,
        bg_surface=bg_hex,
        bg_alt=alt_bg_hex,
        bg_hover=accent_hex,
        text_main=text_hex,
        text_heading=accent_hex,
        text_hover=bg_hex,
        border_main=accent_hex,
        font_family=random.choice(fonts),
    )


def _build_legacy_format(theme: ThemeConfig) -> List[Dict[str, Any]]:
    base_styles = {
        "table": [("border", f"1px solid {theme.border_main}")],
        "header": [
            ("background-color", theme.bg_primary),
            ("color", theme.text_heading),
        ],
        "col_heading": [
            ("background-color", theme.bg_primary),
            ("color", theme.text_heading),
        ],
        "row_heading": [
            ("background-color", theme.bg_surface),
            ("color", theme.text_heading),
        ],
        "row": [("background-color", theme.bg_surface), ("color", theme.text_main)],
        "alt_row": [("background-color", theme.bg_alt), ("color", theme.text_main)],
        "hover": [("background-color", theme.bg_hover), ("color", theme.text_hover)],
    }

    if theme.font_family and theme.font_family != "inherit":
        for element in ["header", "col_heading", "row_heading", "row", "alt_row"]:
            base_styles[element].append(("font-family", theme.font_family))

    selector_mapping = {
        "table": "table",
        "header": "thead th",
        "col_heading": "th.col_heading.level0",
        "row_heading": "th.row_heading",
        "row": "tbody tr:nth-child(odd)",
        "alt_row": "tbody tr:nth-child(even)",
        "hover": "tbody tr:hover",
        "inner": "th:not(:last-child), td:not(:last-child)",
    }

    formatted_styles = []
    for key, props in base_styles.items():
        if not props:
            continue
        selector = selector_mapping.get(key, key)
        formatted_styles.append({"selector": selector, "props": props})

    return formatted_styles


def _build_modern_format(theme: ThemeConfig) -> str:
    css_lines = [
        f"--dgcv-bg-primary: {theme.bg_primary};",
        f"--dgcv-bg-surface: {theme.bg_surface};",
        f"--dgcv-bg-alt: {theme.bg_alt};",
        f"--dgcv-bg-hover: {theme.bg_hover};",
        f"--dgcv-text-main: {theme.text_main};",
        f"--dgcv-text-heading: {theme.text_heading};",
        f"--dgcv-text-hover: {theme.text_hover};",
        f"--dgcv-border-main: {theme.border_main};",
        f"--dgcv-font-family: {theme.font_family};",
        f"--dgcv-text-alt: {theme.text_alt if getattr(theme, 'text_alt', None) else theme.text_main};",
    ]

    for var_name, var_value in theme.custom_css_vars.items():
        css_lines.append(f"{var_name}: {var_value};")

    joined_vars = "\n    ".join(css_lines)

    return f":root {{\n    {joined_vars}\n}}"


def get_style(theme_name: str, legacy: bool = True) -> Union[List[Dict[str, Any]], str]:
    if theme_name == "chalkboard":
        theme_name = "chalkboard_green"
    if theme_name == "gruv":
        theme_name = "gruvbox_dark"
    if theme_name == "dark":
        theme_name = "dark_modern"

    if theme_name not in THEME_REGISTRY and theme_name != "random":
        theme_name = "dark_modern"

    theme_data = (
        get_random_theme() if theme_name == "random" else THEME_REGISTRY[theme_name]
    )

    if legacy:
        return _build_legacy_format(theme_data)

    return _build_modern_format(theme_data)


def _build_legacy_format(theme: ThemeConfig) -> List[Dict[str, Any]]:
    b_width = theme.custom_css_vars.get("--dgcv-border-width", "1px")
    border_spec = f"{b_width} solid {theme.border_main}"

    table_props = [("border", border_spec)]
    if "--dgcv-table-background" in theme.custom_css_vars:
        table_props.append(
            ("background", theme.custom_css_vars["--dgcv-table-background"])
        )
    else:
        table_props.append(("background-color", theme.bg_primary))

    if "--dgcv-table-shadow" in theme.custom_css_vars:
        table_props.append(("box-shadow", theme.custom_css_vars["--dgcv-table-shadow"]))

    return [
        {"selector": "table", "props": table_props},
        {
            "selector": "thead th",
            "props": [
                ("background-color", theme.bg_primary),
                ("color", theme.text_heading),
                ("font-family", theme.font_family),
            ],
        },
        {
            "selector": "th.col_heading.level0",
            "props": [
                ("background-color", theme.bg_primary),
                ("color", theme.text_heading),
                ("font-family", theme.font_family),
            ],
        },
        {
            "selector": "th.row_heading",
            "props": [
                ("background-color", theme.bg_surface),
                ("color", theme.text_heading),
                ("font-family", theme.font_family),
            ],
        },
        {
            "selector": "tbody tr:nth-child(odd)",
            "props": [
                ("background-color", theme.bg_surface),
                ("color", theme.text_main),
                ("font-family", theme.font_family),
            ],
        },
        {
            "selector": "tbody tr:nth-child(even)",
            "props": [
                ("background-color", theme.bg_alt),
                ("color", theme.text_main),
                ("font-family", theme.font_family),
            ],
        },
        {
            "selector": "tbody tr:hover",
            "props": [
                ("background-color", theme.bg_hover),
                ("color", theme.text_hover),
            ],
        },
    ]
