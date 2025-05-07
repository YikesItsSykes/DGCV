import random
import warnings

# Base styles shared across all themes
base_style = {
    "header": {
        "selector": "th",
        "props": [("text-align", "center"), ("font-weight", "bold")],
    },
    "col_heading": {
        "selector": "th.col_heading.level0",
        "props": [("font-size", "22px")],
    },
    "row_heading": {
        "selector": "th.row_heading",
        "props": [("text-align", "center"), ("font-weight", "bold")],
    },
    "row": {
        "selector": "tbody tr:nth-child(odd)",
        "props": [("text-align", "center")],
    },
    "alt_row": {
        "selector": "tbody tr:nth-child(even)",
        "props": [],
    },
    "hover": {
        "selector": "tbody tr:hover",
        "props": [("transform", "scale(1.01)")],
    },
}

# Theme-specific overrides
style_guide = {
    "Van_Gogh": {
        "header": {"props": [("background-color", "#1c6ea4"), ("color", "#f7d84b")]},
        "col_heading": {
            "props": [("background-color", "#1c6ea4"), ("color", "#f7d84b")]
        },
        "row_heading": {
            "props": [("background-color", "#ffb300"), ("color", "#1c6ea4")]
        },
        "row": {"props": [("background-color", "#f7f2e7"), ("color", "#333333")]},
        "alt_row": {"props": [("background-color", "#fff7d1")]},
        "hover": {"props": [("background-color", "#ffd700")]},
        "border": {"props": [("border", "1px solid #1c6ea4")]},
    },
    "Monet": {
        "header": {"props": [("background-color", "#95c8d8"), ("color", "#e0f7fa")]},
        "col_heading": {
            "props": [("background-color", "#95c8d8"), ("color", "#e0f7fa")]
        },
        "row_heading": {
            "props": [("background-color", "#a3e0d8"), ("color", "#2a5d67")]
        },
        "row": {"props": [("background-color", "#f7f9e4"), ("color", "#3b5c5e")]},
        "alt_row": {"props": [("background-color", "#c6f3d8")]},
        "hover": {"props": [("background-color", "#d0ece7")]},
        "border": {"props": [("border", "1px solid #2a5d67")]},
    },
    "Rembrandt": {
        "header": {"props": [("background-color", "#523d2e"), ("color", "#e5c07b")]},
        "col_heading": {
            "props": [("background-color", "#523d2e"), ("color", "#e5c07b")]
        },
        "row_heading": {
            "props": [("background-color", "#6c4f3d"), ("color", "#d2b48c")]
        },
        "row": {"props": [("background-color", "#fffaf0"), ("color", "#2e2e2e")]},
        "alt_row": {"props": [("background-color", "#f7e6d5")]},
        "hover": {"props": [("background-color", "#d4b996")]},
        "border": {"props": [("border", "1px solid #6c4f3d")]},
    },
    "Picasso_blue": {
        "header": {"props": [("background-color", "#1f618d"), ("color", "#d5f4e6")]},
        "col_heading": {
            "props": [("background-color", "#1f618d"), ("color", "#d5f4e6")]
        },
        "row_heading": {
            "props": [("background-color", "#2e86c1"), ("color", "#d5f4e6")]
        },
        "row": {"props": [("background-color", "#eaf2f8"), ("color", "#154360")]},
        "alt_row": {"props": [("background-color", "#d6eaf8")]},
        "hover": {"props": [("background-color", "#aed6f1")]},
        "border": {"props": [("border", "1px solid #154360")]},
    },
    "Matisse": {
        "header": {"props": [("background-color", "#ffcc00"), ("color", "#004d00")]},
        "col_heading": {
            "props": [("background-color", "#ffcc00"), ("color", "#004d00")]
        },
        "row_heading": {
            "props": [("background-color", "#ff704d"), ("color", "#004d00")]
        },
        "row": {"props": [("background-color", "#e6ffe6"), ("color", "#004d00")]},
        "alt_row": {"props": [("background-color", "#ffd6cc")]},
        "hover": {"props": [("background-color", "#ffe680")]},
        "border": {"props": [("border", "1px solid #004d00")]},
    },
    "dark_modern": {
        "header": {"props": [("background-color", "#1c1c1c"), ("color", "#f5f5f5")]},
        "col_heading": {
            "props": [
                ("background-color", "#1c1c1c"),
                ("color", "#f5f5f5"),
                ("font-size", "18px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#333333"), ("color", "#f5f5f5")]
        },
        "row": {"props": [("background-color", "#2c2c2c"), ("color", "#f5f5f5")]},
        "alt_row": {"props": [("background-color", "#3a3a3a")]},
        "hover": {"props": [("background-color", "#484848")]},
        "border": {"props": [("border", "1px solid #444444")]},
    },
    "dark_high_contrast": {
        "header": {"props": [("background-color", "#000000"), ("color", "#ffffff")]},
        "col_heading": {
            "props": [
                ("background-color", "#000000"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#222222"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#1e1e1e"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#333333")]},
        "hover": {"props": [("background-color", "#4d4d4d")]},
        "border": {"props": [("border", "1px solid #ffffff")]},
    },
    "dark_blue": {
        "header": {"props": [("background-color", "#001f3f"), ("color", "#7fdbff")]},
        "col_heading": {
            "props": [
                ("background-color", "#001f3f"),
                ("color", "#7fdbff"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#001a33"), ("color", "#7fdbff")]
        },
        "row": {"props": [("background-color", "#002b4f"), ("color", "#7fdbff")]},
        "alt_row": {"props": [("background-color", "#004080")]},
        "hover": {"props": [("background-color", "#0059b3")]},
        "border": {"props": [("border", "1px solid #7fdbff")]},
    },
    "dark_moody": {
        "header": {"props": [("background-color", "#2c003e"), ("color", "#e6e6e6")]},
        "col_heading": {
            "props": [
                ("background-color", "#2c003e"),
                ("color", "#e6e6e6"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#1f0029"), ("color", "#e6e6e6")]
        },
        "row": {"props": [("background-color", "#3b1a4f"), ("color", "#d3d3d3")]},
        "alt_row": {"props": [("background-color", "#503a66")]},
        "hover": {"props": [("background-color", "#4a3a57")]},
        "border": {"props": [("border", "1px solid #8c0099")]},
    },
    "dark_high_contrast_bright": {
        "header": {
            "props": [("background-color", "#000000"), ("color", "#00ffff")]
        },  # Bright Aqua
        "col_heading": {
            "props": [
                ("background-color", "#000000"),
                ("color", "#7fff00"),
                ("font-size", "20px"),
            ]
        },  # Lime
        "row_heading": {
            "props": [("background-color", "#333333"), ("color", "#da70d6")]
        },  # Violet
        "row": {
            "props": [("background-color", "#1e1e1e"), ("color", "#ff69b4")]
        },  # Hot Pink
        "alt_row": {"props": [("background-color", "#4d4d4d")]},  # Dark Gray
        "hover": {"props": [("background-color", "#9370db")]},  # Medium Purple Hover
        "border": {"props": [("border", "1px solid #ffff00")]},  # Bright Yellow Border
    },
    "presentation": {
        "header": {
            "props": [
                ("background-color", "#007acc"),
                ("color", "white"),
                ("font-size", "22px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#007acc"),
                ("color", "white"),
                ("font-size", "26px"),
            ]
        },
        "row_heading": {"props": [("background-color", "#005f99"), ("color", "white")]},
        "row": {"props": [("background-color", "#ffffff"), ("color", "#000000")]},
        "alt_row": {"props": [("background-color", "#e0f4ff")]},
        "hover": {"props": [("background-color", "#cce5ff")]},
        "border": {"props": [("border", "1px solid #007acc")]},
    },
    "safari": {
        "header": {"props": [("background-color", "#8b6e4e"), ("color", "#f7d94c")]},
        "col_heading": {
            "props": [
                ("background-color", "#8b6e4e"),
                ("color", "#f7d94c"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#c1a97d"), ("color", "#4b5320")]
        },
        "row": {"props": [("background-color", "#f3e6d4"), ("color", "#4b4b4b")]},
        "alt_row": {"props": [("background-color", "#e4d7c5")]},
        "hover": {"props": [("background-color", "#d1b998")]},
        "border": {"props": [("border", "1px solid #8b6e4e")]},
    },
    "Banksy": {
        "header": {"props": [("background-color", "#2c2c2c"), ("color", "#ff0000")]},
        "col_heading": {
            "props": [
                ("background-color", "#2c2c2c"),
                ("color", "rgb(51,108,178)"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#595959"), ("color", "rgb(255, 196, 13)")]
        },
        "row": {"props": [("background-color", "#3a3a3a"), ("color", "#ff0000")]},
        "alt_row": {"props": [("background-color", "#4c4c4c")]},
        "hover": {"props": [("background-color", "rgb(255, 196, 13)")]},
        "border": {"props": [("border", "1px solid #ffffff")]},
    },
    "lunar": {
        "header": {"props": [("background-color", "#394b59"), ("color", "#b0c4de")]},
        "col_heading": {
            "props": [
                ("background-color", "#394b59"),
                ("color", "#b0c4de"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#596e79"), ("color", "#d8dee9")]
        },
        "row": {"props": [("background-color", "#dfe7ec"), ("color", "#333333")]},
        "alt_row": {"props": [("background-color", "#f5f8fa")]},
        "hover": {"props": [("background-color", "#cbd6e2")]},
        "border": {"props": [("border", "1px solid #394b59")]},
    },
    "gothic": {
        "header": {"props": [("background-color", "#2c0033"), ("color", "#a80000")]},
        "col_heading": {
            "props": [
                ("background-color", "#2c0033"),
                ("color", "#a80000"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#4d004d"), ("color", "#e6e6e6")]
        },
        "row": {"props": [("background-color", "#1c1c1c"), ("color", "#e6e6e6")]},
        "alt_row": {"props": [("background-color", "#330033")]},
        "hover": {"props": [("background-color", "#660000")]},
        "border": {"props": [("border", "1px solid #a80000")]},
    },
    "rain_forest": {
        "table": {
            "props": [
                ("border-image", "linear-gradient(45deg, #006400, #228b22) 1"),
                ("border-radius", "15px"),
                ("border-collapse", "separate"),
            ]
        },
        "header": {
            "props": [
                ("background-color", "#228b22"),
                ("color", "#ffffff"),
                ("text-shadow", "0px 0px 4px #006400"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#228b22"),
                ("color", "#ffffff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#8b4513"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#90ee90"), ("color", "#2f4f4f")]},
        "alt_row": {"props": [("background-color", "#98fb98"), ("color", "#2f4f4f")]},
        "hover": {
            "props": [
                ("background-color", "#006400"),
                ("color", "#ffffff"),
                ("font-weight", "bold"),
                ("box-shadow", "0 4px 8px rgba(0, 100, 0, 0.7)"),
            ]
        },
    },
    "ocean": {
        "table": {
            "props": [
                ("border-image", "linear-gradient(90deg, #00bfff, #1e90ff, #4682b4) 1"),
                ("border-radius", "12px"),
                ("border-collapse", "separate"),
            ]
        },
        "header": {
            "props": [
                ("background-color", "#1e90ff"),
                ("color", "#ffffff"),
                ("text-shadow", "0px 0px 4px #00bfff"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#1e90ff"),
                ("color", "#ffffff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#00bfff"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#4682b4"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#87cefa"), ("color", "#ffffff")]},
        "hover": {
            "props": [
                ("background-color", "#1e90ff"),
                ("color", "#ffffff"),
                ("font-weight", "bold"),
                ("box-shadow", "0 4px 8px rgba(30, 144, 255, 0.6)"),
            ]
        },
    },
    "dessert": {
        "table": {
            "props": [
                ("border-image", "linear-gradient(45deg, #d2b48c, #ffd700) 1"),
                ("border-radius", "10px"),
                ("border-collapse", "separate"),
            ]
        },
        "header": {
            "props": [
                ("background-color", "#ff8c00"),
                ("color", "#ffffff"),
                ("text-shadow", "0px 0px 5px #ffa500"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ff8c00"),
                ("color", "#ffffff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#d2b48c"), ("color", "#8b4513")]
        },
        "row": {"props": [("background-color", "#ffebcd"), ("color", "#8b4513")]},
        "alt_row": {"props": [("background-color", "#fffacd"), ("color", "#8b4513")]},
        "hover": {
            "props": [
                ("background-color", "#ffd700"),
                ("color", "#ffffff"),
                ("box-shadow", "0 4px 8px rgba(255, 215, 0, 0.6)"),
            ]
        },
    },
    "1980s_neon": {
        "table": {
            "props": [
                ("border-image", "linear-gradient(90deg, #ff1493, #9400d3) 1"),
                ("border-radius", "12px"),
                ("border-collapse", "separate"),
            ]
        },
        "header": {
            "props": [
                ("background-color", "#9400d3"),
                ("color", "#00ff00"),
                ("text-shadow", "0px 0px 6px #00ff00"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ff1493"),
                ("color", "#00ff00"),
                ("font-size", "24px"),
                ("text-shadow", "0px 0px 6px #00ff00"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#ff1493"), ("color", "#f0e68c")]
        },
        "row": {"props": [("background-color", "#3b3b58"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#282a36"), ("color", "#ffffff")]},
        "hover": {
            "props": [
                ("background-color", "#00ff7f"),
                ("color", "#000000"),
                ("box-shadow", "0 4px 8px rgba(0, 255, 127, 0.7)"),
            ]
        },
    },
    "back_to_the_future": {
        "table": {
            "props": [
                ("border-image", "linear-gradient(45deg, #c0c0c0, #ff7f50, #00ffff) 1"),
                ("border-radius", "12px"),
                ("border-collapse", "separate"),
            ]
        },
        "header": {
            "props": [
                ("background-color", "#003366"),
                ("color", "#f7e014"),
                ("text-shadow", "0px 0px 5px #ff7f50"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ff7f50"),
                ("color", "#00ffff"),
                ("font-size", "24px"),
                ("text-shadow", "0px 0px 6px #00ffff"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#f7e014"),
                ("color", "#003366"),
                ("text-shadow", "0px 0px 5px #00ffff"),
            ]
        },
        "row": {"props": [("background-color", "#0059b3"), ("color", "#f7e014")]},
        "alt_row": {"props": [("background-color", "#004080"), ("color", "#e0e0e0")]},
        "hover": {
            "props": [
                ("background-color", "#ff7f50"),
                ("color", "#000000"),
                ("box-shadow", "0 4px 8px rgba(255, 127, 80, 0.8)"),
            ]
        },
    },
    "blueprint": {
        "table": {
            "props": [("background-color", "#003366"), ("border-collapse", "collapse")]
        },
        "header": {
            "props": [
                ("background-color", "#003366"),
                ("color", "#ffffff"),
                ("font-family", "Roboto Mono, monospace"),
                ("font-size", "18px"),
                ("border", "2px solid #cccccc"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#003366"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
                ("font-family", "Roboto Mono, monospace"),
                ("border", "2px solid #cccccc"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#336699"),
                ("color", "#ffffff"),
                ("font-family", "Roboto Mono, monospace"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#002b4f"),
                ("color", "#ffffff"),
                ("font-family", "Roboto Mono, monospace"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#003366"),
                ("color", "#ffffff"),
                ("font-family", "Roboto Mono, monospace"),
            ]
        },
        "hover": {"props": [("background-color", "#336699"), ("color", "#ffffff")]},
    },
    "graph_paper": {
        "table": {
            "props": [("background-color", "#ffffff"), ("border-collapse", "collapse")]
        },
        "header": {
            "props": [
                ("background-color", "#ffffff"),
                ("color", "#000000"),
                ("font-family", "Roboto, sans-serif"),
                ("font-size", "18px"),
                ("border", "1px solid #cccccc"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ffffff"),
                ("color", "#000000"),
                ("font-size", "20px"),
                ("font-family", "Roboto, sans-serif"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#e6f7ff"),
                ("color", "#000000"),
                ("font-family", "Roboto, sans-serif"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#f2faff"),
                ("color", "#000000"),
                ("font-family", "Roboto, sans-serif"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#ffffff"),
                ("color", "#000000"),
                ("font-family", "Roboto, sans-serif"),
            ]
        },
        "hover": {"props": [("background-color", "#e6f7ff"), ("color", "#000000")]},
    },
    "newspaper": {
        "table": {
            "props": [
                ("background-color", "#cccccc"),
                ("border-collapse", "collapse"),
                ("box-shadow", "0px 2px 5px rgba(0, 0, 0, 0.1)"),
            ]
        },
        "header": {
            "props": [
                ("background-color", "#fafafa"),
                ("color", "#000000"),
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
                ("border-bottom", "2px solid #333333"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#fafafa"),
                ("color", "#b22222"),
                ("font-size", "22px"),
                ("font-family", "Georgia, serif"),
                ("border-bottom", "2px solid #000000"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#ffffff"),
                ("color", "#333333"),
                ("font-family", "Georgia, serif"),
                ("font-style", "italic"),
                ("font-size", "16px"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#f4f4f4"),
                ("color", "#000000"),
                ("font-family", "Georgia, serif"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#fafafa"),
                ("color", "#000000"),
                ("font-family", "Georgia, serif"),
            ]
        },
        "hover": {"props": [("background-color", "#e0e0e0"), ("color", "#000000")]},
    },
    "chalkboard_purple": {
        "table": {"props": [("border", "2px solid #673ab7")]},  # Purple frame
        "header": {
            "props": [
                ("background-color", "#7e57c2"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#7e57c2"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#5e35b1"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#673ab7"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#9575cd"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "hover": {"props": [("background-color", "#512da8")]},  # Darker purple hover
    },
    "wooden_borders": {
        "table": {
            "props": [
                # Add a border to simulate the wooden frame
                ("border", "10px solid #8b4513"),  # Rich brown wood tone
                # Add an inner shadow to create depth
                (
                    "box-shadow",
                    "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
                ),
                # Optional: Rounded corners for a smoother finish
                ("border-radius", "8px"),
            ]
        }
    },
    "chalkboard_green": {
        "table": {
            "props": [
                ("border-top", "5px Solid #8b4513"),  # Inset for top border
                ("border-right", "5px Solid #8b4513"),  # Outset for right border
                ("border-bottom", "5px Solid #8b4513"),  # Outset for bottom border
                ("border-left", "5px Solid #8b4513"),  # Inset for left border
                (
                    "box-shadow",
                    "0 0 10px rgba(0, 0, 0, 0.5), inset 0 0 5px rgba(139, 69, 19, 0.8)",
                ),  # Inner shadow remains the same
            ]
        },
        "header": {
            "props": [
                ("background-color", "#355e3b"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#355e3b"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#4a7c59"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#3c6e47"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#4a7c59"), ("color", "#ffffff")]},
        "hover": {"props": [("background-color", "#2c5a33")]},
    },
    "chalkboard_black": {
        "table": {"props": [("border", "2px solid #8b4513")]},  # Dark wooden frame
        "header": {
            "props": [
                ("background-color", "#2b2b2b"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#2b2b2b"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#333333"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#3c3c3c"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#454545"), ("color", "#ffffff")]},
        "hover": {"props": [("background-color", "#1c1c1c")]},  # Intense black hover
    },
    "chalkboard_red": {
        "table": {"props": [("border", "2px solid #5c1e1e")]},  # Deep burgundy frame
        "header": {
            "props": [
                ("background-color", "#731919"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#731919"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#5c1e1e"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#822626"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#6e1e1e"), ("color", "#ffffff")]},
        "hover": {"props": [("background-color", "#4a1010")]},  # Deep burgundy hover
    },
    "chalkboard_yellow": {
        "table": {"props": [("border", "2px solid #d1a054")]},  # Golden frame
        "header": {
            "props": [
                ("background-color", "#d4a017"),
                ("color", "#000000"),
                ("font-family", "Courier New, monospace"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#d4a017"),
                ("color", "#000000"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#e3b436"), ("color", "#000000")]
        },
        "row": {"props": [("background-color", "#f2c849"), ("color", "#000000")]},
        "alt_row": {"props": [("background-color", "#e8c04a"), ("color", "#000000")]},
        "hover": {"props": [("background-color", "#cfa524")]},  # Rich golden hover
    },
    "chalkboard_blue": {
        "table": {"props": [("border", "2px solid #1d3f73")]},  # Navy frame
        "header": {
            "props": [
                ("background-color", "#2c528c"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#2c528c"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#1d3f73"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#305e91"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#457bc1"), ("color", "#ffffff")]},
        "hover": {"props": [("background-color", "#193a71")]},  # Dark navy hover
    },
    "chalkboard_teal": {
        "table": {"props": [("border", "2px solid #00695c")]},  # Teal frame
        "header": {
            "props": [
                ("background-color", "#00897b"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#00897b"),
                ("color", "#ffffff"),
                ("font-size", "20px"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#00796b"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#004d40"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#00695c"),
                ("color", "#ffffff"),
                ("font-family", "Courier New, monospace"),
            ]
        },
        "hover": {"props": [("background-color", "#004d40")]},  # Darker teal hover
    },
    "outer_space": {
        "table": {"props": [("border", "2px solid #6c63ff")]},  # Galaxy-inspired border
        "header": {
            "props": [
                ("background-color", "#2b2d42"),
                ("color", "#ffffff"),
                ("font-family", "Orbitron, sans-serif"),
                ("font-weight", "bold"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#2b2d42"),
                ("color", "#ffffff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#3a3d59"),
                ("color", "#ffffff"),
                ("font-family", "Orbitron, sans-serif"),
            ]
        },
        "row": {"props": [("background-color", "#1d1d1d"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#333366"), ("color", "#ffffff")]},
        "hover": {
            "props": [
                ("background-color", "#4b0082"),
                ("box-shadow", "0 4px 10px rgba(75, 0, 130, 0.8)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "underwater": {
        "table": {"props": [("border", "2px solid #4682b4")]},  # Sea-inspired border
        "header": {
            "props": [
                ("background-color", "#00ced1"),
                ("color", "#ffffff"),
                ("font-family", "Trebuchet MS, sans-serif"),
                ("font-weight", "bold"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#00ced1"),
                ("color", "#ffffff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#20b2aa"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#87ceeb"), ("color", "#006994")]},
        "alt_row": {"props": [("background-color", "#afeeee"), ("color", "#006994")]},
        "hover": {
            "props": [
                ("background-color", "#1e90ff"),
                ("box-shadow", "0 4px 10px rgba(30, 144, 255, 0.6)"),
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "warm_orange_purple": {
        "header": {
            "props": [
                ("background-color", "#4B0082"),  # Deep indigo
                ("color", "#FFDAB9"),             # Soft peach for header text
                ("font-family", "Georgia, serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#FF4500"),  # Vibrant orange-red for column headings
                ("color", "#FFF5EE"),             # Seashell white text for contrast
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#800080"),  # Rich purple for row headings
                ("color", "#FFDAB9"),             # Soft peach text
            ]
        },
        "row": {
            "props": [
                ("background-color", "#FFDEAD"),  # Navajo white for rows
                ("color", "#4B0082"),             # Deep indigo text for readability
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#FFE4B5"),  # Moccasin for alternate rows
                ("color", "#4B0082"),             # Deep indigo text
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#FF7F50"),  # Coral on hover for a bright accent
                ("color", "#4B0082"),
                ("transition", "background-color 0.5s ease"),
            ]
        },
    },
    "graffiti": {
        "table": {"props": [("border", "2px solid #ffa500")]},  # Graffiti-style border
        "header": {
            "props": [
                ("background-color", "#1e90ff"),
                ("color", "#ffffff"),
                ("font-family", "Permanent Marker, cursive"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ff6347"),
                ("color", "#ffffff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#32cd32"), ("color", "#ffffff")]
        },
        "row": {"props": [("background-color", "#ff4500"), ("color", "#ffffff")]},
        "alt_row": {"props": [("background-color", "#ffa500"), ("color", "#ffffff")]},
        "hover": {
            "props": [
                ("background-color", "#1e90ff"),
                ("box-shadow", "0 0 15px rgba(30, 144, 255, 0.9)"),
            ]
        },
    },
    "sci_fi": {
        "table": {
            "props": [("border", "2px solid #7fdbff")]
        },  # Hologram-like cyan border
        "header": {
            "props": [
                ("background-color", "#001f3f"),
                ("color", "#7fdbff"),
                ("font-family", "Orbitron, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#001f3f"),
                ("color", "#7fdbff"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#011627"), ("color", "#7fdbff")]
        },
        "row": {"props": [("background-color", "#000000"), ("color", "#7fdbff")]},
        "alt_row": {"props": [("background-color", "#011627"), ("color", "#7fdbff")]},
        "hover": {
            "props": [
                ("background-color", "#7fdbff"),
                ("color", "#001f3f"),
                ("box-shadow", "0 0 15px rgba(127, 219, 255, 0.9)"),
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "sci_fi_green": {
        "table": {"props": [("border", "2px solid #39FF14")]},  # Hologram-like neon green border
        "header": {
            "props": [
                ("background-color", "#001f00"),  # Very dark green/black background
                ("color", "#39FF14"),             # Neon green text
                ("font-family", "Orbitron, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#001f00"),
                ("color", "#39FF14"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#012200"), ("color", "#39FF14")]
        },
        "row": {"props": [("background-color", "#000000"), ("color", "#39FF14")]},
        "alt_row": {"props": [("background-color", "#012200"), ("color", "#39FF14")]},
        "hover": {
            "props": [
                ("background-color", "#39FF14"),
                ("color", "#001f00"),
                ("box-shadow", "0 0 15px rgba(57, 255, 20, 0.9)"),
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "sci_fi_magenta": {
        "table": {"props": [("border", "2px solid #FF00FF")]},  # Hologram-like magenta border
        "header": {
            "props": [
                ("background-color", "#300033"),  # Dark purple background
                ("color", "#FF00FF"),             # Vivid magenta text
                ("font-family", "Orbitron, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#300033"),
                ("color", "#FF00FF"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#3A0040"), ("color", "#FF00FF")]
        },
        "row": {"props": [("background-color", "#1A001F"), ("color", "#FF00FF")]},
        "alt_row": {"props": [("background-color", "#3A0040"), ("color", "#FF00FF")]},
        "hover": {
            "props": [
                ("background-color", "#FF00FF"),
                ("color", "#300033"),
                ("box-shadow", "0 0 15px rgba(255, 0, 255, 0.9)"),
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "sci_fi_orange": {
        "table": {"props": [("border", "2px solid #FFA500")]},  # Hologram-like orange border
        "header": {
            "props": [
                ("background-color", "#331900"),  # Deep, dark brownish-orange background
                ("color", "#FFA500"),             # Bright orange text
                ("font-family", "Orbitron, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#331900"),
                ("color", "#FFA500"),
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#442600"), ("color", "#FFA500")]
        },
        "row": {"props": [("background-color", "#1A0D00"), ("color", "#FFA500")]},
        "alt_row": {"props": [("background-color", "#442600"), ("color", "#FFA500")]},
        "hover": {
            "props": [
                ("background-color", "#FFA500"),
                ("color", "#331900"),
                ("box-shadow", "0 0 15px rgba(255, 165, 0, 0.9)"),
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "blue_plain": {
        "header": {
            "props": [
                ("background-color", "#0056b3"),
                ("color", "white"),
                ("text-align", "center"),
                ("font-weight", "bold"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#0056b3"),
                ("color", "white"),
                ("font-size", "24px"),
            ]
        },
        "row_heading": {"props": [("background-color", "#0056b3"), ("color", "white")]},
        "row": {"props": [("background-color", "#f7f7f7"), ("color", "#000000")]},
        "alt_row": {"props": [("background-color", "#ffffff")]},
        "hover": {"props": [("background-color", "#cce5ff")]},
    },
    "parchment": {
        "table": {"props": [("border", "2px solid #5c4327")]},  # Elegant brown borders
        "header": {
            "props": [
                ("background-color", "#f3e2c7"),
                ("color", "#5c4327"),
                ("font-family", "Dancing Script, cursive"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#f3e2c7"),
                ("color", "#5c4327"),
                ("font-family", "EB Garamond, serif"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#f0dbb0"), ("color", "#5c4327")]
        },
        "row": {"props": [("background-color", "#f3e2c7"), ("color", "#5c4327")]},
        "alt_row": {"props": [("background-color", "#f9f2e7"), ("color", "#5c4327")]},
        "hover": {
            "props": [
                ("background-color", "#e8d4af"),
                ("color", "#4a301f"),
                ("box-shadow", "0 0 10px rgba(92, 67, 39, 0.5)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "mist": {
        "table": {"props": [("border", "2px solid #b2dfdb")]},  # teal borders
        "header": {
            "props": [
                ("background-color", "#e0f7fa"),
                ("color", "#00796b"),
                ("font-family", "Cormorant, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#e0f7fa"),
                ("color", "#00796b"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#b2ebf2"), ("color", "#00796b")]
        },
        "row": {"props": [("background-color", "#e0f7fa"), ("color", "#00796b")]},
        "alt_row": {"props": [("background-color", "#b2ebf2"), ("color", "#00796b")]},
        "hover": {
            "props": [
                ("background-color", "#a7c4c7"),
                ("color", "#004d40"),
                ("box-shadow", "0 0 15px rgba(0, 121, 107, 0.5)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "autumn": {
        "table": {"props": [("border", "2px solid #bf360c")]},  # Burnt orange borders
        "header": {
            "props": [
                ("background-color", "#ffcc80"),
                ("color", "#bf360c"),
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ffcc80"),
                ("color", "#bf360c"),
                ("font-size", "20px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#ffe0b2"), ("color", "#bf360c")]
        },
        "row": {"props": [("background-color", "#ffcc80"), ("color", "#bf360c")]},
        "alt_row": {"props": [("background-color", "#ffe0b2"), ("color", "#bf360c")]},
        "hover": {
            "props": [
                ("background-color", "#ffb74d"),
                ("color", "#e65100"),
                ("box-shadow", "0 0 10px rgba(191, 54, 12, 0.5)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "gruvbox_dark": {
        "table": {"props": [("border", "2px solid #ddc7a1")]},  # Gruvbox white borders
        "header": {
            "props": [
                ("background-color", "#2a2827"),  # Gruvbox dark background
                ("color", "#a9b665"),  # Gruvbox green for header text
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#3c3836"),  # Gruvbox medium-dark
                ("color", "#e78a4e"),  # Gruvbox warm yellow for column headings
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#3c3836"), ("color", "#7daea3")],  # Gruvbox blue
        },
        "row": {
            "props": [
                ("background-color", "#202020"),  # Gruvbox dark background for rows
                ("color", "#ddc7a1"),  # Gruvbox white for text
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#282828"),  # Gruvbox medium-dark for alternate rows
                ("color", "#a9b665"),  # Gruvbox green for alternate text
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#3c3836"),  # Gruvbox medium-dark for hover
                ("color", "#7daea3"),  # Gruvbox blue/cyan for brighter text on hover
                ("box-shadow", "0 0 10px rgba(125, 174, 163, 0.5)"),  # Blue shadow to match text color
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "gruvbox_light": {
        "table": {"props": [("border", "2px solid #a9b665")]},  # Gruvbox green for borders
        "header": {
            "props": [
                ("background-color", "#f9f5d7"),  # Gruvbox light-muted
                ("color", "#7c6f64"),  # Gruvbox medium-muted for text
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ebdbb2"),  # Gruvbox light-muted
                ("color", "#d3869b"),  # Gruvbox magenta for column headings
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#fbf1c7"), ("color", "#7daea3")],  # Gruvbox blue
        },
        "row": {"props": [("background-color", "#fbf1c7"), ("color", "#665c54")]},  # Gruvbox muted text
        "alt_row": {"props": [("background-color", "#f2e5bc"), ("color", "#928374")]},  # Muted alternate rows
        "hover": {
            "props": [
                ("background-color", "#a9b665"),  # Gruvbox green for hover
                ("color", "#282828"),  # Dark text for contrast
                ("box-shadow", "0 0 10px rgba(169, 182, 101, 0.5)"),  # Green shadow
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "gruvbox_colorful": {
        "table": {"props": [("border", "2px solid #d3869b")]},  # Gruvbox magenta
        "header": {
            "props": [
                ("background-color", "#7daea3"),  # Gruvbox aqua for header background
                ("color", "#202020"),  # Dark text
                ("font-family", "Comic Sans MS, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#d8a657"),  # Gruvbox yellow
                ("color", "#202020"),  # Dark text
                ("font-size", "18px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#a9b665"), ("color", "#202020")],  # Gruvbox green
        },
        "row": {"props": [("background-color", "#89b482"), ("color", "#282828")]},  # Gruvbox cyan
        "alt_row": {"props": [("background-color", "#7daea3"), ("color", "#282828")]},  # Gruvbox aqua
        "hover": {
            "props": [
                ("background-color", "#d3869b"),  # Gruvbox magenta for hover
                ("color", "#202020"),  # Dark text
                ("box-shadow", "0 0 10px rgba(211, 134, 155, 0.5)"),  # Magenta shadow
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "coffee_shop": {
        "table": {
            "props": [("border", "2px solid #8b4513"), ("border-radius", "8px")]
        },  # brown border
        "header": {
            "props": [
                ("background-color", "#d2b48c"),  # Soft tan
                ("color", "#3e2723"),  # coffee brown text
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
                ("text-align", "center"),
                ("font-weight", "bold"),
                ("text-shadow", "1px 1px 2px #8b4513"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#f5deb3"),  # Wheat color
                ("color", "#3e2723"),
                ("font-family", "Georgia, serif"),
                ("font-size", "20px"),
                ("text-align", "center"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#cdb79e"),  # Warm beige
                ("color", "#3e2723"),
                ("font-family", "Georgia, serif"),
                ("font-weight", "bold"),
                ("text-align", "center"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#fffaf0"),  # Creamy coffee color
                ("color", "#3e2723"),
                ("text-align", "center"),
                ("font-family", "Verdana, sans-serif"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#f5f5dc"),  # Light beige
                ("color", "#3e2723"),
                ("text-align", "center"),
                ("font-family", "Verdana, sans-serif"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#deb887"),  # Toasty brown hover
                (
                    "box-shadow",
                    "0 0 10px rgba(139, 69, 19, 0.4)",
                ),  # Soft coffee-colored glow
            ]
        },
    },
    "dark_purple": {
        "table": {
            "props": [
                ("border", "1px solid #ffffff"),  # White border, mimicking starlight
                (
                    "background",
                    "radial-gradient(circle, #1a1a2e, #0f3460)",
                ),  # Deep night sky gradient
            ]
        },
        "header": {
            "props": [
                ("background-color", "#2c2c54"),  # Dark indigo
                ("color", "#ffffff"),  # White text
                ("font-family", "Trebuchet MS, sans-serif"),  # Modern yet standard font
                ("text-align", "center"),
                ("font-weight", "bold"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#40407a"),  # Soft navy
                ("color", "#ffffff"),
                ("font-family", "Trebuchet MS, sans-serif"),
                ("font-size", "18px"),
                ("text-align", "center"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#474787"),  # Subtle indigo highlight
                ("color", "#ffffff"),
                ("font-family", "Trebuchet MS, sans-serif"),
                ("text-align", "center"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#1a1a2e"),  # Deep cosmic black
                ("color", "#e0e0e0"),  # Light gray text
                ("font-family", "Verdana, sans-serif"),
                ("text-align", "center"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#0f3460"),  # Midnight blue
                ("color", "#e0e0e0"),
                ("text-align", "center"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#22a6b3"),  # Teal hover effect
                ("color", "#ffffff"),
                (
                    "box-shadow",
                    "0 0 10px rgba(34, 166, 179, 0.8)",
                ),  # Glowing star effect
            ]
        },
    },
    "starry_night": {
        "table": {
            "props": [
                ("border", "1px solid #f7d84b"),  # Yellow border, like swirling stars
                (
                    "background",
                    "linear-gradient(to bottom, #2a2a72, #009ffd)",
                ),  # Rich blues for the sky
            ]
        },
        "header": {
            "props": [
                ("background-color", "#344e86"),  # Deep blue
                ("color", "#f7d84b"),  # Starry yellow
                ("font-family", "Georgia, serif"),  # Classic and artistic font
                ("text-align", "center"),
                ("font-weight", "bold"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#3e5ba9"),  # Slightly lighter blue
                ("color", "#f9e79f"),  # Warm yellow
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
                ("text-align", "center"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#283c63"),  # Dark navy
                ("color", "#f7d84b"),
                ("font-family", "Georgia, serif"),
                ("text-align", "center"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#2a2a72"),  # Starry blue
                ("color", "#ffffff"),
                ("font-family", "Verdana, sans-serif"),  # Clean, legible font
                ("text-align", "center"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#009ffd"),  # Swirling blue highlight
                ("color", "#ffffff"),
                ("text-align", "center"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#f7d84b"),  # Starry yellow hover
                ("color", "#2a2a72"),  # Contrasting blue text
                ("box-shadow", "0 0 10px rgba(247, 216, 75, 0.8)"),  # Warm glow effect
            ]
        },
    },
    "appalachian": {
        "table": {"props": [("border", "2px solid #4682B4")]},  # Steel blue border
        "header": {
            "props": [
                ("background-color", "#2E8B57"),  # SeaGreen for header
                ("color", "#F0F8FF"),          # AliceBlue text
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#4682B4"),  # SteelBlue for column headings
                ("color", "#F0F8FF"),               # AliceBlue text
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#4682B4"),
                ("color", "#F0F8FF"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#B0E0E6"),  # PowderBlue for rows
                ("color", "#2F4F4F"),              # DarkSlateGray text
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#AFEEEE"),  # PaleTurquoise for alternate rows
                ("color", "#2F4F4F"),              # DarkSlateGray text
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#5F9EA0"),  # CadetBlue for hover effect
                ("color", "#ffffff"),
                ("box-shadow", "0 0 8px rgba(255, 255, 255, 0.3)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "turtle_shell": {
        "table": {"props": [("border", "2px solid #556b2f")]},  # Olive-green border reminiscent of a turtle's shell
        "header": {
            "props": [
                ("background-color", "#6b8e23"),  # Earthy green background
                ("color", "#f5f5f5"),  # Clean, light text
                ("font-family", "Tahoma, sans-serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#78866b"),  # Subtle, mossy green
                ("color", "#ffffff"),
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#78866b"),
                ("color", "#f5f5f5"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#556b2f"),
                ("color", "#e0e0e0"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#6b8e23"),
                ("color", "#ffffff"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#3e5e1b"),
                ("color", "#ffffff"),
                ("box-shadow", "0 0 8px rgba(230, 230, 230, 0.3)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "sourdough": {
        "table": {"props": [("border", "2px solid #d2a679")]},  # Warm, baked crust border
        "header": {
            "props": [
                ("background-color", "#f4e1d2"),  # Soft, floury background
                ("color", "#8b5a2b"),  # Toasted brown text
                ("font-family", "Verdana, sans-serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#e6d0b3"),  # Warm mid-tone reminiscent of a well-baked loaf
                ("color", "#8b5a2b"),
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#e6d0b3"),
                ("color", "#8b5a2b"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#f4e1d2"),
                ("color", "#5c4033"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#f9e8dc"),  # Lighter, airy tone
                ("color", "#5c4033"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#d2a679"),
                ("color", "#ffffff"),
                ("box-shadow", "0 0 8px rgba(255, 255, 255, 0.3)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "Sakura": {
        "table": {"props": [("border", "2px solid #f7cac9")]},  # Gentle pink border
        "header": {
            "props": [
                ("background-color", "#fff0f5"),  # Soft, blush background
                ("color", "#d6336c"),  # Vibrant cherry blossom pink text
                ("font-family", "Palatino, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#ffeef8"),  # Light and airy pink
                ("color", "#d6336c"),
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#ffeef8"),
                ("color", "#d6336c"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#fff0f5"),
                ("color", "#8a2a2b"),
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#ffeef8"),
                ("color", "#8a2a2b"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#f7cac9"),
                ("color", "#ffffff"),
                ("box-shadow", "0 0 8px rgba(255, 255, 255, 0.3)"),
                ("transform", "scale(1.01)"),
            ]
        },
    },
    "too_much": {
        "table": {"props": [("border", "2px dashed #FF4500")]},  # Bold orange dashed border
        "header": {
            "props": [
                ("background-color", "#000000"),  # Black background
                ("color", "#FF4500"),             # Vibrant orange text
                ("font-family", "Impact, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#FF4500"),  # Orange for column headings
                ("color", "#000000"),             # Black text
                ("font-size", "18px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#FF4500"),  # Consistent orange tone
                ("color", "#000000"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#1a1a1a"),  # Dark gray for rows
                ("color", "#FFFFFF"),             # White text
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#333333"),  # Lighter dark gray for alternate rows
                ("color", "#FFFFFF"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#FF4500"),  # Bright orange on hover
                ("color", "#FFFFFF"),
                ("box-shadow", "0 0 12px rgba(255, 69, 0, 0.7)"),
                ("transform", "scale(1.03)"),
            ]
        },
    },
    "purples": {
        "table": {"props": [("border", "2px dashed #6A5ACD")]},  # SlateBlue dashed border
        "header": {
            "props": [
                ("background-color", "#2D2B55"),  # Dark blue/purple background
                ("color", "#E6E6FA"),             # Soft lavender text
                ("font-family", "Impact, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#6A5ACD"),  # SlateBlue for column headings
                ("color", "#2D2B55"),             # Dark blue text for contrast
                ("font-size", "18px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#6A5ACD"),  # Consistent SlateBlue tone
                ("color", "#2D2B55"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#3F3F7A"),  # Muted indigo for rows
                ("color", "#F5F5F5"),             # Light, off-white text
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#4A4A8A"),  # Slightly lighter indigo for alternate rows
                ("color", "#F5F5F5"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#6A5ACD"),  # SlateBlue on hover
                ("color", "#F5F5F5"),
                ("box-shadow", "0 0 12px rgba(106, 90, 205, 0.7)"),
                ("transform", "scale(1.03)"),
            ]
        },
    },
    "teals": {
        "table": {"props": [("border", "3px solid #00A8E8")]},  # Bold blue border
        "header": {
            "props": [
                ("background-color", "#005F73"),  # Deep teal for header background
                ("color", "#EDF6F9"),             # Off-white text for contrast
                ("font-family", "Helvetica, sans-serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#00A8E8"),  # Vibrant blue for column headings
                ("color", "#005F73"),             # Matching deep teal text
                ("font-size", "18px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#00A8E8"),  # Consistent vibrant blue tone
                ("color", "#005F73"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#CAF0F8"),  # Light, refreshing blue for rows
                ("color", "#023047"),             # Dark blue text for readability
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#ADE8F4"),  # Slightly different blue for alternate rows
                ("color", "#023047"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#90E0EF"),  # Brighter blue on hover for a dynamic feel
                ("color", "#023047"),
                ("box-shadow", "0 0 10px rgba(0, 168, 232, 0.5)"),
                ("transform", "scale(1.02)"),
            ]
        },
    },
    "brass": {
        "table": {"props": [("border", "2px solid #B87333")]},  # Brass-colored border
        "header": {
            "props": [
                ("background-color", "#B87333"),  # Brass background
                ("color", "#FFF8DC"),             # Soft cream text (Cornsilk)
                ("font-family", "Georgia, serif"),
                ("font-size", "20px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#A97142"),  # Slightly darker brass tone
                ("color", "#FFF8DC"),             # Creamy text for contrast
                ("font-size", "22px"),
            ]
        },
        "row_heading": {
            "props": [
                ("background-color", "#B87333"),
                ("color", "#FFF8DC"),
            ]
        },
        "row": {
            "props": [
                ("background-color", "#D2B48C"),  # Tan background for rows
                ("color", "#4B3621"),             # Dark brown text for readability
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#E6D5B8"),  # Lighter tan for alternate rows
                ("color", "#4B3621"),
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#CD7F32"),  # A deeper brass tone on hover
                ("color", "#FFF8DC"),
                ("box-shadow", "0 0 8px rgba(205, 127, 50, 0.5)"),
                ("transition", "background-color 0.3s ease"),
            ]
        },
    },
    "blue": {
        "table": {"props": [("border", "2px solid #a9b1d6")]},  # light gray borders
        "header": {
            "props": [
                ("background-color", "#24283b"),  # dark blue background
                ("color", "#a9b1d6"),  # accent blue for header text
                ("font-family", "Georgia, serif"),
                ("font-size", "18px"),
            ]
        },
        "col_heading": {
            "props": [
                ("background-color", "#2f3549"),  # medium-dark blue background
                ("color", "#a9b1d6"),  # aqua column headings
                ("font-size", "16px"),
            ]
        },
        "row_heading": {
            "props": [("background-color", "#24283b"), ("color", "#a9b1d6")],  # accent blue
        },
        "row": {
            "props": [
                ("background-color", "#24283b"),  # dark background for rows
                ("color", "#a9b1d6"),  # light gray for text
            ]
        },
        "alt_row": {
            "props": [
                ("background-color", "#2f3549"),  # medium-dark blue background
                ("color", "#c0caf5"),  # lighter gray for alternate text
            ]
        },
        "hover": {
            "props": [
                ("background-color", "#2f3549"),  # medium-dark blue for hover
                ("color", "#7dcfff"),  # blue/cyan for brighter text on hover
                ("box-shadow", "0 0 10px rgba(125, 207, 255, 0.5)"),  # Blue shadow to match text color
                ("transform", "scale(1.005)")
            ]
        },
        "inner": {
            "props": [
                ("border-right", "1px solid #c0caf5")
            ]
        },
    },
}

def get_DGCV_themes():
    warnings.warn(
        "`get_DGCV_themes` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library. "
        "It will be removed in 2026. Please use `get_dgcv_themes` instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_dgcv_themes()
def get_dgcv_themes():
    """Returns a list of available theme names (keys) from the style_guide dictionary."""
    return sorted(style_guide.keys())


dgcv_display_theme = 'dark'
def _set_dgcv_default_theme(theme:str):
    global dgcv_display_theme
    dgcv_display_theme = theme

def get_style(theme_name):
    """
    Retrieve the style properties for the given theme name in the required format.
    """
    if theme_name == 'random':
        theme_name = random.choice(list(style_guide.keys()))
    if theme_name == 'chalkboard':
        theme_name = random.choice([
            "chalkboard_green", "chalkboard_blue", "chalkboard_red",
            "chalkboard_black", "chalkboard_teal", "chalkboard_yellow"
        ])
    if theme_name == 'gruv':
        theme_name = random.choice([
            "gruvbox_dark", "gruvbox_colorful", "gruvbox_light"
        ])
    if theme_name == 'dark':
        theme_name = random.choice([
            "gruvbox_dark", "dark_high_contrast_bright", 'dark_blue',
            'dark_high_contrast', 'dark_high_contrast_bright',
            'dark_modern', 'dark_moody', 'dark_purple'
        ])
    if theme_name not in style_guide:
        theme_name = random.choice([
            "gruvbox_dark", "gruvbox_colorful", "chalkboard_teal",
            "chalkboard_black", "blueprint"
        ])

    theme = style_guide[theme_name]

    # Map the refactored keys back to selectors
    selector_mapping = {
        "table": "table",
        "header": "th",
        "col_heading": "th.col_heading.level0",
        "row_heading": "th.row_heading",
        "row": "tbody tr:nth-child(odd)",
        "alt_row": "tbody tr:nth-child(even)",
        "hover": "tbody tr:hover",
        "inner": "th:not(:last-child), td:not(:last-child)",
    }

    # Convert to the correct structure
    formatted_styles = []
    for key, styles in theme.items():
        selector = selector_mapping.get(key, key)
        formatted_styles.append({"selector": selector, "props": styles["props"]})

    return formatted_styles
