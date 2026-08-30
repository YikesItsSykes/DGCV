from __future__ import annotations

from ..._aux._utilities._config import get_dgcv_settings_registry
from ..._aux._utilities._styles import get_style


def html_style(theme=None, container_id=None, slim=False):
    if not isinstance(theme, str):
        theme = get_dgcv_settings_registry().get("theme", "dark")

    scope = f"#{container_id}" if container_id else ""

    if slim:
        return ""

    theme_vars = get_style(theme, legacy=False)
    scoped_vars = theme_vars.replace(":root", scope) if scope else theme_vars

    base_styles = f"""
{scoped_vars}

{scope}.tree-container {{
    padding: 20px;
    overflow-x: auto;
    max-width: 100%;
    white-space: nowrap;
    font-family: var(--dgcv-font-family, sans-serif);
    background: transparent !important;
    color: var(--dgcv-text-main);
}}

{scope}.tree-container ul {{ position: relative; padding-top: 10px; list-style-type: none; margin: 0; }}
{scope}.tree-container li {{ position: relative; padding: 25px 5px 0 40px; list-style-type: none; }}

{scope}.tree-container li::after {{
    content: ""; position: absolute; top: -10px; left: 0;
    border-left: var(--dgcv-border-width, 2px) solid var(--dgcv-border-main); 
    border-bottom: var(--dgcv-border-width, 2px) solid var(--dgcv-border-main);
    width: 40px; height: 52px; border-radius: 0 0 0 10px;
}}

{scope}.tree-container li:not(:last-child)::before {{
    content: ""; position: absolute; top: -10px; left: 0;
    border-left: var(--dgcv-border-width, 2px) solid var(--dgcv-border-main); 
    height: 100%;
}}

{scope}.tree-container > ul > li {{ padding-left: 0; }}
{scope}.tree-container > ul > li::after, {scope}.tree-container > ul > li::before {{ display: none; }}

{scope} .compound-node {{ 
    display: inline-table;
    border-collapse: separate;
    border-spacing: 0;
    position: relative; 
    z-index: 2; 
    transition: var(--dgcv-hover-transition, transform 0.2s, box-shadow 0.2s);
    vertical-align: middle;
    background-color: transparent !important;
}}

{scope} .compound-node:has(label.node-label:hover) {{
    transform: var(--dgcv-hover-transform, none);
}}

{scope} .node-label, {scope} .cond-box, {scope} .complete-msg, {scope} .note-msg, {scope} .fold-count {{
    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    box-shadow: var(--dgcv-table-shadow, none);
    border-image: var(--dgcv-border-image, none);
}}

{scope} .node-label, {scope} .cond-box, {scope} .complete-msg, {scope} .fold-count {{
    display: table-cell;
    padding: 8px 15px;
    vertical-align: middle;
    white-space: normal;
    word-wrap: break-word;
}}

{scope} .cond-box {{ padding: 0; vertical-align: top; }}

{scope} .cond-compartment {{
    position: relative;
    padding: 7px 14px 0;
}}

{scope} .cond-compartment::before {{
    content: "corollaries";
    display: block;
    height: 0;
    visibility: hidden;
    white-space: nowrap;
    font-size: 8px;
    font-style: normal;
    letter-spacing: 0.5px;
    text-transform: lowercase;
    padding: 0 5px;
    margin-left: 6px;
    margin-right: min(2px, var(--dgcv-border-width, 1px));
    box-sizing: content-box;
}}

{scope} .cond-compartment + .cond-compartment {{
    border-top-style: solid;
    border-top-color: var(--dgcv-border-main);
    border-top-width: min(2px, var(--dgcv-border-width, 1px));
    border-image: var(--dgcv-border-image, none);
}}

{scope} .cond-chip {{
    --computed-border-width: min(2px, var(--dgcv-border-width, 1px));
    position: absolute;
    top: 0;
    left: 6px;
    transform: translateY(-50%);
    padding: 0 5px;
    font-size: 8px;
    font-style: normal;
    line-height: 1.2;
    letter-spacing: 0.5px;
    text-transform: lowercase;
    white-space: nowrap;
    background: var(--dgcv-bg-alt);
    color: var(--dgcv-text-alt);
    border-radius: min(4px, var(--dgcv-border-radius, 12px));
}}
{scope} .cond-chip::before {{
    content: "";
    position: absolute;
    top: calc(-1 * var(--computed-border-width));
    left: calc(-1 * var(--computed-border-width));
    right: calc(-1 * var(--computed-border-width));
    bottom: 50%;
    box-sizing: border-box;
    border-style: solid;
    border-color: var(--dgcv-border-main);
    border-width: var(--computed-border-width);
    border-bottom: none;
    border-image: var(--dgcv-border-image, none);
    border-radius: 
        calc(min(4px, var(--dgcv-border-radius, 12px)) + var(--computed-border-width)) 
        calc(min(4px, var(--dgcv-border-radius, 12px)) + var(--computed-border-width)) 
        0 0;
    pointer-events: none;
}}

{scope} .cond-content, {scope} .cell-content {{
    max-width: var(--dgcv-node-content-max-width, min(420px, 60vw));
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
    scrollbar-width: none;
    -ms-overflow-style: none;
    background-image:
        linear-gradient(to right, var(--dgcv-fade-bg, var(--dgcv-bg-surface)), transparent),
        linear-gradient(to left, var(--dgcv-fade-bg, var(--dgcv-bg-surface)), transparent),
        linear-gradient(to right, rgba(128, 128, 128, 0.35), transparent),
        linear-gradient(to left, rgba(128, 128, 128, 0.35), transparent);
    background-position: left center, right center, left center, right center;
    background-repeat: no-repeat;
    background-size: 24px 100%, 24px 100%, 10px 100%, 10px 100%;
    background-attachment: local, local, scroll, scroll;
}}

{scope} .cond-content::-webkit-scrollbar,
{scope} .cell-content::-webkit-scrollbar {{ display: none; }}

{scope} .cond-content {{ font-size: 12px; }}

{scope} .cell-content {{ white-space: pre-line; }}

{scope} .complete-msg .cell-content {{ --dgcv-fade-bg: var(--dgcv-bg-alt); }}

{scope} .complete-msg {{ padding-bottom: 2px; }}

{scope} .node-label, {scope} .complete-msg {{
    max-width: var(--dgcv-node-content-max-width, min(420px, 60vw));
}}

{scope} .complete-msg {{ position: relative; }}

{scope} .sample-chip {{
    white-space: normal;
    text-align: center;
    line-height: 1.15;
}}

{scope} .complete-msg:has(.sample-chip) {{ padding-top: 14px; }}

{scope} .complete-msg:has(.sample-chip)::before {{
    content: "via numeric";
    display: block;
    height: 0;
    visibility: hidden;
    white-space: nowrap;
    font-size: 8px;
    letter-spacing: 0.5px;
    padding: 0 5px;
    margin-left: 6px;
    box-sizing: content-box;
}}

{scope} .node-label {{
    font-weight: bold; 
    font-size: 14px; 
    background: var(--dgcv-special-background, var(--dgcv-bg-surface));
    color: var(--dgcv-special-text,var(--dgcv-text-heading));
    text-shadow: var(--dgcv-text-shadow, none);
}}

{scope} .root-wrapper .node-label {{ 
    border-radius: var(--dgcv-border-radius, 12px) var(--dgcv-border-radius, 12px) 0 0; 
}}
{scope} .root-wrapper:has(.note-msg) .node-label {{ 
    border-radius: var(--dgcv-border-radius, 12px) 0 0 0; 
}}

{scope} .cond-box {{
    font-size: 12px; 
    font-style: italic;
    border-left: none;
    background-color: var(--dgcv-bg-surface);
    color: var(--dgcv-text-heading);
}}

{scope} .compound-node:not(:has(.complete-msg)) .cond-box {{
    border-radius: 0 var(--dgcv-border-radius, 12px) var(--dgcv-border-radius, 12px) 0;
}}

{scope} .compound-node:has(.note-msg):not(:has(.complete-msg)) .cond-box {{
    border-bottom-right-radius: 0;
}}

{scope} .complete-msg {{
    font-size: 11px; 
    border-left: none;
    background-color: var(--dgcv-bg-alt);
    color: var(--dgcv-text-alt);
    border-radius: 0 var(--dgcv-border-radius, 12px) var(--dgcv-border-radius, 12px) 0;
}}

{scope} .compound-node:has(.note-msg) .complete-msg {{ border-radius: 0 var(--dgcv-border-radius, 12px) 0 0; }}

{scope} .note-msg {{
    display: table-caption;
    caption-side: bottom;
    padding: 4px 12px 0;
    font-size: 10px; 
    background-color: var(--dgcv-bg-surface);
    color: var(--dgcv-text-heading);
    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
    border-top: none;
    border-radius: 0 0 var(--dgcv-border-radius, 12px) 0;
    border-image: var(--dgcv-border-image, none);
    white-space: normal;
    word-wrap: break-word;
}}

{scope} .compound-node:not(:has(.complete-msg)) .note-msg {{
    border-bottom-right-radius: 0;
}}

{scope} label.node-label:hover {{
    background: var(--dgcv-bg-hover);
    background-color: var(--dgcv-bg-hover) !important;
    color: var(--dgcv-text-hover) !important;
    border-color: var(--dgcv-text-hover) !important;
}}
{scope} label.node-label:hover .cond-chip {{
    background: var(--dgcv-bg-hover) !important;
    color: var(--dgcv-text-hover) !important;
}}
{scope} label.node-label:hover .cond-chip::before {{
    border-color: var(--dgcv-text-hover);
}}
{scope} .children-ul {{ margin-left: 10px; }}

{scope} .tree-toggle {{
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    margin: 0;
    padding: 0;
    border: none;
    pointer-events: none;
}}

{scope} label.node-label {{ cursor: pointer; }}

{scope} .node-label {{ position: relative; }}

{scope} .node-label::after {{
    content: "case";
    display: block;
    height: 0;
    visibility: hidden;
    white-space: nowrap;
    font-size: 8px;
    font-weight: normal;
    letter-spacing: 0.5px;
    padding: 0 5px;
    margin-left: 6px;
    box-sizing: content-box;
}}

{scope} label.node-label::before {{
    content: "\\25be";
    position: absolute;
    top: 7px;
    left: 6px;
    font-size: 8px;
    font-weight: normal;
    line-height: 1;
    opacity: 0.8;
}}

{scope} .tree-toggle:checked + .compound-node label.node-label::before {{
    content: "\\25b8";
}}

{scope} .node-chip {{
    font-weight: normal;
    text-shadow: none;
    z-index: 3;
}}

{scope} .node-phantom {{ visibility: hidden; }}

{scope} .tree-toggle:focus-visible + .compound-node label.node-label {{
    outline: var(--dgcv-border-width, 2px) solid var(--dgcv-text-hover);
    outline-offset: 2px;
}}

{scope} .tree-toggle:checked ~ .children-ul {{ display: none; }}

{scope} .fold-count {{
    display: none;
    font-size: 11px;
    border-left: none;
    background-color: var(--dgcv-bg-alt);
    color: var(--dgcv-text-alt);
}}

{scope} .tree-toggle:checked + .compound-node .fold-count {{
    display: table-cell;
    border-radius: 0 var(--dgcv-border-radius, 12px) var(--dgcv-border-radius, 12px) 0;
}}

{scope} .tree-toggle:checked + .compound-node:has(.note-msg) .fold-count {{
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}}

{scope} .tree-toggle:checked + .compound-node .cond-box {{ border-radius: 0; }}

{scope} .tree-toggle:checked + .compound-node.root-wrapper .node-label {{
    border-radius: var(--dgcv-border-radius, 12px) 0 0 var(--dgcv-border-radius, 12px);
}}
    """
    return f"<style>{base_styles}</style>"


