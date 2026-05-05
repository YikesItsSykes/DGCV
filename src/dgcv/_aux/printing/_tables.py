"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.printing._tables

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
import html as _html
import numbers
import re
import textwrap
import uuid
from html import escape as _esc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

__all__ = ["TableView", "panel_view", "build_plain_table", "build_matrix_table"]

# -----------------------------------------------------------------------------
# core classes
# -----------------------------------------------------------------------------
CSSRule = Dict[str, object]  # {"selector": str, "props": List[Tuple[str,str]]}


def _props_to_css(props: Iterable[Tuple[str, str]]) -> str:
    return "; ".join(f"{k}: {v}" for (k, v) in props)


def styles_to_css(formatted_styles: List[CSSRule]) -> str:
    rules = []
    for entry in formatted_styles:
        sel = entry.get("selector", "")
        props = entry.get("props", [])
        if sel and props:
            rules.append(f"{sel} {{{_props_to_css(props)}}}")
    rules.append("table { border-collapse: collapse; }")
    return "\n".join(rules)


def merge_styles(*style_lists: List[CSSRule]) -> List[CSSRule]:
    out: List[CSSRule] = []
    for lst in style_lists:
        if lst:
            out.extend(lst)
    return out


def _strip_tags_simple(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return _html.unescape(s)


def _coerce_html(x: object, *, html_safe: bool) -> str:
    if x is None:
        return ""
    s = str(x)
    return s if html_safe else _esc(s)


def _matrix_extras(
    *,
    mirror_header_to_index: bool,
    dashed_corner: bool,
    header_underline_exclude_index: bool,
) -> str:
    main_border = "var(--dgcv-border-width, 1px) solid var(--dgcv-border-main)"
    utility_border = "var(--dgcv-border-width, 1px) dashed var(--dgcv-border-alt)"

    css_lines = [
        ".dgcv-data-table { border-collapse: collapse; }",
        textwrap.dedent(
            """
            .dgcv-data-table th.row_heading {
                background-color: var(--dgcv-bg-surface) !important;
                color: var(--dgcv-text-heading) !important;
                border-right: """
            + main_border
            + """;
                text-shadow: var(--dgcv-text-shadow, none);
                font-weight: bold;
            }
        """
        ).strip(),
    ]

    if header_underline_exclude_index:
        css_lines.append(
            f".dgcv-data-table thead th:not(:first-child) {{ border-bottom: {main_border}; }}"
        )
    else:
        css_lines.append(
            f".dgcv-data-table thead th {{ border-bottom: {main_border}; }}"
        )

    if dashed_corner:
        css_lines.append(
            textwrap.dedent(f"""
                .dgcv-data-table thead th:first-child {{
                    border-right: {utility_border} !important;
                    border-bottom: {utility_border} !important;
                }}
            """).strip()
        )

    if mirror_header_to_index:
        css_lines.append(
            ".dgcv-data-table th.row_heading { color: var(--dgcv-text-heading); font-style: italic; }"
        )

    return "\n".join(css_lines)


def _sanitize_html_str(s: str) -> str:
    import re

    s = re.sub(
        r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>",
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(
        r"\s+on[a-zA-Z]+\s*=\s*(['\"]).*?\1", "", s, flags=re.IGNORECASE | re.DOTALL
    )
    return s


# ------------


class TableView:
    """
    HTML table builder. Renders a styled data table with optional side panel,
    caption, footer rows, and responsive layout. Theme is supplied via
    ``theme_css_vars`` (a ``:root {}`` block from ``get_style()``), which gets
    scoped to the container ID at render time.

    ``extra_css`` is injected after the core styles and is auto-scoped to the
    container ID, making it the hook for per-callsite overrides. Typical uses:
    - column width overrides: ``.dgcv-data-table th:nth-child(n)``
    - tfoot cell styling: ``.dgcv-data-table tfoot td``
    - side panel layout: ``.dgcv-side-panel``, ``.dgcv-side-scroll-area``
    - row heading overrides: ``.dgcv-data-table th.row_heading``

    Note: ``extra_css`` scoping uses a regex that does not support nested
    ``{}`` blocks (e.g. ``@media`` queries). Keep ``extra_css`` flat.

    ``slim=True`` suppresses all theme and layout CSS, emitting only corner
    radius vars. Intended for embedding inside a stacking builder that supplies
    the theme context. Rendering a slim component standalone will produce an
    unstyled output.
    """

    def __init__(
        self,
        columns: List[str],
        rows: List[List[object]],
        *,
        index_labels: Optional[List[object]] = None,
        caption: str = "",
        preface_html: Optional[str] = None,
        theme_css_vars: str = "",
        extra_css: str = "",
        table_attrs: str = 'style="table-layout:fixed; overflow-x:auto;"',
        cell_align: Optional[str] = None,
        column_align: Optional[Dict[Union[int, str], str]] = None,
        escape_cells: bool = True,
        escape_headers: bool = True,
        escape_index: bool = True,
        truncate_chars: Optional[int] = None,
        truncate_msg: str = "output too long; raise `display_length` to see more.",
        nowrap: bool = False,
        secondary_panel_html: Optional[Union[str, Callable[[], str]]] = None,
        layout: str = "row",
        gap_px: int = 16,
        side_width: Union[int, str] = "320px",
        breakpoint_px: int = 900,
        container_id: Optional[str] = None,
        footer_rows: Optional[List[List[object]]] = None,
        ul: Union[int, str] = 10,
        ur: Union[int, str] = 10,
        lr: Union[int, str] = 10,
        ll: Union[int, str] = 10,
        table_scroll=False,
        cell_scroll=False,
        show_headers: bool = True,
        hover_mode: str = "row",
        slim: bool = False,
    ):
        self.columns = columns
        self.rows = rows
        self.footer_rows = footer_rows or []
        self.index_labels = index_labels
        self.caption = caption
        self.theme_css_vars = theme_css_vars
        self.extra_css = extra_css
        self.table_attrs = table_attrs
        self.cell_align = cell_align
        self.escape_cells = escape_cells
        self.escape_headers = escape_headers
        self.escape_index = escape_index
        self.truncate_chars = truncate_chars
        self.truncate_msg = truncate_msg
        self.nowrap = nowrap
        self.secondary_panel_html = secondary_panel_html
        self.layout = layout
        self.gap_px = gap_px
        self.side_width = (
            f"{side_width}px"
            if isinstance(side_width, numbers.Integral)
            else str(side_width)
        )
        self.breakpoint_px = breakpoint_px
        self.container_id = container_id or f"dgcv-view-{uuid.uuid4().hex[:8]}"
        base_id = container_id or f"dgcv-view-{uuid.uuid4().hex[:8]}"
        self.container_id = (
            f"{base_id}-{uuid.uuid4().hex[:8]}" if container_id else base_id
        )
        self.preface_html = preface_html
        self.ul = f"{ul}px" if isinstance(ul, numbers.Integral) else str(ul)
        self.ur = f"{ur}px" if isinstance(ur, numbers.Integral) else str(ur)
        self.lr = f"{lr}px" if isinstance(lr, numbers.Integral) else str(lr)
        self.ll = f"{ll}px" if isinstance(ll, numbers.Integral) else str(ll)
        self.table_scroll = table_scroll
        self.cell_scroll = cell_scroll
        self.show_headers = show_headers
        self.hover_mode = hover_mode
        self.slim = slim

        if column_align:
            name_to_idx = {name: i for i, name in enumerate(columns)}
            _norm: Dict[int, str] = {}
            for k, v in column_align.items():
                if isinstance(k, numbers.Integral) and 0 <= k < len(columns):
                    _norm[k + 1] = v
                elif isinstance(k, str) and k in name_to_idx:
                    _norm[name_to_idx[k] + 1] = v
            self._column_align_idx = _norm
        else:
            self._column_align_idx = None

    def _render_cell(
        self, cell: object, *, tag: str = "td", col_idx: Optional[int] = None
    ) -> str:
        attrs = {}
        use_tag = tag
        if isinstance(cell, dict):
            html_raw = cell.get("html", "")
            attrs = (cell.get("attrs", {}) or {}).copy()
            use_tag = cell.get("tag", use_tag)
            html = _coerce_html(html_raw, html_safe=not self.escape_cells)
        else:
            html = _coerce_html(cell, html_safe=not self.escape_cells)

        align = "left"
        if (
            col_idx is not None
            and self._column_align_idx
            and col_idx in self._column_align_idx
        ):
            align = self._column_align_idx[col_idx]
        elif self.cell_align:
            align = self.cell_align

        existing_style = attrs.get("style", "")
        attrs["style"] = f"text-align: {align}; {existing_style}".strip()

        html = f'<div class="table-cell">{self._truncate(html)}</div>'
        attr_str = "".join(f' {k}="{_esc(str(v))}"' for k, v in attrs.items())
        return f"<{use_tag}{attr_str}>{html}</{use_tag}>"

    def _truncate(self, s: str) -> str:
        if self.truncate_chars is None:
            return s
        if len(_strip_tags_simple(s)) <= self.truncate_chars:
            return s
        return _esc(self.truncate_msg) if self.escape_cells else self.truncate_msg

    def _thead_html(self) -> str:
        if not self.show_headers:
            return ""
        cols = []
        if self.index_labels is not None:
            cols.append('<th scope="col" class="row_heading"></th>')

        for c_idx, c in enumerate(self.columns):
            cell_data = c
            if not isinstance(c, dict):
                cell_data = {"html": c, "attrs": {"class": "col_heading"}}
            else:
                cell_data = c.copy()
                attrs = cell_data.get("attrs", {}).copy()
                attrs["class"] = f"col_heading {attrs.get('class', '')}".strip()
                cell_data["attrs"] = attrs

            cols.append(self._render_cell(cell_data, tag="th", col_idx=c_idx + 1))

        return "<thead><tr>" + "".join(cols) + "</tr></thead>"

    def _tbody_html(self) -> str:
        has_index = self.index_labels is not None
        body = []
        for r_idx, row in enumerate(self.rows):
            tds = []
            if has_index:
                idx_val = "" if self.index_labels is None else self.index_labels[r_idx]
                idx_html = _coerce_html(idx_val, html_safe=not self.escape_index)
                tds.append(f'<th scope="row" class="row_heading">{idx_html}</th>')

            for c_idx, cell in enumerate(row):
                tds.append(self._render_cell(cell, tag="td", col_idx=c_idx + 1))
            body.append("<tr>" + "".join(tds) + "</tr>")
        return "<tbody>" + "".join(body) + "</tbody>"

    def _tfoot_html(self) -> str:
        if not self.footer_rows:
            return ""
        has_index = self.index_labels is not None
        rows_html = []
        for row in self.footer_rows:
            tds = []
            if has_index:
                tds.append('<th scope="row" class="row_heading"></th>')
            for cell in row:
                tds.append(self._render_cell(cell, tag="td"))
            rows_html.append("<tr>" + "".join(tds) + "</tr>")
        return "<tfoot>" + "".join(rows_html) + "</tfoot>"

    def _caption_html(self) -> str:
        if not self.caption:
            return ""
        t = _coerce_html(self.caption, html_safe=False)
        return f'<div class="dgcv-table-caption">{t}</div>'

    def _panel_html(self) -> Optional[str]:
        if self.secondary_panel_html is None:
            return None
        html = (
            self.secondary_panel_html()
            if callable(self.secondary_panel_html)
            else self.secondary_panel_html
        )
        return html or ""

    def _theme_and_layout_css(self) -> str:
        cid = self.container_id
        has_panel = bool(self._panel_html())

        direction = "row" if self.layout == "row" else "column"
        r_tl, r_tr, r_br, r_bl = self.ul, self.ur, self.ll, self.lr
        gap = int(self.gap_px)

        if has_panel and direction == "row":
            m_tr, m_br, m_bl = "0px", "0px", r_bl
            s_tl, s_tr, s_br, s_bl = "0px", r_tr, r_br, "0px"
        elif has_panel and direction == "column":
            m_tr, m_br, m_bl = r_tr, "0px", "0px"
            s_tl, s_tr, s_br, s_bl = "0px", "0px", r_br, r_bl
        else:
            m_tr, m_br, m_bl = r_tr, r_br, r_bl
            s_tl, s_tr, s_br, s_bl = r_tl, r_tr, r_br, r_bl

        if has_panel:
            mq_m_tr, mq_m_br, mq_m_bl = r_tr, "0px", "0px"
            mq_s_tl, mq_s_tr, mq_s_br, mq_s_bl = "0px", "0px", r_br, r_bl
        else:
            mq_m_tr, mq_m_br, mq_m_bl = r_tr, r_br, r_bl
            mq_s_tl, mq_s_tr, mq_s_br, mq_s_bl = r_tl, r_tr, r_br, r_bl

        if self.slim:
            return textwrap.dedent(f"""
                <style>
                #{cid} {{
                    --m-r-tr: {m_tr}; --m-r-br: {m_br}; --m-r-bl: {m_bl};
                    --s-r-tl: {s_tl}; --s-r-tr: {s_tr}; --s-r-br: {s_br}; --s-r-bl: {s_bl};
                }}
                @media (max-width: {int(self.breakpoint_px)}px) {{
                    #{cid} {{
                        --m-r-tr: {mq_m_tr}; --m-r-br: {mq_m_br}; --m-r-bl: {mq_m_bl};
                        --s-r-tl: {mq_s_tl}; --s-r-tr: {mq_s_tr}; --s-r-br: {mq_s_br}; --s-r-bl: {mq_s_bl};
                    }}
                }}
                </style>
            """).strip()

        scoped_vars = (
            self.theme_css_vars.replace(":root", f"#{cid}")
            if self.theme_css_vars
            else ""
        )
        scoped_extra = (
            _scope_extra_css(self.extra_css, self.container_id)
            if self.extra_css
            else ""
        )

        r_tl_eff = "0px" if self.caption else r_tl
        f_br, f_bl = (
            ("var(--m-r-br)", "var(--m-r-bl)")
            if not self.footer_rows
            else ("0px", "0px")
        )

        scroll_css = f"width: {'max-content; min-width: 100%' if self.table_scroll else '100%'}; table-layout: fixed; border-collapse: separate; border-spacing: 0;"
        wrap_css = f"overflow-x: auto; max-width: 100%; {'contain: inline-size;' if self.table_scroll else ''}"
        cell_scroll_css = f"#{cid} .dgcv-data-table td .table-cell {{ overflow-x: {'auto' if self.cell_scroll else 'visible'}; white-space: {'nowrap' if self.cell_scroll else 'normal'}; }}"

        if getattr(self, "show_headers", True):
            tl_left, tl_right = (
                "thead tr:first-child th:first-child",
                "thead tr:first-child th:last-child",
            )
        else:
            has_idx = self.index_labels is not None
            tl_left = (
                "tbody tr:first-child th.row_heading"
                if has_idx
                else "tbody tr:first-child td:first-child"
            )
            tl_right = "tbody tr:first-child td:last-child"

        h_bg = "var(--dgcv-bg-hover)"
        h_txt = "var(--dgcv-text-hover)"
        h_trans = "var(--dgcv-hover-transform, none)"
        h_speed = "var(--dgcv-hover-transition, transform 0.2s, box-shadow 0.2s, background-color 0.2s)"

        if self.hover_mode == "cell":
            hover_css = textwrap.dedent(f"""
                #{cid} .dgcv-data-table tbody td {{ transition: {h_speed}; }}
                #{cid} .dgcv-data-table tbody td:hover {{ 
                    background-color: {h_bg} !important; 
                    color: {h_txt} !important; 
                    transform: {h_trans}; 
                }}""").strip()
        elif self.hover_mode == "row":
            hover_css = textwrap.dedent(f"""
                #{cid} .dgcv-data-table tbody tr {{ transition: {h_speed}; }}
                #{cid} .dgcv-data-table tbody tr:hover td,
                #{cid} .dgcv-data-table tbody tr:hover th.row_heading {{ 
                    background-color: {h_bg} !important; 
                    color: {h_txt} !important; 
                    transform: {h_trans}; 
                }}
                #{cid} .dgcv-data-table tbody tr:hover th.row_heading {{
                    background-color: var(--dgcv-bg-surface) !important;
                    color: var(--dgcv-text-heading) !important;
                    transform: none !important;
                }}""").strip()
        else:
            hover_css = ""

        if has_panel and direction == "row":
            flex_css = f"""display: grid; grid-template-columns: 1fr {self.side_width}; grid-template-rows: auto 1fr; column-gap: {gap}px; align-items: start;"""
            main_css = """display: grid; grid-row: 1 / -1; grid-column: 1; grid-template-rows: subgrid;"""
            side_css = """grid-row: 2; grid-column: 2; box-sizing: border-box; display: flex; flex-direction: column; align-items: stretch; max-width: 100%;"""
        else:
            flex_css = f"""display: flex; flex-direction: {direction}; gap: {gap}px; align-items: flex-start;"""
            main_css = """flex: 0 1 auto; max-width: 100%; min-width: 0; display: flex; flex-direction: column;"""
            side_css = f"""flex: 0 0 {self.side_width}; max-width: 40%; box-sizing: border-box; display: flex; flex-direction: column; align-items: stretch;"""

        or_m_tr = "var(--dgcv-border-radius, var(--m-r-tr))"
        or_m_br = "var(--dgcv-border-radius, var(--m-r-br))"
        or_m_bl = "var(--dgcv-border-radius, var(--m-r-bl))"
        or_s_tl = "var(--dgcv-border-radius, var(--s-r-tl))"
        or_s_tr = "var(--dgcv-border-radius, var(--s-r-tr))"
        or_s_br = "var(--dgcv-border-radius, var(--s-r-br))"
        or_s_bl = "var(--dgcv-border-radius, var(--s-r-bl))"
        or_tl_eff = _outer_radius(r_tl_eff)
        or_f_br = _outer_radius(f_br)
        or_f_bl = _outer_radius(f_bl)

        return textwrap.dedent(f"""
            <style>
            {scoped_vars}
            #{cid} {{
                max-width: 100%;
                overflow-x: hidden;
                font-family: var(--dgcv-font-family, inherit);
                --m-r-tr: {m_tr}; --m-r-br: {m_br}; --m-r-bl: {m_bl};
                --s-r-tl: {s_tl}; --s-r-tr: {s_tr}; --s-r-br: {s_br}; --s-r-bl: {s_bl};
            }}
            #{cid} .dgcv-flex {{ {flex_css} }}
            #{cid} .dgcv-main {{ {main_css} }}
            #{cid} .dgcv-side {{ {side_css} }}
            #{cid} .dgcv-caption-wrapper {{ display: flex; justify-content: flex-start; align-items: flex-end; width: 100%; margin: 0; padding: 0; }}
            #{cid} .dgcv-table-caption {{ 
                background: var(--dgcv-special-background, var(--dgcv-bg-surface)); 
                color: var(--dgcv-special-text,var(--dgcv-text-heading));
                font-weight: bold; 
                padding: 8px 20px; 
                border-top-left-radius: var(--dgcv-border-radius, 12px); 
                border-top-right-radius: var(--dgcv-border-radius, 12px); 
                border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
                border-bottom: none; 
                box-sizing: border-box; 
                text-align: left; 
                width: fit-content; 
                max-width: 100%; 
                margin: 0;
                text-shadow: var(--dgcv-text-shadow, none);
            }}
            #{cid} .dgcv-data-table {{ 
                {scroll_css} 
                background: var(--dgcv-special-background, var(--dgcv-bg-surface)); 
                color: var(--dgcv-special-text,var(--dgcv-text-heading));
                border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
                border-image: var(--dgcv-border-image, none);
                box-shadow: var(--dgcv-table-shadow, none); 
                border-radius: {or_tl_eff} {or_m_tr} {or_m_br} {or_m_bl}; 
                overflow: hidden; 
            }}
            #{cid} .dgcv-table-wrap {{ {wrap_css} }}
            #{cid} .dgcv-side-panel {{ 
                border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
                border-radius: {or_s_tl} {or_s_tr} {or_s_br} {or_s_bl};
                color: var(--dgcv-special-text,var(--dgcv-text-heading));
                overflow: hidden; 
                background: var(--dgcv-special-background,var(--dgcv-bg-surface)); 
            }}
            {cell_scroll_css}
            #{cid} .dgcv-data-table thead th, #{cid} .dgcv-data-table th.col_heading {{ 
                background-color: var(--dgcv-bg-surface); 
                color: var(--dgcv-text-heading); 
                border-bottom: 2px solid var(--dgcv-border-main); 
                padding: 8px 12px; 
                text-shadow: var(--dgcv-text-shadow, none);
            }}
            #{cid} .dgcv-data-table th.row_heading {{ 
                background-color: var(--dgcv-bg-surface) !important; 
                color: var(--dgcv-text-heading) !important; 
                padding: 8px 12px; 
                font-weight: bold;
                text-shadow: var(--dgcv-text-shadow, none);
            }}
            #{cid} .dgcv-data-table tbody td, 
            #{cid} .dgcv-data-table tbody tr:nth-child(odd) th.row_heading {{ 
                background-color: var(--dgcv-bg-primary);
                color: var(--dgcv-text-main);
            }}
            #{cid} .dgcv-data-table tbody tr:nth-child(even) td,
            #{cid} .dgcv-data-table tbody tr:nth-child(even) th.row_heading {{ 
                background-color: var(--dgcv-bg-alt);
                color: var(--dgcv-text-alt);
            }}
            #{cid} .dgcv-data-table tbody td {{ 
                padding: 8px 12px; 
                border-bottom: 1px solid var(--dgcv-border-alt); 
            }}
            {hover_css}
            #{cid} .dgcv-data-table {tl_left}  {{ border-top-left-radius: {or_tl_eff}; }}
            #{cid} .dgcv-data-table {tl_right} {{ border-top-right-radius: {or_m_tr}; }}
            #{cid} .dgcv-data-table tfoot tr:last-child td:first-child {{ border-bottom-left-radius: {or_m_bl}; }}
            #{cid} .dgcv-data-table tbody tr:last-child td:first-child, #{cid} .dgcv-data-table tbody tr:last-child th.row_heading {{ border-bottom-left-radius: {or_f_bl}; }}
            #{cid} .dgcv-data-table tfoot tr:last-child td:last-child  {{ border-bottom-right-radius: {or_m_br}; }}
            #{cid} .dgcv-data-table tbody tr:last-child td:last-child  {{ border-bottom-right-radius: {or_f_br}; }}
            #{cid} *::-webkit-scrollbar {{ height: 8px; width: 8px; }}
            #{cid} *::-webkit-scrollbar-track {{ background: transparent; }}
            #{cid} *::-webkit-scrollbar-thumb {{ 
                background: var(--dgcv-bg-alt); 
                border-radius: 10px; 
                border: 2px solid transparent; 
                background-clip: content-box; 
            }}
            #{cid} *::-webkit-scrollbar-thumb:hover {{ 
                background-color: var(--dgcv-border-main); 
            }}
            @media (max-width: {int(self.breakpoint_px)}px) {{
                #{cid} .dgcv-flex {{ display: flex; flex-direction: column; align-items: flex-start; row-gap: {gap}px; }}
                #{cid} .dgcv-main, #{cid} .dgcv-side {{ max-width: 100%; width: 100%; }}
                #{cid} .dgcv-main {{ display: flex; flex-direction: column; grid-row: unset; grid-column: unset; }}
                #{cid} .dgcv-side {{ grid-row: unset; grid-column: unset; }}
                #{cid} {{
                    --m-r-tr: {mq_m_tr};
                    --m-r-br: {mq_m_br};
                    --m-r-bl: {mq_m_bl};
                    --s-r-tl: {mq_s_tl};
                    --s-r-tr: {mq_s_tr};
                    --s-r-br: {mq_s_br};
                    --s-r-bl: {mq_s_bl};
                }}
            }}
            {scoped_extra}
            </style>
        """).strip()

    def _table_html_only(self) -> str:
        thead = self._thead_html()
        tbody = self._tbody_html()
        tfoot = self._tfoot_html()
        return (
            f'<table class="dgcv-data-table" {self.table_attrs}>'
            f"{thead}"
            f"{tbody}"
            f"{tfoot}"
            f"</table>"
        )

    def to_html(self, *args, **kwargs) -> str:
        panel = self._panel_html()
        table_html = self._table_html_only()
        cap_html = self._caption_html()
        preface = self.preface_html or ""
        css_block = self._theme_and_layout_css()

        if not panel:
            return (
                f'<div id="{self.container_id}">'
                f"{css_block}{preface}"
                f'<div class="dgcv-caption-wrapper">{cap_html}</div>'
                f'<div class="dgcv-table-wrap">{table_html}</div>'
                f"</div>"
            )

        return (
            f'<div id="{self.container_id}">'
            f"{css_block}{preface}"
            f'<div class="dgcv-flex">'
            f'  <div class="dgcv-main">'
            f'    <div class="dgcv-caption-wrapper">{cap_html}</div>'
            f'    <div class="dgcv-table-wrap">{table_html}</div>'
            f"  </div>"
            f'  <aside class="dgcv-side">{panel}</aside>'
            f"</div></div>"
        )

    def _repr_html_(self) -> str:
        return self.to_html()

    def to_text(self, col_sep: str = " | ") -> str:
        cols = list(self.columns)
        if self.index_labels is not None:
            cols = [""] + cols
        header = col_sep.join(cols)
        sep = "-" * max(3, len(header))
        lines = [header, sep]
        for i, row in enumerate(self.rows):
            cells = [
                _strip_tags_simple(_coerce_html(c, html_safe=not self.escape_cells))
                if not isinstance(c, dict)
                else _strip_tags_simple(
                    _coerce_html(c.get("html", ""), html_safe=not self.escape_cells)
                )
                for c in row
            ]
            if self.truncate_chars is not None:
                cells = [
                    c
                    if len(c) <= self.truncate_chars
                    else _strip_tags_simple(self.truncate_msg)
                    for c in cells
                ]
            if self.index_labels is not None:
                idx = _strip_tags_simple(
                    _coerce_html(self.index_labels[i], html_safe=not self.escape_index)
                )
                lines.append(col_sep.join([idx] + cells))
            else:
                lines.append(col_sep.join(cells))
        for frow in self.footer_rows:
            cells = [
                _strip_tags_simple(_coerce_html(c, html_safe=not self.escape_cells))
                if not isinstance(c, dict)
                else _strip_tags_simple(
                    _coerce_html(c.get("html", ""), html_safe=not self.escape_cells)
                )
                for c in frow
            ]
            lines.append(col_sep.join(cells))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    def to_plain_text(self, col_sep: str = " | ") -> str:
        return self.to_text(col_sep=col_sep)


class panel_view:
    """
    HTML panel builder. Renders a styled card with a header, optional body
    text, itemized list, and footer. Theme is supplied via ``theme_css_vars``
    (a ``:root {}`` block from ``get_style()``), scoped to the container ID.

    ``extra_css`` is auto-scoped to the container ID. Typical uses:
    - panel body layout: ``.dgcv-panel-body``
    - title sizing: ``.dgcv-panel-title``
    - chip/inline list styling: ``.dgcv-chip``, ``.dgcv-inline``
    - footer overrides: ``.dgcv-panel-footer``

    Note: ``extra_css`` scoping does not support nested ``{}`` blocks.

    ``slim=True`` emits only corner radius CSS, for use inside stacking
    builders. Standalone rendering of a slim panel will be unstyled.
    """

    def __init__(
        self,
        *,
        header: Union[str, Any],
        primary_text: Optional[Union[str, Any]] = None,
        itemized_text: Optional[Union[List[Union[str, Any]], Tuple]] = None,
        footer: Optional[Union[str, Any]] = None,
        theme_css_vars: str = "",
        extra_css: str = "",
        list_variant: str = "bulleted",
        use_latex: bool = False,
        sanitize: bool = True,
        container_id: Optional[str] = None,
        ul: Union[int, str] = 10,
        ur: Union[int, str] = 10,
        lr: Union[int, str] = 10,
        ll: Union[int, str] = 10,
        slim: bool = False,
    ):
        self.header = header
        self.primary_text = primary_text
        self.itemized_text = list(itemized_text) if itemized_text else []
        self.footer = footer
        self.theme_css_vars = theme_css_vars
        self.extra_css = extra_css
        self.list_variant = list_variant
        self.use_latex = use_latex
        self.sanitize = sanitize
        self.container_id = container_id or f"dgcv-panel-{uuid.uuid4().hex[:8]}"
        base_id = container_id or f"dgcv-panel-{uuid.uuid4().hex[:8]}"
        self.container_id = (
            f"{base_id}-{uuid.uuid4().hex[:8]}" if container_id else base_id
        )
        self.ul = f"{ul}px" if isinstance(ul, numbers.Integral) else str(ul)
        self.ur = f"{ur}px" if isinstance(ur, numbers.Integral) else str(ur)
        self.lr = f"{lr}px" if isinstance(lr, numbers.Integral) else str(lr)
        self.ll = f"{ll}px" if isinstance(ll, numbers.Integral) else str(ll)
        self.slim = slim

    def _coerce_block(self, x) -> str:
        if x is None:
            return ""
        if hasattr(x, "to_html"):
            s = x.to_html()
        else:
            s = _coerce_html(x, html_safe=False)
        if self.sanitize:
            s = _sanitize_html_str(s)
        return s

    def _layout_css(self) -> str:
        cid = self.container_id

        r_tl, r_tr, r_br, r_bl = self.ul, self.ur, self.lr, self.ll

        if self.slim:
            return textwrap.dedent(f"""
                <style>
                #{cid} .dgcv-panel {{
                        border-radius: {_outer_radius(r_tl)} {_outer_radius(r_tr)} {_outer_radius(r_br)} {_outer_radius(r_bl)};
                }}
                #{cid} .dgcv-panel-head {{
                    border-top-left-radius: {_outer_radius(f"calc({r_tl} - var(--dgcv-border-width, 1px))")};
                    border-top-right-radius: {_outer_radius(f"calc({r_tr} - var(--dgcv-border-width, 1px))")};
                }}
                #{cid} .dgcv-panel-footer {{
                    border-bottom-left-radius: {_outer_radius(f"calc({r_bl} - var(--dgcv-border-width, 1px))")};
                    border-bottom-right-radius: {_outer_radius(f"calc({r_br} - var(--dgcv-border-width, 1px))")};
                }}
                </style>
            """).strip()

        scoped_vars = (
            self.theme_css_vars.replace(":root", f"#{cid}")
            if self.theme_css_vars
            else ""
        )
        scoped_extra = (
            _scope_extra_css(self.extra_css, self.container_id)
            if self.extra_css
            else ""
        )

        return textwrap.dedent(f"""
                <style>
                {scoped_vars}
                #{cid} .dgcv-panel {{
                    border-radius: {r_tl} {r_tr} {r_br} {r_bl};
                    background: var(--dgcv-special-background, var(--dgcv-bg-surface));
                    box-shadow: var(--dgcv-table-shadow, none);
                    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
                    border-image: var(--dgcv-border-image, none);
                    color: var(--dgcv-special-text,var(--dgcv-text-heading));
                    font-family: var(--dgcv-font-family, inherit);
                    overflow: hidden;
                }}
                #{cid} .dgcv-panel-head {{ 
                    margin: 0; 
                    padding: 0.75rem 1rem; 
                    background: var(--dgcv-special-background, var(--dgcv-bg-surface));
                    border-top-left-radius: calc({r_tl} - var(--dgcv-border-width, 1px));
                    border-top-right-radius: calc({r_tr} - var(--dgcv-border-width, 1px));
                }}
                #{cid} .dgcv-panel-title {{ 
                    margin: 0; 
                    font-size: 1rem; 
                    line-height: 1.3; 
                    font-weight: 600; 
                    color: var(--dgcv-special-text,var(--dgcv-text-heading));
                    text-shadow: var(--dgcv-text-shadow, none);
                }}
                #{cid} .dgcv-panel-rule {{ 
                    border: 0; 
                    height: 2px; 
                    background: var(--dgcv-border-main); 
                    margin: 0; 
                }}
                #{cid} .dgcv-panel-body {{ 
                    padding: 0.75rem 1rem; 
                }}
                #{cid} .dgcv-panel-footer {{ 
                    padding: 0.5rem 1rem; 
                    background: var(--dgcv-bg-alt); 
                    color: var(--dgcv-text-alt);
                    border-top: 1px solid var(--dgcv-border-alt);
                    border-bottom-left-radius: calc({r_bl} - var(--dgcv-border-width, 1px));
                    border-bottom-right-radius: calc({r_br} - var(--dgcv-border-width, 1px));
                }}
                #{cid} .dgcv-panel-list {{ 
                    margin: 0.5rem 0 0; 
                    padding: 0; 
                }}
                #{cid} .dgcv-panel-list ul, #{cid} .dgcv-panel-list ol {{ 
                    margin: 0.25rem 0 0 1.25rem; 
                }}
                #{cid} .dgcv-inline {{ 
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 0.5rem; 
                    list-style: none; 
                    padding: 0; 
                    margin-top: 0.5rem; 
                }}
                #{cid} .dgcv-chips {{ 
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 0.4rem; 
                    list-style: none; 
                    padding: 0; 
                    margin-top: 0.5rem; 
                }}
                #{cid} .dgcv-chip {{ 
                    padding: 0.2rem 0.5rem; 
                    border-radius: 999px; 
                    background: var(--dgcv-bg-hover); 
                    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-alt); 
                    color: var(--dgcv-text-hover);
                    font-size: 0.9em; 
                }}
                {scoped_extra}
                </style>
            """).strip()

    def _header_html(self) -> str:
        t = self._coerce_block(self.header)
        return f'<div class="dgcv-panel-head"><h3 class="dgcv-panel-title">{t}</h3></div><hr class="dgcv-panel-rule"/>'

    def _primary_html(self) -> str:
        if not self.primary_text:
            return ""
        return f'<div class="dgcv-panel-primary">{self._coerce_block(self.primary_text)}</div>'

    def _list_html(self) -> str:
        if not self.itemized_text:
            return ""
        items = [self._coerce_block(it) for it in self.itemized_text]
        if self.list_variant == "numbered":
            lis = "".join(f"<li>{i}</li>" for i in items)
            return f'<div class="dgcv-panel-list"><ol>{lis}</ol></div>'
        if self.list_variant == "inline":
            lis = "".join(f"<li>{i}</li>" for i in items)
            return (
                f'<div class="dgcv-panel-list"><ul class="dgcv-inline">{lis}</ul></div>'
            )
        if self.list_variant == "chips":
            lis = "".join(f'<li class="dgcv-chip">{i}</li>' for i in items)
            return (
                f'<div class="dgcv-panel-list"><ul class="dgcv-chips">{lis}</ul></div>'
            )
        lis = "".join(f"<li>{i}</li>" for i in items)
        return f'<div class="dgcv-panel-list"><ul>{lis}</ul></div>'

    def _footer_html(self) -> str:
        if not self.footer:
            return ""
        return f'<div class="dgcv-panel-footer">{self._coerce_block(self.footer)}</div>'

    def to_html(self, *args, **kwargs) -> str:
        layout_css = self._layout_css()
        head = self._header_html()
        body = f'<div class="dgcv-panel-body">{self._primary_html()}{self._list_html()}</div>'
        foot = self._footer_html()
        return f'<div id="{self.container_id}">{layout_css}<aside class="dgcv-panel">{head}{body}{foot}</aside></div>'

    def _repr_html_(self) -> str:
        return self.to_html()


# -----------------------------------------------------------------------------
# template builders
# -----------------------------------------------------------------------------


def build_plain_table(
    columns: List[str],
    rows: List[List[object]],
    *,
    caption: str = "",
    theme_css_vars: str = "",
    extra_css: str = "",
    slim: bool = False,
    **kwargs,
) -> TableView:
    return TableView(
        columns=columns,
        rows=rows,
        index_labels=None,
        caption=caption,
        theme_css_vars=theme_css_vars,
        extra_css=extra_css,
        slim=slim,
        **kwargs,
    )


def build_matrix_table(
    index_labels: List[object],
    columns: List[str],
    rows: List[List[object]],
    *,
    caption: str = "",
    theme_css_vars: str = "",
    extra_css: str = "",
    mirror_header_to_index: bool = True,
    dashed_corner: bool = True,
    header_underline_exclude_index: bool = True,
    slim: bool = False,
    **kwargs,
) -> TableView:
    matrix_specific_css = _matrix_extras(
        mirror_header_to_index=mirror_header_to_index,
        dashed_corner=dashed_corner,
        header_underline_exclude_index=header_underline_exclude_index,
    )

    combined_extra_css = matrix_specific_css + "\n" + extra_css

    return TableView(
        columns=columns,
        rows=rows,
        index_labels=index_labels,
        caption=caption,
        theme_css_vars=theme_css_vars,
        extra_css=combined_extra_css,
        slim=slim,
        **kwargs,
    )


def _scope_extra_css(css: str, cid: str) -> str:
    result = []
    i = 0
    css = css.strip()

    while i < len(css):
        while i < len(css) and css[i].isspace():
            i += 1
        if i >= len(css):
            break

        if css[i] == "@":
            j = css.index("{", i)
            at_header = css[i:j].strip()
            i = j + 1
            depth = 1
            body_start = i
            while i < len(css) and depth > 0:
                if css[i] == "{":
                    depth += 1
                elif css[i] == "}":
                    depth -= 1
                i += 1
            body = css[body_start : i - 1].strip()
            scoped_body = _scope_extra_css(body, cid)
            result.append(f"{at_header} {{\n{scoped_body}\n}}")

        else:
            j = css.index("{", i)
            selector = css[i:j].strip()
            i = j + 1
            k = css.index("}", i)
            declarations = css[i:k].strip()
            i = k + 1
            scoped = ", ".join(f"#{cid} {s.strip()}" for s in selector.split(","))
            result.append(f"{scoped} {{\n    {declarations}\n}}")

    return "\n".join(result)


def _outer_radius(r: str) -> str:
    if r == "0px" or r == "0":
        return r
    return f"var(--dgcv-border-radius, {r})"
