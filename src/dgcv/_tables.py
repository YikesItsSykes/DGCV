import html as _html
import re
import uuid
from html import escape as _esc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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

def _scoped_css(
    scope_id: str,
    base_css: str,
    *,
    column_align: Optional[Dict[int, str]] = None,
    has_index: bool = False,
    cell_align: Optional[str] = None,
    nowrap: bool = False,
) -> str:
    lines = []
    if base_css.strip():
        lines.append("\n".join(f"#{scope_id} {ln}" for ln in base_css.splitlines()))
    if cell_align in {"left", "center", "right"}:
        lines.append(f"#{scope_id} td {{ text-align: {cell_align}; }}")
        lines.append(f"#{scope_id} th {{ text-align: {cell_align}; }}")
    if nowrap:
        lines.append(f"#{scope_id} td {{ white-space: nowrap; }}")
        lines.append(f"#{scope_id} th {{ white-space: nowrap; }}")
    if column_align:
        offset = 1 if has_index else 0
        for col0, align in column_align.items():
            if align not in {"left", "center", "right"}:
                continue
            nth = col0 + offset  # 1-based index for nth-child
            lines.append(f"#{scope_id} thead th:nth-child({nth}) {{ text-align: {align}; }}")
            lines.append(f"#{scope_id} tbody td:nth-child({nth}) {{ text-align: {align}; }}")
    return "<style>\n" + "\n".join(lines) + "\n</style>"

def _parse_theme_border(theme_styles: List[CSSRule]):
    val = None
    sides = set()
    for sd in theme_styles:
        if sd.get("selector") == "table":
            for k, v in sd.get("props", []):
                if k in {"border-top","border-right","border-bottom","border-left"}:
                    sides.add(k)
                    if not val: 
                        val = v
                elif k == "border" and not val:
                    val = v
    if not val: 
        return ("1px","solid","#ccc", sides)
    parts = val.split()
    thickness = parts[0] if parts else "1px"
    color = parts[-1] if parts else "#ccc"
    return (thickness, "solid", color, sides)

def _matrix_extras(theme_styles: List[CSSRule], *, mirror_header_to_index: bool, dashed_corner: bool, header_underline_exclude_index: bool):
    t, _, color, side_keys = _parse_theme_border(theme_styles)
    solid = f"{t} solid {color}"
    dashed = f"{t} dashed {color}"

    extras: List[CSSRule] = []
    extras.append({"selector":"table","props":[("border-collapse","collapse")]})
    if not side_keys:
        extras[-1]["props"].append(("border", solid))

    if header_underline_exclude_index:
        extras.append({"selector": "thead th:not(:first-child)", "props": [("border-bottom", solid)]})
    else:
        extras.append({"selector": "thead th", "props": [("border-bottom", solid)]})

    extras.append({"selector": "tbody th", "props": [("border-right", solid)]})

    if mirror_header_to_index:
        col_head = []
        for sd in theme_styles:
            if sd.get("selector") == "th.col_heading.level0":
                col_head = sd.get("props", [])
                break
        row_visual = [(k,v) for (k,v) in col_head if not k.startswith("border")]
        if row_visual:
            extras.append({"selector":"th.row_heading","props":row_visual})

    if dashed_corner:
        extras.append({"selector":"thead th:first-child","props":[("border-right", dashed),("border-bottom", dashed)]})

    return extras

class TableView:
    def __init__(
        self,
        columns: List[str],
        rows: List[List[object]],
        *,
        index_labels: Optional[List[object]] = None,
        caption: str = "",
        preface_html: Optional[str] = None,
        theme_styles: Optional[List[CSSRule]] = None,
        extra_styles: Optional[List[CSSRule]] = None,
        table_attrs: str = 'style=" table-layout:fixed; overflow-x:auto;"',
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
        table_scroll = False,
        cell_scroll = False,
        show_headers: bool = True
    ):
        self.columns = columns
        self.rows = rows
        self.footer_rows = footer_rows or []
        self.index_labels = index_labels
        self.caption = caption
        self.theme_styles = theme_styles or []
        self.extra_styles = extra_styles or []
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
        self.side_width = f"{side_width}px" if isinstance(side_width, int) else str(side_width)
        self.breakpoint_px = breakpoint_px
        self.container_id = container_id or f"dgcv-view-{uuid.uuid4().hex[:8]}"
        self.preface_html = preface_html
        self.ul = f"{ul}px" if isinstance(ul, int) else str(ul)
        self.ur = f"{ur}px" if isinstance(ur, int) else str(ur)
        self.lr = f"{lr}px" if isinstance(lr, int) else str(lr)
        self.ll = f"{ll}px" if isinstance(ll, int) else str(ll)
        self.table_scroll = table_scroll
        self.cell_scroll = cell_scroll
        self.show_headers = show_headers

        if column_align:
            name_to_idx = {name: i for i, name in enumerate(columns)}
            _norm: Dict[int, str] = {}
            for k, v in column_align.items():
                if isinstance(k, int) and 0 <= k < len(columns):
                    _norm[k + 1] = v
                elif isinstance(k, str) and k in name_to_idx:
                    _norm[name_to_idx[k] + 1] = v
            self._column_align_idx = _norm
        else:
            self._column_align_idx = None

    def _render_cell(self, cell: object, *, tag: str = "td") -> str:
        """
        Accepts:
          - plain values (str/num/etc.) -> coerced via _coerce_html
          - dict cells: {"html": "...", "attrs": {"colspan": 3, ...}, "tag": "td"|"th"}
        """
        attrs = {}
        use_tag = tag
        if isinstance(cell, dict):
            html_raw = cell.get("html", "")
            attrs = cell.get("attrs", {}) or {}
            use_tag = cell.get("tag", use_tag)
            html = _coerce_html(html_raw, html_safe=not self.escape_cells)
        else:
            html = _coerce_html(cell, html_safe=not self.escape_cells)

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
        for c in self.columns:
            text = _coerce_html(c, html_safe=not self.escape_headers)
            cols.append(f'<th scope="col" class="col_heading level0">{text}</th>')
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
            for cell in row:
                tds.append(self._render_cell(cell, tag="td"))  
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
        return f"<caption>{_coerce_html(self.caption, html_safe=False)}</caption>" if self.caption else ""

    def _panel_html(self) -> Optional[str]:
        if self.secondary_panel_html is None:
            return None
        html = self.secondary_panel_html() if callable(self.secondary_panel_html) else self.secondary_panel_html
        return html or ""

    def _layout_css(self) -> str:
        cid = self.container_id
        direction = "row" if self.layout == "row" else "column"
        r_tl, r_tr, r_br, r_bl = self.ul, self.ur, self.lr, self.ll
        if not self.footer_rows:
            s_br, s_bl = r_br, r_bl
        else:
            s_br, s_bl = 0, 0
        gap = int(self.gap_px)

        if self.table_scroll:
            additional_str1 = f"""
            #{cid} .dgcv-data-table {{
            width: max-content;
            min-width: 100%;
            table-layout: fixed;
            border-collapse: separate;
            border-spacing: 0;
            }}
            #{cid} .dgcv-table-wrap {{
            overflow-x: auto;
            max-width: 100%;
            }}
            """.strip()
        else:
            additional_str1 = f"""
            #{cid} .dgcv-data-table {{
            width: auto;
            table-layout: fixed;
            border-collapse: separate;
            border-spacing: 0;
            }}
            #{cid} .dgcv-table-wrap {{
            overflow-x: auto;
            max-width: 100%;
            }}
            """.strip()

        additional_str2 = f"""
            #{cid} .dgcv-data-table td .table-cell {{
            overflow-x: {'auto' if self.cell_scroll else 'visible'};
            white-space: {'nowrap' if self.cell_scroll else 'normal'};
            }}
            """.strip()

        if getattr(self, "show_headers", True):
            tl_left_sel  = "thead tr:first-child th:first-child"
            tl_right_sel = "thead tr:first-child th:last-child"
            flex_tr_right_sel = "thead tr:first-child th:last-child"
            media_tl_left_sel  = "thead tr:first-child th:first-child"
            media_tl_right_sel = "thead tr:first-child th:last-child"
        else:
            has_index = self.index_labels is not None
            tl_left_sel  = "tbody tr:first-child th.row_heading" if has_index else "tbody tr:first-child td:first-child"
            tl_right_sel = "tbody tr:first-child td:last-child"
            flex_tr_right_sel = "tbody tr:first-child td:last-child"
            media_tl_left_sel  = tl_left_sel
            media_tl_right_sel = tl_right_sel

        return f"""
<style>
#{cid} .dgcv-flex {{
  display: flex;
  flex-direction: {direction};
  gap: {gap}px;
  align-items: flex-start;
  justify-content: flex-start;
  flex-wrap: wrap;
}}
#{cid} .dgcv-main {{ flex: 0 1 auto; max-width: calc(60% - {gap}px); min-width: 0; margin: 0;}}
#{cid} .dgcv-side {{ flex: 0 0 {self.side_width}; max-width: 40%; box-sizing: border-box; overflow-y: visible; margin: 0;}}
{additional_str1}
{additional_str2}

#{cid} .dgcv-data-table,
#{cid} .dgcv-data-table thead,
#{cid} .dgcv-data-table tfoot {{ background-color: inherit; }}

#{cid} .dgcv-data-table th,
#{cid} .dgcv-data-table td {{ background-clip: padding-box; }}

#{cid} .dgcv-data-table {tl_left_sel}  {{ border-top-left-radius: {r_tl}; }}
#{cid} .dgcv-data-table {tl_right_sel} {{ border-top-right-radius: {r_tr}; }}
#{cid} .dgcv-data-table tfoot tr:last-child td:first-child    {{ border-bottom-left-radius: {r_bl}; }}
#{cid} .dgcv-data-table tbody tr:last-child td:first-child    {{ border-bottom-left-radius: {s_bl}; }}
#{cid} .dgcv-data-table tbody tr:last-child th.row_heading    {{ border-bottom-left-radius: {s_bl}; }}
#{cid} .dgcv-data-table tfoot tr:last-child td:last-child     {{ border-bottom-right-radius: {r_br}; }}
#{cid} .dgcv-data-table tbody tr:last-child td:last-child     {{ border-bottom-right-radius: {s_br}; }}

#{cid} .dgcv-data-table {{ border-radius: {r_tl} {r_tr} {r_br} {r_bl}; overflow: hidden; }}

#{cid} .dgcv-flex .dgcv-data-table {flex_tr_right_sel} {{ border-top-right-radius: 0; }}
#{cid} .dgcv-flex .dgcv-data-table tfoot tr:last-child td:last-child,
#{cid} .dgcv-flex .dgcv-data-table tbody tr:last-child td:last-child   {{ border-bottom-right-radius: 0; }}
#{cid} .dgcv-flex .dgcv-data-table tfoot tr:last-child td:first-child  {{ border-bottom-left-radius: {r_bl}; }}
#{cid} .dgcv-flex .dgcv-data-table tbody tr:last-child td:first-child  {{ border-bottom-left-radius: {s_bl}; }}
#{cid} .dgcv-flex .dgcv-data-table tbody tr:last-child th.row_heading  {{ border-bottom-left-radius: {s_bl}; }}
#{cid} .dgcv-flex .dgcv-data-table {{ border-radius: {r_tl} 0 0 {r_bl}; overflow: hidden; }}
#{cid} .dgcv-flex .dgcv-side-panel {{ border-radius: 0 {r_tr} {r_br} 0; }}

@media (max-width: {int(self.breakpoint_px)}px) {{
#{cid} .dgcv-flex {{
  flex-direction: column;
  display: block;
  align-items: stretch;
}}
#{cid} .dgcv-main {{max-width: 100%;}}
#{cid} .dgcv-main, #{cid} .dgcv-side, #{cid} .dgcv-table-wrap, #{cid} .dgcv-side-panel {{
  display: block;
  width: 100%;
  box-sizing: border-box;
  margin-bottom: {gap}px;
}}
#{cid} .dgcv-table-wrap {{ margin: 0;}}
#{cid} .dgcv-side {{
  max-width: 100%;
  flex: 1;
  overflow-y: visible;
}}

#{cid} .dgcv-data-table {{ width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 0; }}
#{cid} .dgcv-flex .dgcv-data-table {media_tl_left_sel}  {{ border-top-left-radius: {r_tl}; }}
#{cid} .dgcv-flex .dgcv-data-table {media_tl_right_sel} {{ border-top-right-radius: {r_tr}; }}
#{cid} .dgcv-flex .dgcv-data-table tfoot tr:last-child td:first-child,
#{cid} .dgcv-flex .dgcv-data-table tbody tr:last-child td:first-child {{ border-bottom-left-radius: 0; }}
#{cid} .dgcv-flex .dgcv-data-table tbody tr:last-child th.row_heading {{ border-bottom-left-radius: 0; }}
#{cid} .dgcv-flex .dgcv-data-table tfoot tr:last-child td:last-child,
#{cid} .dgcv-flex .dgcv-data-table tbody tr:last-child td:last-child {{ border-bottom-right-radius: 0; }}

#{cid} .dgcv-data-table,
#{cid} .dgcv-data-table thead,
#{cid} .dgcv-data-table tfoot {{ background-color: inherit; }}

#{cid} .dgcv-data-table th,
#{cid} .dgcv-data-table td {{ background-clip: padding-box; }}

#{cid} .dgcv-flex .dgcv-data-table {{ border-radius: {r_tl} {r_tr} 0 0; overflow: hidden; }}
#{cid} .dgcv-flex .dgcv-side-panel {{ border-radius: 0 0 {r_br} {r_bl}; }}
}}
</style>
""".strip()

    def _table_css(self) -> str:
        base = styles_to_css(merge_styles(self.theme_styles, self.extra_styles))
        scoped = _scoped_css(
            self.container_id,
            base,
            column_align=self._column_align_idx,
            has_index=self.index_labels is not None,
            cell_align=self.cell_align,
            nowrap=self.nowrap,
        )
        return scoped

    def _table_html_only(self) -> str:
        cap = self._caption_html()
        thead = self._thead_html()
        tbody = self._tbody_html()
        tfoot = self._tfoot_html()
        css = self._table_css()
        return (
            f"{css}"
            f"<table class=\"dgcv-data-table\" {self.table_attrs}>"
            f"{cap}"
            f"{thead}"
            f"{tbody}"
            f"{tfoot}"
            f"</table>"
        )

    def to_html(self, *args, **kwargs) -> str:
        panel = self._panel_html()
        table_html = self._table_html_only()
        preface = self.preface_html or ""
        layout_css = self._layout_css()
        if not panel:
            return f'<div id="{self.container_id}">{layout_css}{preface}<div class="dgcv-table-wrap">{table_html}</div></div>'

        return (
            f'<div id="{self.container_id}">'
            f"{layout_css}"
            f"{preface}"
            f'<div class="dgcv-flex">'
            f'  <div class="dgcv-main"><div class="dgcv-table-wrap">{table_html}</div></div>'
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
            cells = [_strip_tags_simple(_coerce_html(c, html_safe=not self.escape_cells)) if not isinstance(c, dict)
                     else _strip_tags_simple(_coerce_html(c.get("html", ""), html_safe=not self.escape_cells))
                     for c in row]
            if self.truncate_chars is not None:
                cells = [c if len(c) <= self.truncate_chars else _strip_tags_simple(self.truncate_msg) for c in cells]
            if self.index_labels is not None:
                idx = _strip_tags_simple(_coerce_html(self.index_labels[i], html_safe=not self.escape_index))
                lines.append(col_sep.join([idx] + cells))
            else:
                lines.append(col_sep.join(cells))
        # Footer rows in text mode (simple): render as extra lines
        for frow in self.footer_rows:
            cells = [_strip_tags_simple(_coerce_html(c, html_safe=not self.escape_cells)) if not isinstance(c, dict)
                     else _strip_tags_simple(_coerce_html(c.get("html", ""), html_safe=not self.escape_cells))
                     for c in frow]
            lines.append(col_sep.join(cells))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()

    def to_plain_text(self, col_sep: str = " | ") -> str:
        return self.to_text(col_sep=col_sep)

def _sanitize_html_str(s: str) -> str:
    import re
    s = re.sub(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"\s+on[a-zA-Z]+\s*=\s*(['\"]).*?\1", "", s, flags=re.IGNORECASE | re.DOTALL)
    return s

class panel_view:
    def __init__(
        self,
        *,
        header: Union[str, Any],
        primary_text: Optional[Union[str, Any]] = None,
        itemized_text: Optional[Union[List[Union[str, Any]], tuple]] = None,
        footer: Optional[Union[str, Any]] = None,
        theme_styles: Optional[List["CSSRule"]] = None,
        extra_styles: Optional[List["CSSRule"]] = None,
        list_variant: str = "bulleted",
        use_latex: bool = False,
        sanitize: bool = True,
        container_id: Optional[str] = None,
        ul: Union[int, str] = 10,
        ur: Union[int, str] = 10,
        lr: Union[int, str] = 10,
        ll: Union[int, str] = 10,
    ):
        self.header = header
        self.primary_text = primary_text
        self.itemized_text = list(itemized_text) if itemized_text else []
        self.footer = footer
        self.theme_styles = theme_styles or []
        self.extra_styles = extra_styles or []
        self.list_variant = list_variant
        self.use_latex = use_latex
        self.sanitize = sanitize
        self.container_id = container_id or f"dgcv-panel-{uuid.uuid4().hex[:8]}"
        self.ul = f"{ul}px" if isinstance(ul, int) else str(ul)
        self.ur = f"{ur}px" if isinstance(ur, int) else str(ur)
        self.lr = f"{lr}px" if isinstance(lr, int) else str(lr)
        self.ll = f"{ll}px" if isinstance(ll, int) else str(ll)

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

    def _panel_css(self) -> str:
        base = styles_to_css(merge_styles(self.theme_styles, self.extra_styles))
        scoped = _scoped_css(self.container_id, base)
        return scoped

    def _layout_css(self) -> str:
        cid = self.container_id
        r_tl, r_tr, r_br, r_bl = self.ul, self.ur, self.lr, self.ll
        return f"""
<style>
#{cid} .dgcv-panel {{
  border-radius: {r_tl} {r_tr} {r_br} {r_bl};
  background-color: var(--bg-surface, transparent);
  border: 1px solid var(--border-color, #ddd);
  color: var(--text-title, inherit);
  overflow: hidden;
}}
#{cid} .dgcv-panel-head {{ margin: 0; padding: 0.75rem 1rem; }}
#{cid} .dgcv-panel-title {{ margin: 0; font-size: 1rem; line-height: 1.3; font-weight: 600; color: var(--text-title, inherit); }}
#{cid} .dgcv-panel-rule {{ border: 0; height: 1px; background: var(--border-color, #ddd); margin: 0; }}
#{cid} .dgcv-panel-body {{ padding: 0.75rem 1rem; }}
#{cid} .dgcv-panel-footer {{ padding: 0.5rem 1rem; background: var(--bg-muted, transparent); border-top: 1px solid var(--border-color, #ddd); }}
#{cid} .dgcv-panel-list {{ margin: 0.5rem 0 0; padding: 0; }}
#{cid} .dgcv-panel-list ul, #{cid} .dgcv-panel-list ol {{ margin: 0.25rem 0 0 1.25rem; }}
#{cid} .dgcv-inline {{ display: flex; flex-wrap: wrap; gap: 0.5rem; list-style: none; padding: 0; margin-top: 0.5rem; }}
#{cid} .dgcv-chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; list-style: none; padding: 0; margin-top: 0.5rem; }}
#{cid} .dgcv-chip {{ padding: 0.2rem 0.5rem; border-radius: 999px; background: var(--hover-bg, rgba(0,0,0,0.05)); border: 1px solid var(--border-color, #ddd); font-size: 0.9em; }}
</style>
""".strip()

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
            return f'<div class="dgcv-panel-list"><ul class="dgcv-inline">{lis}</ul></div>'
        if self.list_variant == "chips":
            lis = "".join(f'<li class="dgcv-chip">{i}</li>' for i in items)
            return f'<div class="dgcv-panel-list"><ul class="dgcv-chips">{lis}</ul></div>'
        lis = "".join(f"<li>{i}</li>" for i in items)
        return f'<div class="dgcv-panel-list"><ul>{lis}</ul></div>'

    def _footer_html(self) -> str:
        if not self.footer:
            return ""
        return f'<div class="dgcv-panel-footer">{self._coerce_block(self.footer)}</div>'

    def to_html(self, *args, **kwargs) -> str:
        theme_css = self._panel_css()
        layout_css = self._layout_css()
        head = self._header_html()
        body = f'<div class="dgcv-panel-body">{self._primary_html()}{self._list_html()}</div>'
        foot = self._footer_html()
        return f'<div id="{self.container_id}">{layout_css}{theme_css}<aside class="dgcv-panel">{head}{body}{foot}</aside></div>'

    def _repr_html_(self) -> str:
        return self.to_html()

# template builders
def build_plain_table(
    columns: List[str],
    rows: List[List[object]],
    *,
    caption: str = "",
    theme_styles: Optional[List[CSSRule]] = None,
    extra_styles: Optional[List[CSSRule]] = None,
    **kwargs,
) -> TableView:
    return TableView(
        columns=columns,
        rows=rows,
        index_labels=None,
        caption=caption,
        theme_styles=theme_styles,
        extra_styles=extra_styles,
        **kwargs,
    )

def build_matrix_table(
    index_labels: List[object],
    columns: List[str],
    rows: List[List[object]],
    *,
    caption: str = "",
    theme_styles: Optional[List[CSSRule]] = None,
    extra_styles: Optional[List[CSSRule]] = None,
    mirror_header_to_index: bool = True,
    dashed_corner: bool = True,
    header_underline_exclude_index: bool = True,
    **kwargs,
) -> TableView:
    theme_styles = theme_styles or []
    extras = list(extra_styles or [])
    extras = (_matrix_extras(
        theme_styles,
        mirror_header_to_index=mirror_header_to_index,
        dashed_corner=dashed_corner,
        header_underline_exclude_index=header_underline_exclude_index,
    ) + extras)
    return TableView(
        columns=columns,
        rows=rows,
        index_labels=index_labels,
        caption=caption,
        theme_styles=theme_styles,
        extra_styles=extras,
        **kwargs,
    )
