"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.branching_analysis

module: algebras.branching_analysis.branching_analysis


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

SPDX-License-Identifier: Apache-2.0


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------

import textwrap
import uuid

from .._aux._backends._symbolic_router import get_free_symbols, simplify, subs
from .._aux._utilities._config import dgcv_warning, latex_in_html
from .._aux.printing.printing._dgcv_display import LaTeX_eqn_system, LaTeX_list, show

__all__ = ["case_tree"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
class case_tree:
    """
    Directory structure for organizing branching case analysis of an equation system.

    Parameters:
    -----------
    label: str (optional, default='root')
        string label of node on the case tree
    equation_system: iterable (optional, default = [])
        An iterable containing objects that dgcv can compare with zero
    parameters: list|tuple|set (optional, default = [])
        list of variables in the equation system that can be constrained in the case tree
    completion_condition: callable (optional, default compares elements in case-rules-reduced eqn system to 0),
        This should be a callable function that can be applied to the self.reduced_equation_system returning a bool indicating if the eqn system is satisfied
    evaluate_with_simplifies: bool (optional, default = True)
        Determines if internal simplify calls are usesd or omitted. Can affect compute times and completion_condition accuracy
    completion_message: str|callable (optional)
        This messages is displayed in tree summaries at complete state nodes. If str, then the str literal is displayed. if callable, then the callable(self) is displayed, so callable(self) should return a str
    summary: callable (optional)
        If callable the default self.case_summary will be replaced by `summary(self)`
    """

    def __init__(
        self,
        label: str = None,
        equation_system=None,
        parameters=None,
        completion_condition=None,
        evaluate_with_simplifies=True,
        **kwargs,
    ):
        self.general_equation_system = (
            [] if equation_system is None else equation_system
        )
        if "_simplify_rule" in kwargs:
            s = kwargs["_simplify_rule"]
        elif evaluate_with_simplifies is True:

            def s(x):
                return simplify(x)
        else:

            def s(x):
                return x

        self._internal_simplify = s

        if parameters is None:
            try:
                params = set()
                for eqn in self.general_equation_system:
                    params |= get_free_symbols(eqn)
                self.system_parameters = params
            except Exception:
                self.system_parameters = set()
        else:
            self.system_parameters = set(parameters)
        if not callable(completion_condition):

            def _cd(obj):
                try:

                    def ztest(x):
                        sx = self._internal_simplify(x)
                        return getattr(sx, "is_zero", False) or sx == 0

                    return all(ztest(x) for x in obj)
                except Exception:
                    return False

            self.completion_condition = _cd
        else:
            self.completion_condition = completion_condition
        self.label = label if label else "root"
        self._subcases = {}
        for k, v in kwargs.items():
            setattr(self, k, v)
        if "case_rules" not in kwargs:
            self.case_rules = {}
        if "_new_case_rules" not in kwargs:
            self._new_case_rules = self.case_rules
        elif not isinstance(kwargs["case_rules"], dict):
            dgcv_warning(
                "case_tree recieved `case_rules` in an unsuported format, so `case_rules` was ignored.",
                UserWarning,
                stacklevel=2,
            )
            self.case_rules = {}

        self._complete = None
        self._ev_eqn_system = None
        self._free_variables = None

    @property
    def _completion_message(self):
        if hasattr(self, "completion_message"):
            f = self.completion_message
            if callable(f):
                return str(f(self))
            else:
                return str(f)
        else:
            return (
                f"number of free variables = {len(getattr(self, 'free_variables', {}))}"
            )

    @property
    def complete(self):
        if self._complete is None:
            self._complete = self.completion_condition(self.reduced_equation_system)
        return self._complete

    def add_case(self, label, **kwargs):
        if not isinstance(label, str):
            raise TypeError("The `label` parameter must be a string.")
        if "case_rules" in kwargs:
            new_cr = kwargs["case_rules"]
            cr = {
                k: subs(v, new_cr) for k, v in getattr(self, "case_rules", {}).items()
            } | new_cr
            kwargs["case_rules"] = cr
            kwargs["_new_case_rules"] = new_cr
            new_tree = case_tree(
                label=label,
                equation_system=self.general_equation_system,
                parameters=self.system_parameters,
                completion_condition=self.completion_condition,
                **{"_simplify_rule": self._internal_simplify, **kwargs},
            )
        setattr(self, label, new_tree)
        self._subcases[label] = getattr(self, label)

    def _repr_latex_(self, raw=False, **kwargs):
        out = LaTeX_eqn_system(
            getattr(self, "case_rules", {}),
            **{"one_line": True, "punctuation": ".", **kwargs},
        )
        if raw is True:
            from .._aux.printing.printing._string_processing import _strip_displaystyles

            out = _strip_displaystyles(out)
        return out

    def __str__(self):
        return getattr(self, "inheritance_path", "") + self.label

    def show_case_rules(self, **kwargs):
        show(self._repr_latex_(**kwargs))

    @property
    def reduced_equation_system(self):
        if self._ev_eqn_system is None:
            if hasattr(self.general_equation_system, "subs"):
                self._ev_eqn_system = self._internal_simplify(
                    subs(self.general_equation_system, self.case_rules)
                )
            else:
                self._ev_eqn_system = [
                    self._internal_simplify(
                        subs(x, self.case_rules) for x in self.general_equation_system
                    )
                ]
        return self._ev_eqn_system

    @property
    def free_variables(self):
        if self._free_variables is None:
            scr = getattr(self, "case_rules", {})
            fvd = {v: subs(v, scr) for v in self.system_parameters}
            fv = set()
            for k, v in fvd.items():
                fv |= get_free_symbols(v)
            self._free_variables = [v for v in fv if v in self.system_parameters]
        return self._free_variables

    @property
    def _tree(self):
        tree_dict = {self.label: {}}
        if self._subcases:
            for sc, subtree in self._subcases.items():
                tree_dict[self.label] |= subtree._tree
        else:
            if self.complete:
                addon = (
                    " - " + self._completion_message if self._completion_message else ""
                )
                tree_dict[self.label] = f"complete{addon}"
            else:
                tree_dict[self.label] = "incomplete"
        return tree_dict

    @property
    def _verbose_tree(self):
        branch_conditions = getattr(self, "_new_case_rules", {})
        branch_conditions = (
            [f"{k}={v}" for k, v in branch_conditions.items()]
            if isinstance(branch_conditions, dict)
            else []
        )
        tree_dict = {
            self.label: {"branch_conditions": branch_conditions, "descendants": {}}
        }
        if self._subcases:
            for sc, subtree in self._subcases.items():
                tree_dict[self.label]["descendants"] |= subtree._verbose_tree
        else:
            if self.complete:
                addon = (
                    " - " + self._completion_message if self._completion_message else ""
                )
                tree_dict[self.label]["descendants"] = f"complete{addon}"
            else:
                tree_dict[self.label]["descendants"] = "incomplete"
        return tree_dict

    @property
    def _latex_verbose_tree(self):
        branch_conditions = getattr(self, "_new_case_rules", {})
        branch_conditions = (
            [
                f"{LaTeX_eqn_system({k: v}, math_mode='$')}"
                for k, v in branch_conditions.items()
            ]
            if isinstance(branch_conditions, dict)
            else []
        )

        tree_dict = {
            self.label: {"branch_conditions": branch_conditions, "descendants": {}}
        }
        if hasattr(self, "note"):
            tree_dict[self.label]["note"] = str(self.note)
        if self._subcases:
            for sc, subtree in self._subcases.items():
                tree_dict[self.label]["descendants"] |= subtree._latex_verbose_tree
        else:
            if self.complete:
                addon = (
                    " - " + self._completion_message if self._completion_message else ""
                )
                tree_dict[self.label]["descendants"] = f"complete{addon}"
            else:
                tree_dict[self.label]["descendants"] = "incomplete"
        return tree_dict

    def tree_summary(self, theme=None):
        return latex_in_html(_full_tree_html(self._latex_verbose_tree, theme=theme))

    @classmethod
    def _print_path_tree(cls, data, indent="", path="", root=True):
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            clean_key = str(key).strip("_")
            if root:
                display_label = clean_key
                connector = ""
                line_indent = ""
                child_path = ""
            else:
                display_label = f"{path}.{clean_key}" if path else clean_key
                connector = "└── " if is_last else "├── "
                line_indent = indent
                child_path = display_label

            if isinstance(value, dict):
                print(f"{line_indent}{connector}{display_label}")
                next_indent = (
                    indent + ("    " if is_last else "│   ") if not root else ""
                )
                cls._print_path_tree(value, next_indent, child_path, root=False)
            else:
                print(f"{line_indent}{connector}{display_label}: {value}")

    @classmethod
    def _print_verbose_tree(cls, data, indent="", path="", root=True):

        if root and "root" in data:
            print("root")
            cls._print_verbose_tree(data["root"], indent, path="", root=False)
            return

        if isinstance(data, dict) and "descendants" in data:
            items = (
                list(data["descendants"].items())
                if isinstance(data["descendants"], dict)
                else []
            )
        else:
            return

        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            clean_key = str(key).strip("_")
            display_label = f"{path}.{clean_key}" if path else clean_key

            conds = value.get("branch_conditions", [])
            if isinstance(conds, str):
                conds = [conds.strip("[]' ")]

            raw_text = ", ".join(conds)
            chunks = textwrap.wrap(raw_text, width=24, break_long_words=True)
            max_w = max(len(c) for c in chunks) if chunks else 0

            prefix = "└── " if is_last else "├── "
            vertical_gate = "    " if is_last else "│   "

            descendants = value.get("descendants")
            is_dict = isinstance(descendants, dict)
            suffix = "" if is_dict else f": {descendants}"

            for idx, text in enumerate(chunks):
                if idx == 0:
                    b_open = "⎡" if len(chunks) > 1 else "["
                    b_close = "⎤" if len(chunks) > 1 else "]"
                    print(
                        f"{indent}{prefix}{b_open}{text.ljust(max_w)}{b_close}─{display_label}{suffix}"
                    )
                else:
                    b_mid = " ⎢" if idx < len(chunks) - 1 else " ⎣"
                    b_end = "⎥" if idx < len(chunks) - 1 else "⎦"

                    # If node B has children, add a strut '│' aligned under the label
                    # The label starts after prefix(4) + brackets(2) + max_w + 1(dash)
                    strut = ""
                    if is_dict:
                        # Space to get past the box width, then the vertical line
                        strut = " " * (len(display_label) // 2) + "│"

                    print(
                        f"{indent}{vertical_gate[:3]}{b_mid}{text.ljust(max_w)}{b_end}{strut}"
                    )

            if is_dict:
                # Align next level children under the strut we just created
                padding_width = (max_w + 3) if chunks else 0
                next_indent = indent + vertical_gate + (" " * padding_width)
                cls._print_verbose_tree(value, next_indent, display_label, root=False)

    def case_summary(self, punctuation="."):
        f = getattr(self, "summary", None)
        if callable(f):
            return f(self)
        print("Conditions")
        self.show_case_rules(punctuation=",")
        print("imply")
        show(
            LaTeX_eqn_system(
                {"0=[A,B]": self.general_equation_system}, punctuation=punctuation
            )
        )
        print("The remaining free parameters in this branch are")
        show(LaTeX_list(self.free_variables, one_line=True, punctuation="."))
        if self.complete:
            print("********** The branch is complete! **********")
            print(f"     {len(self.free_variables)} dim. parameter space remaining ")


def _html_style(theme=None, container_id=None):
    if not isinstance(theme, str):
        from .._aux._utilities._config import get_dgcv_settings_registry

        theme = get_dgcv_settings_registry().get("theme", "dark")

    from .._aux._utilities._styles import get_style

    theme_vars = get_style(theme, legacy=False)

    scope = f"#{container_id}" if container_id else ""
    scoped_vars = theme_vars.replace(":root", scope) if scope else theme_vars

    base_styles = f"""
{scoped_vars}

{scope}.tree-container {{ 
    padding: 20px; 
    overflow-x: auto; 
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
    max-width: 450px;
    background-color: transparent !important; 
}}

{scope} .compound-node:hover {{
    transform: var(--dgcv-hover-transform, none);
}}

{scope} .node-label, {scope} .cond-box, {scope} .complete-msg, {scope} .note-msg {{
    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    box-shadow: var(--dgcv-table-shadow, none);
    border-image: var(--dgcv-border-image, none);
}}

{scope} .node-label, {scope} .cond-box, {scope} .complete-msg {{
    display: table-cell;
    padding: 8px 15px;
    vertical-align: middle;
    white-space: normal;
    word-wrap: break-word;
}}

{scope} .node-label {{
    font-weight: bold; 
    font-size: 14px; 
    background: var(--dgcv-table-background, var(--dgcv-bg-primary));
    color: var(--dgcv-text-main);
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
    border-radius: 0 var(--dgcv-border-radius, 12px) 0 0;
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
    padding: 4px 12px; 
    font-size: 10px; 
    background-color: var(--dgcv-bg-surface);
    color: var(--dgcv-text-heading);
    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
    border-top: none;
    border-radius: 0 0 var(--dgcv-border-radius, 12px) 0;
    white-space: normal;
    word-wrap: break-word;
}}

{scope} .compound-node:not(:has(.complete-msg)) .note-msg {{
    border-bottom-right-radius: 0;
}}

{scope} .compound-node:hover .node-label,
{scope} .compound-node:hover .cond-box, 
{scope} .compound-node:hover .complete-msg, 
{scope} .compound-node:hover .note-msg {{ 
    background-color: var(--dgcv-bg-hover) !important; 
    color: var(--dgcv-text-hover) !important; 
    font-weight: var(--dgcv-hover-font-weight, inherit);
}}

{scope} .children-ul {{ margin-left: 10px; }}
    """
    return f"<style>{base_styles}</style>"


def _to_html_tree(data, path="", is_root=True, container_id=None):
    if not isinstance(data, dict):
        return ""
    import html

    if is_root and container_id is None:
        container_id = f"tree-{uuid.uuid4().hex[:8]}"
    res = (
        f'<div id="{container_id}" class="tree-container"><ul>'
        if is_root
        else '<ul class="children-ul">'
    )
    if is_root and "root" in data:
        res += '<li><div class="compound-node root-wrapper"><div class="node-label">root</div></div>'
        res += _to_html_tree(data["root"], "", False, container_id=container_id)
        res += "</li>"
    else:
        items = list(data.get("descendants", {}).items())
        for key, value in items:
            clean_key = str(key).strip("_")
            current_path = f"{path}.{clean_key}" if path else clean_key
            conds = (
                value.get("branch_conditions", []) if isinstance(value, dict) else []
            )
            if isinstance(conds, str):
                conds = [conds.strip("[]'")]
            descendants = value.get("descendants") if isinstance(value, dict) else value
            is_dict = isinstance(descendants, dict)
            res += '<li><div class="compound-node">'
            res += f'<div class="node-label">{html.escape(current_path)}</div>'
            cond_text = (
                ",\u00a0  ".join(html.escape(str(c)) for c in conds)
                if any(conds)
                else "None"
            )
            res += f'<div class="cond-box">{cond_text}</div>'
            if not is_dict:
                res += (
                    f'<div class="complete-msg">{html.escape(str(descendants))}</div>'
                )
            note = value.get("note") if isinstance(value, dict) else None
            if note is not None:
                res += f'<div class="note-msg">{html.escape(str(note))}</div>'
            res += "</div>"
            if is_dict:
                res += _to_html_tree(
                    value, current_path, False, container_id=container_id
                )
            res += "</li>"
    res += "</ul>"
    return (res + "</div>") if is_root else res


def _full_tree_html(data, theme=None):
    cid = f"tree-{uuid.uuid4().hex[:8]}"
    styles = _html_style(theme=theme, container_id=cid)
    tree = _to_html_tree(data, is_root=True, container_id=cid)
    return styles + tree
