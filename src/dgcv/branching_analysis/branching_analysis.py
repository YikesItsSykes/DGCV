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
from .._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
    latex_in_html,
)
from .._aux._utilities._styles import get_style
from .._aux._vmf._safeguards import check_dgcv_category
from .._aux.printing._tables import build_plain_table
from .._aux.printing.printing._dgcv_display import (
    LaTeX_eqn_system,
    LaTeX_list,
    show,
)

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
        variables=None,
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

        if variables is None:
            try:
                vari = set()
                for eqn in self.general_equation_system:
                    vari |= get_free_symbols(eqn)
                self.system_variables = vari
            except Exception:
                self.system_variables = set()
        else:
            self.system_variables = set(variables)
        if parameters is None:
            try:
                params = set()
                for eqn in self.general_equation_system:
                    params |= set(
                        filter(lambda x: x not in variables, get_free_symbols(eqn))
                    )
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
        self._free_parameters = None

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

    def add_case(self, label: str = None, case_rules: dict = None, **kwargs):
        if label is None:
            for idx in range(1, len(self._subcases) + 2):
                pref = f"_{idx}"
                if any(str(x) == f"_{idx}" for x in self._subcases):
                    continue
                label = pref
                break

        if not isinstance(case_rules, dict):
            case_rules = {}
        verbose = kwargs.get("verbose", None)
        if not isinstance(verbose, bool):
            verbose = True is getattr(self, "verbose", False)
        if not isinstance(label, str):
            raise TypeError("The `label` parameter must be a string.")
        if label.isnumeric():
            raise ValueError(
                "Pure numeric labels for subcases are not supported. Recommendation: prepend the numeric label with an underscore."
            )
        if label in self._subcases:
            if verbose:
                dgcv_warning(
                    "Overwriting an existing subcase branch.",
                    wc_label="dgcvOperationsNote",
                )
        elif hasattr(self, label):
            raise ValueError(
                "subcases cannot be assigned names coinciding with the `case_tree` class' base attributes."
            )
        new_cr = case_rules
        cr = {
            k: subs(v, new_cr) for k, v in getattr(self, "case_rules", {}).items()
        } | new_cr
        kwargs["case_rules"] = cr
        kwargs["_new_case_rules"] = new_cr
        new_tree = case_tree(
            label=label,
            equation_system=self.general_equation_system,
            variables=self.system_variables,
            parameters=self.system_parameters,
            completion_condition=self.completion_condition,
            **{"_simplify_rule": self._internal_simplify, **kwargs},
        )
        setattr(self, label, new_tree)
        self._subcases[label] = getattr(self, label)

    def add_corollary(self, label: str, case_rules=None):
        return self.add_case(label=label, case_rules=case_rules, note=r"$\implies $")

    def remove_case(self, label):
        if label in self._subcases:
            _ = self._subcases.pop(label, None)
            if hasattr(self, label):
                delattr(self, label)

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

    def show_case_rules(self, plain_text=False, **kwargs):
        if plain_text is True:
            print(self.case_rules)
        else:
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
            fvd = {v: subs(v, scr) for v in self.system_variables}
            fv = set()
            for k, v in fvd.items():
                fv |= get_free_symbols(v)
            self._free_variables = [v for v in fv if v in self.system_variables]
        return self._free_variables

    @property
    def free_parameters(self):
        if self._free_parameters is None:
            scr = getattr(self, "case_rules", {})
            fvd = {v: subs(v, scr) for v in self.system_parameters}
            fv = set()
            for v in fvd.values():
                fv |= get_free_symbols(v)
            self._free_parameters = [v for v in fv if v in self.system_parameters]
        return self._free_parameters

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

    def _verbose_tree(self, filter_conditions=None):
        branch_conditions = getattr(self, "_new_case_rules", {})
        if filter_conditions is not None:
            try:
                branch_conditions = (
                    [
                        f"{filter_conditions(k)}={filter_conditions(v)}"
                        for k, v in branch_conditions.items()
                    ]
                    if isinstance(branch_conditions, dict)
                    else []
                )
            except Exception:
                dgcv_warning(
                    "The given value for `filter_conditions` is not in a supported format. It must be a callable function that transforms symbolic expressions (e.g., simplify/allToReal/expand/etc.)"
                )
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
                tree_dict[self.label]["descendants"] |= subtree._verbose_tree(
                    filter_conditions=filter_conditions
                )
        else:
            if self.complete:
                addon = (
                    " - " + self._completion_message if self._completion_message else ""
                )
                tree_dict[self.label]["descendants"] = f"complete{addon}"
            else:
                tree_dict[self.label]["descendants"] = "incomplete"
        return tree_dict

    def _latex_verbose_tree(self, filter_conditions=None):
        branch_conditions = getattr(self, "_new_case_rules", {})
        if filter_conditions is not None:
            try:
                branch_conditions = (
                    [
                        f"{LaTeX_eqn_system({filter_conditions(k): filter_conditions(v)}, math_mode='$')}"
                        for k, v in branch_conditions.items()
                    ]
                    if isinstance(branch_conditions, dict)
                    else []
                )
            except Exception:
                dgcv_warning(
                    "The given value for `filter_conditions` is not in a supported format. It must be a callable function that transforms symbolic expressions (e.g., simplify/allToReal/expand/etc.)"
                )
        else:
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
                tree_dict[self.label]["descendants"] |= subtree._latex_verbose_tree(
                    filter_conditions=filter_conditions
                )
        else:
            if self.complete:
                addon = (
                    " - " + self._completion_message if self._completion_message else ""
                )
                tree_dict[self.label]["descendants"] = f"complete{addon}"
            else:
                tree_dict[self.label]["descendants"] = "incomplete"
        return tree_dict

    def tree_summary(
        self,
        theme: str = None,
        root_label: str = None,
        filter_conditions=None,
        use_latex=None,
    ):
        use_latex = (
            use_latex
            if use_latex
            else get_dgcv_settings_registry().get("use_latex", False)
        )
        tree = (
            self._latex_verbose_tree(filter_conditions=filter_conditions)
            if use_latex
            else self._verbose_tree(filter_conditions=filter_conditions)
        )
        return latex_in_html(
            _full_tree_html(tree, theme=theme, root_label=root_label or self.label)
        )

    def leaf_summary(self, theme=None, sort_by=None, reverse=False, **kwargs):
        return _tree_leaves_html(
            self, theme=theme, sort_by=sort_by, reverse=reverse, **kwargs
        )

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

    def case_summary(self, plain_text=False):
        f = getattr(self, "summary", None)
        if callable(f):
            return f(self)
        param_count = len(self.system_parameters)
        if param_count == 0:
            pass
        purality_note = "pameter" if param_count == 1 else "pameters"
        print(f"The system is parameterized with {param_count} {purality_note}.")

        eqns = self.reduced_equation_system
        if check_dgcv_category(eqns):
            eqns = {0: eqns}
        else:
            eqns = [var for var in eqns]
        c_numb = len(self.case_rules)
        if c_numb == 0:
            print("The general equation system is")
        else:
            print("The condition" if c_numb == 1 else "Conditions")
            self.show_case_rules(punctuation=",", plain_text=plain_text)
            print("implies" if c_numb == 1 else "imply")

        if plain_text:
            print(eqns)
        else:
            show(
                LaTeX_eqn_system(
                    eqns,
                    punctuation=".",
                )
            )
        if param_count > 0:
            fp = self.free_parameters
            if len(fp) == 0:
                print("No free parameters remain in this branch.")
            else:
                print("The remaining free parameters in this branch are")
                if plain_text is True:
                    print(fp)
                else:
                    show(LaTeX_list(fp, one_line=True, punctuation="."))
        fv = self.free_variables
        if len(fv) == 0:
            print("No free variables remain in this branch.")
        else:
            print("The remaining free variables in this branch are")
            if plain_text is True:
                print(self.free_variables)
            else:
                show(LaTeX_list(self.free_variables, one_line=True, punctuation="."))
        if self.complete:
            print("********** The branch is complete! **********")
            print(f"     {len(self.free_variables)} dim. parameter space remaining ")


def _html_style(theme=None, container_id=None):
    if not isinstance(theme, str):
        theme = get_dgcv_settings_registry().get("theme", "dark")

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
    border-color: var(--dgcv-text-hover) !important;
}}
{scope} .compound-node:hover .node-label {{ 
    background-color: var(--dgcv-bg-hover) !important; 
    color: var(--dgcv-text-hover) !important;
    border-color: var(--dgcv-text-hover) !important;
    font-weight: bold; 
    font-size: 14px; 
    text-shadow: var(--dgcv-text-shadow, none);
}}


{scope} .children-ul {{ margin-left: 10px; }}
    """
    return f"<style>{base_styles}</style>"


def _to_html_tree(data, path="", is_root=True, root_label=None, container_id=None):
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
    if not isinstance(root_label, str):
        root_label = "root"
    if is_root and root_label in data:
        res += f'<li><div class="compound-node root-wrapper"><div class="node-label" style="min-width: 10px;">{root_label}</div></div>'
        res += _to_html_tree(data[root_label], "", False, container_id=container_id)
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
                res += (
                    f'<div class="note-msg">{html.escape("Note: " + str(note))}</div>'
                )
            res += "</div>"
            if is_dict:
                res += _to_html_tree(
                    value, current_path, False, container_id=container_id
                )
            res += "</li>"
    res += "</ul>"
    return (res + "</div>") if is_root else res


def _full_tree_html(data, theme=None, root_label=None):
    cid = f"tree-{uuid.uuid4().hex[:8]}"
    styles = _html_style(theme=theme, container_id=cid)
    tree = _to_html_tree(data, is_root=True, root_label=root_label, container_id=cid)
    return styles + tree


def _tree_leaves_html(
    tree: case_tree,
    theme=None,
    use_latex=True,
    return_displayable=False,
    sort_by: str | list = None,
    reverse=False,
    **kwargs,
):
    if not isinstance(theme, str):
        theme = get_dgcv_settings_registry().get("theme", "dark")
    data = tree._latex_verbose_tree() if use_latex else tree._verbose_tree()
    if not isinstance(data, dict):
        return
    leaves = {}

    def scan_and_descend(folder, pref=""):
        if not isinstance(folder, dict):
            return
        for k, v in folder.items():
            if isinstance(v, dict):
                dec = v.get("descendants", None)
                path = str(pref) + "." + str(k)
                if isinstance(dec, str):
                    leaves[path] = {"state": dec}
                else:
                    scan_and_descend(dec, path)

    for _, v in data.items():
        scan_and_descend(v.get("descendants", None))
    for k in leaves:
        root = tree
        steps = filter(None, k.split("."))
        for step in steps:
            root = root._subcases.get(step, None)
            if root is None:
                break
        if root is None:
            continue
        leaves[k]["conditions"] = root.case_rules
        leaves[k]["free_vars"] = root.free_variables
        leaves[k]["free_params"] = root.free_parameters

    def process_conditions(conds):
        eqns = []
        for k, v in conds.items():
            if use_latex:
                eqns.append(LaTeX_eqn_system({k: v}, math_mode="$"))
            else:
                eqns.append(str(k) + "=" + str(v))
        return ", ".join(eqns)

    def state(x):
        return "solved" if x.startswith("complete") else "unsolved"

    rows = []
    no_params = len(tree.free_parameters) == 0
    for k, v in leaves.items():
        rows.append(
            [
                (k[1:] if k.startswith(".") else k).replace("._", "."),
                state(v.get("state", "")),
                str(len(v.get("free_vars", []))),
            ]
            + ([] if no_params else [str(len(v.get("free_params", [])))])
            + [
                process_conditions(v.get("conditions", [])),
            ]
        )
    headers = (
        [
            "subcase",
            "equation state",
            "free variables",
        ]
        + ([] if no_params else ["free parameters"])
        + [
            "conditions",
        ]
    )

    def sort(rs, property):
        aliases = {
            "case": "subcase",
            "cases": "subcase",
            "subcases": "subcase",
            "label": "subcase",
            "state": "equation state",
            "equation states": "equation state",
            "states": "equation state",
            "variable": "free variables",
            "variables": "free variables",
            "var": "free variables",
            "vars": "free variables",
            "par": "free parameters",
            "pars": "free parameters",
            "parameters": "free parameters",
            "parameter": "free parameters",
            "param": "free parameters",
            "params": "free parameters",
            "case_rules": "conditions",
        }
        idxs = {"subcase": 0, "equation state": 1, "free variables": 2}
        if no_params:
            idxs |= {"conditions": 3}
        else:
            idxs |= {"free parameters": 3, "conditions": 4}
        numerics = set(
            filter(None, {idxs.get("free variables"), idxs.get("free parameters")})
        )
        tuple_sort = isinstance(property, (list, tuple))
        if tuple_sort:
            idx = [idxs.get(aliases.get(prop, prop), None) for prop in property]
        else:
            idx = idxs.get(aliases.get(property, property), None)
        if idx is None:
            if reverse:
                return rs[-1::-1]
            return rs

        def sort_key(x):
            if tuple_sort:
                return tuple(sort_class(x, y) for y in idx)
            return sort_class(x, idx)

        def sort_class(x, y):
            return int(x[y]) if y in numerics else x[y]

        return sorted(
            rs,
            key=lambda x: sort_key(x),
            reverse=reverse,
        )

    rows = sort(rows, sort_by)

    table = build_plain_table(
        columns=headers,
        rows=rows,
        theme_css_vars=get_style(theme, legacy=False),
        caption="Case tree leaves",
        table_attrs='style="table-layout:auto;"',
        container_id="tree-leaves-summary",
    )
    extra_support_for_math_in_tables = bool(
        get_dgcv_settings_registry().get("extra_support_for_math_in_tables") is True
    )
    out = latex_in_html(
        table, extra_support_for_math_in_tables=extra_support_for_math_in_tables
    )
    if return_displayable:
        return out
    show(out)
