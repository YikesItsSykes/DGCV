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

import html
import textwrap
import uuid

from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
    subs,
)
from .._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
    latex_in_html,
)
from .._aux._utilities._styles import get_style
from .._aux._vmf._safeguards import check_dgcv_category
from .._aux._vmf.vmf import order_coordinates
from .._aux.printing._tables import build_plain_table
from .._aux.printing.printing._dgcv_display import (
    LaTeX_eqn_system,
    LaTeX_list,
    show,
)
from .._aux.printing.printing._string_processing import _strip_displaystyles

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
        legacy_cr = kwargs.pop("case_rules", None)

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
                if parameters:
                    vari = {v for v in vari if v not in parameters}
                self.system_variables = vari
            except Exception:
                self.system_variables = set()
        else:
            self.system_variables = set(variables)
        if parameters is None:
            try:
                all_symbols = set()
                for eqn in self.general_equation_system:
                    all_symbols |= get_free_symbols(eqn)
                self.system_parameters = {
                    x for x in all_symbols if x not in self.system_variables
                }
            except Exception:
                self.system_parameters = set()
        else:
            self.system_parameters = set(parameters)
        if not callable(completion_condition):

            def _cd(obj):
                try:

                    def ztest(x):
                        return _scalar_is_zero(self._internal_simplify(x))

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
        if not isinstance(getattr(self, "_initial_rules", None), dict):
            self._initial_rules = {}
        if legacy_cr:
            self._merge_conditions(
                self._initial_rules, self._normalize_conditions(legacy_cr), "d"
            )
        self._complete = None
        self._ev_eqn_system = None
        self._free_variables = None
        self._free_parameters = None
        self._processed_cr = None
        self._closed_cr = None
        self._open_cr = None

    @staticmethod
    def _normalize_conditions(raw):
        if not isinstance(raw, dict):
            return {"closed": {}, "open": {}}
        if "closed" in raw or "open" in raw:
            return {
                "closed": dict(raw.get("closed", {})),
                "open": dict(raw.get("open", {})),
            }
        return {"closed": dict(raw), "open": {}}

    @staticmethod
    def _merge_conditions(store, conditions, source):
        # source codes: "i" inherited, "d" defining, "c" corollary
        for e, value in conditions.get("closed", {}).items():
            if "closed" in store.get(e, {}):
                raise ValueError(
                    f"Cannot add a condition to `{e}`: it already carries a closed "
                    f"condition and is no longer present in the reduced system."
                )
            store.setdefault(e, {})["closed"] = (value, source)
        for e, values in conditions.get("open", {}).items():
            if "closed" in store.get(e, {}):
                raise ValueError(
                    f"Cannot add a condition to `{e}`: it already carries a closed "
                    f"condition and is no longer present in the reduced system."
                )
            if not isinstance(values, (set, frozenset, list, tuple)):
                values = {values}
            slot = store.setdefault(e, {})
            merged = dict(slot.get("open", ()))
            for value in values:
                merged[value] = source
            slot["open"] = tuple(merged.items())
        return store

    @staticmethod
    def _backsub_store(store, sub_dict):
        out = {}
        for e, slot in store.items():
            ns = {}
            if "closed" in slot:
                ns["closed"] = (subs(slot["closed"][0], sub_dict), "i")
            if "open" in slot:
                ns["open"] = tuple((subs(v, sub_dict), "i") for v, _ in slot["open"])
            out[e] = ns
        return out

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

    @property
    def case_rules(self):
        if self._processed_cr is None:
            s = self._internal_simplify
            out = {}
            for e, slot in self._initial_rules.items():
                ns = {}
                if "closed" in slot:
                    v, src = slot["closed"]
                    ns["closed"] = (s(v), src)
                if "open" in slot:
                    ns["open"] = tuple((s(v), src) for v, src in slot["open"])
                out[e] = ns
            self._processed_cr = out
        return self._processed_cr

    @property
    def closed_case_rules(self):
        if self._closed_cr is None:
            self._closed_cr = {
                e: slot["closed"][0]
                for e, slot in self.case_rules.items()
                if "closed" in slot
            }
        return self._closed_cr

    @property
    def open_case_rules(self):
        if self._open_cr is None:
            self._open_cr = {
                e: {v for v, _ in slot["open"]}
                for e, slot in self.case_rules.items()
                if slot.get("open")
            }
        return self._open_cr

    def add_case(
        self,
        label: str = None,
        defining_conditions: dict = None,
        corollary_conditions: dict = None,
        **kwargs,
    ):
        legacy_cr = kwargs.pop("case_rules", None)

        if label is None:
            for idx in range(1, len(self._subcases) + 2):
                pref = f"_{idx}"
                if any(str(x) == f"_{idx}" for x in self._subcases):
                    continue
                label = pref
                break

        verbose = kwargs.get("verbose", None)
        if not isinstance(verbose, bool):
            verbose = getattr(self, "verbose", False) is True
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

        defining = self._normalize_conditions(defining_conditions)
        if legacy_cr:
            defining["closed"] = {
                **defining["closed"],
                **self._normalize_conditions(legacy_cr)["closed"],
            }
        corollary = self._normalize_conditions(corollary_conditions)

        new_closed = {**defining["closed"], **corollary["closed"]}
        child_rules = self._backsub_store(self.case_rules, new_closed)
        self._merge_conditions(child_rules, defining, "d")
        self._merge_conditions(child_rules, corollary, "c")

        kwargs["_initial_rules"] = child_rules
        new_tree = case_tree(
            label=label,
            equation_system=self.general_equation_system,
            variables=self.system_variables,
            parameters=self.system_parameters,
            completion_condition=self.completion_condition,
            **{"_simplify_rule": self._internal_simplify, **kwargs},
        )
        setattr(self, label, new_tree)
        self._subcases[label] = new_tree
        return new_tree

    def add_corollary(self, label: str, corollary_conditions: dict = None, **kwargs):
        legacy_cr = kwargs.pop("case_rules", None)
        if corollary_conditions is None and legacy_cr is not None:
            corollary_conditions = legacy_cr
        return self.add_case(
            label=label, corollary_conditions=corollary_conditions, **kwargs
        )

    def remove_case(self, label):
        if label in self._subcases:
            _ = self._subcases.pop(label, None)
            if hasattr(self, label):
                delattr(self, label)

    def _repr_latex_(self, raw=False, **kwargs):
        out = self._conditions_str(
            {"closed": self.closed_case_rules, "open": self.open_case_rules},
            _punct="",
            **kwargs,
        )
        if raw is True:
            out = _strip_displaystyles(out)
        return out

    def __str__(self):
        return getattr(self, "inheritance_path", "") + self.label

    def show_case_rules(self, plain_text=False, **kwargs):
        if plain_text is True:
            print({"closed": self.closed_case_rules, "open": self.open_case_rules})
        else:
            show(self._repr_latex_(**kwargs))

    @property
    def reduced_equation_system(self):
        if self._ev_eqn_system is None:
            if hasattr(self.general_equation_system, "subs"):
                self._ev_eqn_system = self._internal_simplify(
                    subs(self.general_equation_system, self.closed_case_rules)
                )
            else:
                self._ev_eqn_system = [
                    self._internal_simplify(subs(x, self.closed_case_rules))
                    for x in self.general_equation_system
                ]
        return self._ev_eqn_system

    @property
    def variable_constraints(self):
        params = self.system_parameters
        return {
            "closed": {
                e: v for e, v in self.closed_case_rules.items() if e not in params
            },
            "open": {
                e: vals for e, vals in self.open_case_rules.items() if e not in params
            },
        }

    @property
    def parameter_conditions(self):
        params = self.system_parameters
        return {
            "closed": {e: v for e, v in self.closed_case_rules.items() if e in params},
            "open": {
                e: vals for e, vals in self.open_case_rules.items() if e in params
            },
        }

    @property
    def free_variables(self):
        if self._free_variables is None:
            scr = self.closed_case_rules
            fvd = {v: subs(v, scr) for v in self.system_variables}
            fv = set()
            for _, v in fvd.items():
                fv |= get_free_symbols(v)
            self._free_variables = [v for v in fv if v in self.system_variables]
        return self._free_variables

    @property
    def free_parameters(self):
        if self._free_parameters is None:
            scr = self.closed_case_rules
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

    def _conditions_str(self, bucket, plain_text=False, **kwargs):
        punct = kwargs.pop("_punct", None)
        if not isinstance(punct, str):
            punct = ","
        closed = bucket.get("closed", {})
        open_rules = bucket.get("open", {})
        if plain_text:
            parts = [f"{e} = {v}" for e, v in closed.items()]
            parts += [f"{e} \u2260 {v}" for e, vals in open_rules.items() for v in vals]
            return ", ".join(parts)
        open_list = [(e, v) for e, vals in open_rules.items() for v in vals]
        if len(open_list) == 0:
            out = LaTeX_eqn_system(
                closed, **{"one_line": True, "punctuation": punct, **kwargs}
            )
        else:
            out = LaTeX_eqn_system(
                closed,
                **{
                    "one_line": True,
                    "punctuation": punct,
                    "conjuction": " ",
                    "force_oxford_comma": True,
                    **kwargs,
                },
            )[:-2]
        if len(open_list) == 1:
            if len(closed) > 0:
                if len(closed) == 1 and out[-1:] == ",":
                    out = out[:-1]
                out += (
                    r"\quad\text{and}\quad "
                    + (
                        LaTeX_eqn_system(
                            open_list,
                            **{
                                "one_line": True,
                                "punctuation": punct,
                                "relation": r"\neq",
                                **kwargs,
                            },
                        )[2:]
                    )
                )
            else:
                out = LaTeX_eqn_system(
                    open_list,
                    **{
                        "one_line": True,
                        "punctuation": punct,
                        "relation": r"\neq",
                        **kwargs,
                    },
                )
        elif len(open_list) > 1:
            if len(closed) > 0:
                out += LaTeX_eqn_system(
                    open_list,
                    **{
                        "one_line": True,
                        "punctuation": punct,
                        "relation": r"\neq",
                        "force_oxford_comma": True,
                        **kwargs,
                    },
                )[2:]
            else:
                out = LaTeX_eqn_system(
                    open_list,
                    **{
                        "one_line": True,
                        "punctuation": punct,
                        "relation": r"\neq",
                        **kwargs,
                    },
                )

        return out

    def _condition_strings(
        self, sources, latex, filter_conditions=None, key_filter=None
    ):
        flt = filter_conditions if callable(filter_conditions) else (lambda x: x)
        keep = key_filter if callable(key_filter) else (lambda e: True)
        out = []
        for e, slot in self.case_rules.items():
            if not keep(e):
                continue
            if "closed" in slot:
                v, src = slot["closed"]
                if src in sources:
                    ke, ve = flt(e), flt(v)
                    out.append(
                        LaTeX_eqn_system({ke: ve}, math_mode="$")
                        if latex
                        else f"{ke} = {ve}"
                    )
            for v, src in slot.get("open", ()):
                if src in sources:
                    ke, ve = flt(e), flt(v)
                    out.append(
                        LaTeX_eqn_system({ke: ve}, relation=r" \neq ", math_mode="$")
                        if latex
                        else f"{ke} \u2260 {ve}"
                    )
        return out

    def _verbose_tree(self, filter_conditions=None):
        branch_conditions = self._grouped_branch_conditions(
            latex=False, filter_conditions=filter_conditions
        )
        tree_dict = {
            self.label: {"branch_conditions": branch_conditions, "descendants": {}}
        }
        if getattr(self, "note", None) is not None:
            tree_dict[self.label]["note"] = str(self.note)
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
        branch_conditions = self._grouped_branch_conditions(
            latex=True, filter_conditions=filter_conditions
        )
        tree_dict = {
            self.label: {"branch_conditions": branch_conditions, "descendants": {}}
        }
        if getattr(self, "note", None) is not None:
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

    def _grouped_branch_conditions(self, latex, filter_conditions=None):
        params = self.system_parameters
        return {
            "parameters": self._condition_strings(
                {"d"}, latex, filter_conditions, key_filter=lambda e: e in params
            ),
            "variables": self._condition_strings(
                {"d"}, latex, filter_conditions, key_filter=lambda e: e not in params
            ),
            "corollaries": self._condition_strings({"c"}, latex, filter_conditions),
        }

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
                connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
                line_indent = indent
                child_path = display_label

            if isinstance(value, dict):
                print(f"{line_indent}{connector}{display_label}")
                next_indent = (
                    indent + ("    " if is_last else "\u2502   ") if not root else ""
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

            prefix = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            vertical_gate = "    " if is_last else "\u2502   "

            descendants = value.get("descendants")
            is_dict = isinstance(descendants, dict)
            suffix = "" if is_dict else f": {descendants}"

            for idx, text in enumerate(chunks):
                if idx == 0:
                    b_open = "\u23a1" if len(chunks) > 1 else "["
                    b_close = "\u23a4" if len(chunks) > 1 else "]"
                    print(
                        f"{indent}{prefix}{b_open}{text.ljust(max_w)}{b_close}\u2500{display_label}{suffix}"
                    )
                else:
                    b_mid = " \u23a2" if idx < len(chunks) - 1 else " \u23a3"
                    b_end = "\u23a5" if idx < len(chunks) - 1 else "\u23a6"

                    strut = ""
                    if is_dict:
                        strut = " " * (len(display_label) // 2) + "\u2502"

                    print(
                        f"{indent}{vertical_gate[:3]}{b_mid}{text.ljust(max_w)}{b_end}{strut}"
                    )

            if is_dict:
                padding_width = (max_w + 3) if chunks else 0
                next_indent = indent + vertical_gate + (" " * padding_width)
                cls._print_verbose_tree(value, next_indent, display_label, root=False)

    def case_summary(self, plain_text=False):
        f = getattr(self, "summary", None)
        if callable(f):
            return f(self)
        param_count = len(self.system_parameters)
        purality_note = "parameter" if param_count == 1 else "parameters"
        print(f"The system is parameterized with {param_count} {purality_note}.")

        eqns = self.reduced_equation_system
        if check_dgcv_category(eqns):
            eqns = {0: eqns}
        else:
            eqns = [var for var in eqns]

        param_bucket = self.parameter_conditions
        var_bucket = self.variable_constraints
        p_count = len(param_bucket["closed"]) + sum(
            len(s) for s in param_bucket["open"].values()
        )
        v_count = len(var_bucket["closed"]) + sum(
            len(s) for s in var_bucket["open"].values()
        )

        def _emit(bucket):
            if plain_text:
                print(self._conditions_str(bucket, plain_text=True))
            else:
                show(self._conditions_str(bucket, punctuation=","))

        if p_count == 0 and v_count == 0:
            print("The general equation system is")
        else:
            if p_count > 0:
                print(
                    "Restricting to the subfamily defined by the parameter condition"
                    + ("" if p_count == 1 else "s")
                )
                _emit(param_bucket)
            if v_count > 0:
                print(
                    ("with" if p_count > 0 else "Imposing")
                    + " the variable constraint"
                    + ("" if v_count == 1 else "s")
                )
                _emit(var_bucket)
            print("the equation system becomes")

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
            print(f"     {len(self.free_parameters)} dim. parameter space remaining ")


def _html_style(theme=None, container_id=None, slim=False):
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

{scope} .cond-box {{ padding: 0; vertical-align: top; }}

{scope} .cond-compartment {{
    position: relative;
    padding: 7px 14px 6px;
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

{scope} .cond-content {{ font-size: 12px; }}

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
    border-image: var(--dgcv-border-image, none);
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
    background: var(--dgcv-bg-hover);
    background-color: var(--dgcv-bg-hover) !important; 
    color: var(--dgcv-text-hover) !important;
    border-color: var(--dgcv-text-hover) !important;
}}
{scope} .compound-node:hover .cond-compartment + .cond-compartment {{
    border-top-color: var(--dgcv-text-hover);
}}
{scope} .compound-node:hover .cond-chip {{
    background: var(--dgcv-bg-hover) !important;
    color: var(--dgcv-text-hover) !important;
}}
{scope} .node-label:hover {{
background: var(--dgcv-bg-hover);
color: var(--dgcv-text-hover) !important;
border-color: var(--dgcv-text-hover) !important;
}}
{scope} .compound-node:hover .cond-compartment + .cond-compartment {{
    border-top-color: var(--dgcv-text-hover);
}}
{scope} .compound-node:hover .cond-chip {{
    background: var(--dgcv-bg-hover) !important;
    color: var(--dgcv-text-hover) !important;
}}
{scope} .compound-node:hover .cond-chip::before {{
    border-color: var(--dgcv-text-hover);
}}
{scope} .children-ul {{ margin-left: 10px; }}
    """
    return f"<style>{base_styles}</style>"


def _to_html_tree(
    data, path="", is_root=True, root_label=None, container_id=None, slim=False
):
    if not isinstance(data, dict):
        return ""

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
        root_node = data[root_label] if isinstance(data[root_label], dict) else {}
        root_note = root_node.get("note")
        note_html = (
            f'<div class="note-msg">{html.escape("Note: " + str(root_note))}</div>'
            if root_note is not None
            else ""
        )
        res += (
            f'<li><div class="compound-node root-wrapper">'
            f'<div class="node-label" style="min-width: 10px;">{root_label}</div>'
            f"{note_html}</div>"
        )
        res += _to_html_tree(
            data[root_label], "", False, container_id=container_id, slim=slim
        )
        res += "</li>"
    else:
        items = list(data.get("descendants", {}).items())
        for key, value in items:
            clean_key = str(key).strip("_")
            current_path = f"{path}.{clean_key}" if path else clean_key

            bc = value.get("branch_conditions", {}) if isinstance(value, dict) else {}
            if isinstance(bc, dict):
                groups = [
                    ("parameters", bc.get("parameters", [])),
                    ("variables", bc.get("variables", [])),
                    ("corollaries", bc.get("corollaries", [])),
                ]
            elif isinstance(bc, str):
                groups = [(None, [bc.strip("[]'")] if bc else [])]
            else:
                groups = [(None, list(bc))]

            descendants = value.get("descendants") if isinstance(value, dict) else value
            is_dict = isinstance(descendants, dict)

            res += '<li><div class="compound-node">'
            res += f'<div class="node-label">{html.escape(current_path)}</div>'

            inner = ""
            for chip_label, group_items in groups:
                if not group_items:
                    continue
                body = ",\u00a0  ".join(html.escape(str(c)) for c in group_items)
                chip = (
                    f'<span class="cond-chip">{html.escape(chip_label)}</span>'
                    if chip_label
                    else ""
                )
                inner += (
                    f'<div class="cond-compartment">{chip}'
                    f'<div class="cond-content">{body}</div></div>'
                )
            if not inner:
                inner = (
                    '<div class="cond-compartment">'
                    '<div class="cond-content">None</div></div>'
                )
            res += f'<div class="cond-box">{inner}</div>'

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
                    value, current_path, False, container_id=container_id, slim=slim
                )
            res += "</li>"

    res += "</ul>"
    return (res + "</div>") if is_root else res


def _full_tree_html(data, theme=None, root_label=None, slim=False):

    cid = f"tree-{uuid.uuid4().hex[:8]}"
    styles = _html_style(theme=theme, container_id=cid, slim=slim)
    tree_html = _to_html_tree(
        data, is_root=True, root_label=root_label, container_id=cid, slim=slim
    )

    return styles + tree_html


def _tree_leaves_html(
    tree: case_tree,
    theme=None,
    use_latex=True,
    return_displayable=False,
    sort_by: str | list = None,
    reverse=False,
    hide_variable_constraints: bool = False,
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
        leaves[k]["var_conditions"] = root.variable_constraints
        leaves[k]["param_conditions"] = root.parameter_conditions
        leaves[k]["free_vars"] = root.free_variables
        leaves[k]["free_params"] = root.free_parameters

    def process_conditions(conds):
        closed = conds.get("closed", {})
        open_ = conds.get("open", {})
        eqns = []
        for e in order_coordinates(list(closed)):
            v = closed[e]
            eqns.append(
                LaTeX_eqn_system({e: v}, math_mode="$") if use_latex else f"{e} = {v}"
            )
        for e in order_coordinates(list(open_)):
            for v in sorted(open_[e], key=str):
                eqns.append(
                    LaTeX_eqn_system({e: v}, relation=r" \neq ", math_mode="$")
                    if use_latex
                    else f"{e} \u2260 {v}"
                )
        return ", ".join(eqns)

    def state(x):
        return "solved" if x.startswith("complete") else "unsolved"

    no_params = len(tree.free_parameters) == 0
    show_params = not no_params
    show_var_constraints = not hide_variable_constraints

    headers = ["subcase", "equation state"]
    if show_params:
        headers.append("free parameters")
    headers.append("free variables")
    if show_params:
        headers.append("parameter conditions")
    if show_var_constraints:
        headers.append("variable constraints")

    rows = []
    for k, v in leaves.items():
        row = [
            (k[1:] if k.startswith(".") else k).replace("._", "."),
            state(v.get("state", "")),
        ]
        if show_params:
            row.append(str(len(v.get("free_params", []))))
        row.append(str(len(v.get("free_vars", []))))
        if show_params:
            row.append(process_conditions(v.get("param_conditions", {})))
        if show_var_constraints:
            row.append(process_conditions(v.get("var_conditions", {})))
        rows.append(row)

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
            "case_rules": "variable constraints",
            "conditions": "variable constraints",
            "constraint": "variable constraints",
            "constraints": "variable constraints",
            "variable constraint": "variable constraints",
            "var constraints": "variable constraints",
            "var constraint": "variable constraints",
            "parameter condition": "parameter conditions",
            "param conditions": "parameter conditions",
            "param condition": "parameter conditions",
        }
        idxs = {"subcase": 0, "equation state": 1}
        pos = 2
        if show_params:
            idxs["free parameters"] = pos
            pos += 1
        idxs["free variables"] = pos
        pos += 1
        if show_params:
            idxs["parameter conditions"] = pos
            pos += 1
        if show_var_constraints:
            idxs["variable constraints"] = pos
            pos += 1
        numerics = set(
            filter(
                lambda i: i is not None,
                {idxs.get("free variables"), idxs.get("free parameters")},
            )
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
