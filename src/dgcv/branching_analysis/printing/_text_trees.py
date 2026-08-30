from __future__ import annotations

import textwrap


def print_path_tree(data, indent="", path="", root=True):
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
            print_path_tree(value, next_indent, child_path, root=False)
        else:
            print(f"{line_indent}{connector}{display_label}: {value}")

def print_verbose_tree(data, indent="", path="", root=True):

    if root and "root" in data:
        print("root")
        print_verbose_tree(data["root"], indent, path="", root=False)
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
        suffix = "" if is_dict or not str(descendants) else f": {descendants}"

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
            print_verbose_tree(value, next_indent, display_label, root=False)

