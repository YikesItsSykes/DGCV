"""
package: dgcv - Differential Geometry with Complex Variables
module: printing/_string_processing

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""
# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------

import re
from typing import Any

from .._config import greek_letters


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def _format_label_with_hi_low(label: str) -> str:
    if not label:
        return label

    decoration_cmd = None
    core = label

    prefix_map = {
        "tilde_": r"\widetilde",
        "hat_": r"\hat",
        "widehat_": r"\widehat",
        "bar_": r"\bar",
        "overline_": r"\overline",
        "overline": r"\overline",
        "bar": r"\bar",
    }
    for prefix in sorted(prefix_map, key=len, reverse=True):
        if core.startswith(prefix):
            decoration_cmd = prefix_map[prefix]
            core = core[len(prefix) :]
            break

    if "_low_" in core or "_hi_" in core:
        idx_low = core.find("_low_")
        idx_hi = core.find("_hi_")
        index_start_candidates = [i for i in (idx_low, idx_hi) if i != -1]
        if index_start_candidates:
            index_start = min(index_start_candidates)
            base_part = core[:index_start]
            index_part = core[index_start:]
        else:
            base_part = core
            index_part = ""

        lower_indices = []
        upper_indices = []
        primes_count = 0

        upper_part = ""
        if "_low_" in index_part:
            lower_part = index_part.split("_low_")[1]
            if "_hi_" in lower_part:
                lower_part, upper_part = lower_part.split("_hi_", 1)
            lower_indices = [tok for tok in lower_part.split("_") if tok]
        if upper_part == "" and "_hi_" in index_part:
            upper_part = index_part.split("_hi_")[1]

        if upper_part:
            tokens = [t for t in upper_part.split("_") if t]
            while tokens and re.fullmatch(r"p+", tokens[0]):
                primes_count += len(tokens[0])
                tokens.pop(0)
            upper_indices = tokens

        prime_tex = r"\prime" * primes_count
        if upper_indices:
            upper_tex = prime_tex + ("," if prime_tex else "") + ",".join(upper_indices)
        else:
            upper_tex = prime_tex

        base_tex = latexify_base(base_part)
        if decoration_cmd is not None:
            base_tex = f"{decoration_cmd}{{{base_tex}}}"

        indices_str = ""
        if upper_tex:
            indices_str += f"^{{{upper_tex}}}"
        if lower_indices:
            indices_str += f"_{{{','.join(lower_indices)}}}"

        return base_tex + indices_str

    if "__" not in core and "_" not in core:
        out = latexify_base(core)
        if decoration_cmd is not None:
            out = f"{decoration_cmd}{{{out}}}"
        return out

    m0 = re.search(r"_", core)
    if not m0:
        base_part = core
        rest = ""
    else:
        base_part = core[: m0.start()]
        rest = core[m0.start() :]

    upper_tokens = []
    lower_tokens = []

    pos = 0
    while pos < len(rest):
        if rest[pos] != "_":
            break
        run_start = pos
        while pos < len(rest) and rest[pos] == "_":
            pos += 1
        run_len = pos - run_start
        tok_start = pos
        while pos < len(rest) and rest[pos] != "_":
            pos += 1
        tok = rest[tok_start:pos]
        if not tok:
            continue
        if run_len >= 2:
            upper_tokens.append(tok)
        else:
            lower_tokens.append(tok)

    base_tex = latexify_base(base_part)
    if decoration_cmd is not None:
        base_tex = f"{decoration_cmd}{{{base_tex}}}"

    indices_str = ""
    if upper_tokens:
        indices_str += f"^{{{','.join(upper_tokens)}}}"
    if lower_tokens:
        indices_str += f"_{{{','.join(lower_tokens)}}}"

    return base_tex + indices_str


def latexify_base(base: str) -> str:
    if not base:
        return base
    if "_" not in base and base and base[-1].isdigit():
        m = re.match(r"^(.+?)(\d+)$", base)
        if m:
            name, digits = m.groups()
            return f"{name}_{{{digits}}}"
    if base.startswith("_"):
        return r"\_" + format_latex_subscripts(base[1:])
    return format_latex_subscripts(base)


def _latex_escape_text(s):
    return (
        s.replace("\\", r"\textbackslash ")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("$", r"\$")
        .replace("^", r"\^{}")
        .replace("~", r"\~{}")
    )


def convert_to_greek(var_name):
    for name, greek in greek_letters.items():
        if var_name.lower().startswith(name):
            return var_name.replace(name, greek, 1)
    return var_name


def format_latex_subscripts(var_name, nest_braces=False):
    """for use_latex branches"""
    if var_name[-1] == "_":
        var_name = var_name[:-1]
    if var_name == "":
        return r"\_"
    if var_name[0] == "_":
        return format_latex_subscripts(var_name=var_name[1:], nest_braces=nest_braces)
    parts = var_name.split("_")
    if len(parts) == 1:
        return convert_to_greek(var_name)
    base = convert_to_greek(parts[0])
    subscript = ", ".join(parts[1:])
    if nest_braces is True:
        return f"{{{base}_{{{subscript}}}}}"
    else:
        return f"{base}_{{{subscript}}}"


def latex_superscript(base: str, power: str) -> str:
    if power == "":
        return base
    if len(power) > 1:
        power = f"{{{power}}}"
    if "^" in base:
        return rf"\left({base}\right)^{{{power}}}"
    return f"{base}^{power}"


_dollars = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def clean_LaTeX(word: str, replacements: dict[str, str] | None = None) -> str:
    word = re.sub(r"\\displaystyle\s*", "", word)
    word = _dollars.sub(r"\\[\1\\]", word)
    word = _collapse_double_braces(word)
    word = _format_display_math(word)
    if replacements:
        for old, new in replacements.items():
            word = word.replace(old, new)
    return word


_strip_display_pat = re.compile(r"|".join(map(re.escape, ["$", r"\displaystyle"])))


def _strip_display_dollars(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return _strip_display_pat.sub("", s)


def _strip_displaystyles(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace(r"\displaystyle", "")


def _strip_dollars(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("$", "")


def _unwrap_math_delims(s: str) -> str:
    t = s.strip()
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def _coerce_to_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        try:
            return repr(x)
        except Exception:
            return ""


def _collapse_double_braces(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "{" and i + 1 < n and s[i + 1] == "{":
            depth = 0
            j = i
            last = None
            while j < n:
                c = s[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        last = j
                        break
                j += 1
            if last is not None and last > i + 1 and s[last - 1] == "}":
                inner = s[i + 2 : last - 1]
                inner = _collapse_double_braces(inner)
                out.append("{")
                out.append(inner)
                out.append("}")
                i = last + 1
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _format_display_math(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n and s[i + 1] in ("[", "]"):
            if out and out[-1] != "\n":
                out.append("\n")
            out.append("\\")
            out.append(s[i + 1])
            if i + 2 < n and s[i + 2] != "\n":
                out.append("\n")
            i += 2
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _strip_outer_parens_plain(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return s
        return s[1:-1].strip()
    return s


def _strip_outer_parens_latex(s: str) -> str:
    s = s.strip()
    if s.startswith(r"\left(") and s.endswith(r"\right)"):
        body = s[len(r"\left(") : -len(r"\right)")].strip()
        return body
    return _strip_outer_parens_plain(s)


def _needs_parens_plain(s: str) -> bool:
    s = s.strip()
    if not s:
        return False

    dp = db = dc = 0
    for i, ch in enumerate(s):
        if ch == "(":
            dp += 1
            continue
        if ch == ")":
            dp = max(0, dp - 1)
            continue
        if ch == "[":
            db += 1
            continue
        if ch == "]":
            db = max(0, db - 1)
            continue
        if ch == "{":
            dc += 1
            continue
        if ch == "}":
            dc = max(0, dc - 1)
            continue

        if dp or db or dc:
            continue

        if ch == "+":
            return True
        if ch == "-" and i > 0:
            return True

    return False


def _needs_parens_latex(s: str) -> bool:
    s = s.strip()
    if not s:
        return False

    i = 0
    n = len(s)
    ctx = []
    frac_wait = 0

    while i < n:
        if s.startswith(r"\frac", i):
            i += 5
            frac_wait = 2
            continue

        ch = s[i]

        if ch == "^" and i + 1 < n and s[i + 1] == "{":
            ctx.append("sup")
            i += 2
            continue
        if ch == "_" and i + 1 < n and s[i + 1] == "{":
            ctx.append("sub")
            i += 2
            continue

        if ch == "{":
            if frac_wait > 0:
                ctx.append("frac")
                frac_wait -= 1
            else:
                ctx.append("brace")
            i += 1
            continue
        if ch == "}":
            if ctx:
                ctx.pop()
            i += 1
            continue

        if (ch == "+" or ch == "-") and not ctx:
            return True

        i += 1

    return False


def _split_leading_minus(s: str):
    s = s.strip()
    if s.startswith("-"):
        return True, s[1:].strip()
    return False, s


def coeff_needs_parens_plain(s: str) -> bool:
    return _needs_parens_plain(s)


def coeff_needs_parens_latex(s: str) -> bool:
    s = s.strip()
    if not s:
        return False

    i = 0
    n = len(s)

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    left_depth = 0

    frac_wait = 0
    seen_top_token = False

    while i < n:
        if s.startswith(r"\frac", i):
            i += 5
            frac_wait = 2
            continue

        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "\\":
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            macro = s[i:j]

            if macro == r"\left":
                k = j
                while k < n and s[k].isspace():
                    k += 1
                if k < n:
                    d = s[k]
                    if d == "(":
                        paren_depth += 1
                        left_depth += 1
                        i = k + 1
                        continue
                    if d == "[":
                        bracket_depth += 1
                        left_depth += 1
                        i = k + 1
                        continue
                    if d == r"\\" and k + 1 < n and s[k + 1] == "{":
                        brace_depth += 1
                        left_depth += 1
                        i = k + 2
                        continue
                    if d == "{":
                        brace_depth += 1
                        left_depth += 1
                        i = k + 1
                        continue
                i = j
                continue

            if macro == r"\right":
                k = j
                while k < n and s[k].isspace():
                    k += 1
                if k < n:
                    d = s[k]
                    if d == ")":
                        paren_depth = max(0, paren_depth - 1)
                        left_depth = max(0, left_depth - 1)
                        i = k + 1
                        continue
                    if d == "]":
                        bracket_depth = max(0, bracket_depth - 1)
                        left_depth = max(0, left_depth - 1)
                        i = k + 1
                        continue
                    if d == r"\\" and k + 1 < n and s[k + 1] == "}":
                        brace_depth = max(0, brace_depth - 1)
                        left_depth = max(0, left_depth - 1)
                        i = k + 2
                        continue
                    if d == "}":
                        brace_depth = max(0, brace_depth - 1)
                        left_depth = max(0, left_depth - 1)
                        i = k + 1
                        continue
                i = j
                continue

            i = j
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue

        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            i += 1
            continue

        if ch == "{":
            if frac_wait > 0:
                brace_depth += 1
                frac_wait -= 1
            else:
                brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            i += 1
            continue

        top_level = (
            (paren_depth == 0)
            and (bracket_depth == 0)
            and (brace_depth == 0)
            and (left_depth == 0)
        )

        if top_level:
            if ch == "+":
                return True
            if ch == "-":
                if seen_top_token:
                    return True
                i += 1
                continue
            seen_top_token = True

        i += 1

    return False


def _process_var_label(var, bypass=False):
    if bypass is True:
        return var
    var_str = str(var)
    is_conjugate = False

    if var_str.startswith("BAR"):
        var_str = var_str[3:]
        is_conjugate = True

    match = re.match(r"(.*?)(\d*)$", var_str)
    if match:
        label_part = match.group(1).rstrip("_")
        number_part = match.group(2)
        label_part = convert_to_greek(label_part)
        formatted_label = (
            f"{label_part}_{{{number_part}}}" if number_part else label_part
        )
        return f"\\overline{{{formatted_label}}}" if is_conjugate else formatted_label

    return var_str
