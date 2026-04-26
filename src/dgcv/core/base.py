"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core

module: dgcv.core.base


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .._aux._utilities._config import get_dgcv_settings_registry


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
class dgcv_class:
    def __dgcv_simplify__(self, method=None, **kwargs):
        return self._eval_simplify(**kwargs)

    def _eval_simplify(self, **kwargs):
        return self

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        if (
            type(self)._latex is dgcv_class._latex
            and type(self)._repr_latex_ is dgcv_class._repr_latex_
        ):
            return self.__str__()
        return self._repr_latex_(raw=raw)

    def _repr_latex_(self, raw: bool = False, **kwargs):
        if (
            type(self)._latex is dgcv_class._latex
            and type(self)._repr_latex_ is dgcv_class._repr_latex_
        ):
            return self.__str__()
        s = self._latex(raw=True, **kwargs)
        return s if raw else f"$\\displaystyle {s}$"

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    def __repr__(self):
        dgcvSR = get_dgcv_settings_registry()
        if dgcvSR.get("print_style", None) == "readable":
            return str(self)
        return object.__repr__(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        try:
            latex = self._repr_latex_(raw=False)
        except Exception:
            latex = None

        try:
            plain = self.__repr__()
        except Exception:
            try:
                plain = str(self)
            except Exception:
                plain = ""

        bundle = {"text/plain": plain if isinstance(plain, str) else str(plain)}
        if isinstance(latex, str) and latex.strip():
            bundle["text/latex"] = latex
        return bundle


class annotated_container:
    __slots__ = ("_obj", "_annotations")

    def __init__(self, obj: Any, **annotations: Any) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_annotations", dict(annotations))

    @property
    def object(self) -> Any:
        return self._obj

    @property
    def annotations(self) -> dict[str, Any]:
        return self._annotations

    @property
    def _dgcv_notes(self) -> dict[str, Any]:
        return self._annotations.get("_dgcv_notes", {})

    def get_annotations(self) -> dict[str, Any]:
        return dict(self._annotations)

    def with_annotations(self, **annotations: Any) -> "annotated_container":
        merged = {**self._annotations, **annotations}
        return type(self)(self._obj, **merged)

    def __repr__(self) -> str:
        return repr(self._obj)

    def __str__(self) -> str:
        return str(self._obj)

    def __bytes__(self) -> bytes:
        return bytes(self._obj)

    def __format__(self, format_spec: str) -> str:
        return format(self._obj, format_spec)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(type(self)) + dir(self._obj) + list(self.__slots__)))

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "_obj",
            "_annotations",
            "object",
            "annotations",
            "get_annotations",
            "with_annotations",
            "get_tags",
            "with_tag",
            "__slots__",
            "__dict__",
            "__class__",
        }:
            return object.__getattribute__(self, name)
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_obj", "_annotations"}:
            object.__setattr__(self, name, value)
        elif name in {"object", "annotations"}:
            raise AttributeError(f"{name!r} is read-only")
        else:
            setattr(self._obj, name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"_obj", "_annotations", "object", "annotations"}:
            raise AttributeError(name)
        delattr(self._obj, name)

    def __len__(self) -> int:
        return len(self._obj)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._obj)

    def __reversed__(self) -> Iterator[Any]:
        return reversed(self._obj)

    def __contains__(self, item: Any) -> bool:
        return item in self._obj

    def __getitem__(self, key: Any) -> Any:
        return self._obj[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._obj[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._obj[key]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._obj(*args, **kwargs)

    def __enter__(self) -> Any:
        return self._obj.__enter__()

    def __exit__(self, exc_type, exc, tb) -> Any:
        return self._obj.__exit__(exc_type, exc, tb)

    def __hash__(self) -> int:
        return hash(self._obj)

    def __eq__(self, other: Any) -> bool:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj == other

    def __ne__(self, other: Any) -> bool:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj != other

    def __lt__(self, other: Any) -> bool:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj < other

    def __le__(self, other: Any) -> bool:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj <= other

    def __gt__(self, other: Any) -> bool:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj > other

    def __ge__(self, other: Any) -> bool:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj >= other

    def __bool__(self) -> bool:
        return bool(self._obj)

    def __int__(self) -> int:
        return int(self._obj)

    def __float__(self) -> float:
        return float(self._obj)

    def __index__(self) -> int:
        return self._obj.__index__()

    def __add__(self, other: Any) -> Any:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj + other

    def __radd__(self, other: Any) -> Any:
        other = other._obj if isinstance(other, annotated_container) else other
        return other + self._obj

    def __sub__(self, other: Any) -> Any:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj - other

    def __rsub__(self, other: Any) -> Any:
        other = other._obj if isinstance(other, annotated_container) else other
        return other - self._obj

    def __mul__(self, other: Any) -> Any:
        other = other._obj if isinstance(other, annotated_container) else other
        return self._obj * other

    def __rmul__(self, other: Any) -> Any:
        other = other._obj if isinstance(other, annotated_container) else other
        return other * self._obj
