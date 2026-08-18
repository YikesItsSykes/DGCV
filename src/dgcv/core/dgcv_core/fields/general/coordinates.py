from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, Literal, Tuple

from ....._aux._backends._symbolic_router import _scalar_is_zero
from ....._aux._vmf.vmf import vmf_lookup
from .workers import _missing_system_msg, _variable_spaces_types_algo


class _tensor_field_coordinates:
    @property
    def minimal_coordinate_space(self):
        if self._minimal_coordinate_space is not None:
            return self._minimal_coordinate_space

        coor = set()
        seen = {}
        for key in self.coeff_dict:
            deg = len(key) // 3
            ks, ke = key[:deg], key[2 * deg :]
            for v, z in zip(ks, ke):
                space = self._variable_spaces.get(z)
                if space is None:
                    space = list(range(v))
                variable = space[v]
                info = seen.get(variable)
                if info is None:
                    info = seen[variable] = vmf_lookup(
                        variable,
                        flattened_relatives=True,
                    )
                if info.get("sub_type") == "holo":
                    rels = info.get("flattened_relatives")
                    coor |= {rels[0], rels[2], rels[3]}
                elif info.get("sub_type") == "anti":
                    rels = info.get("flattened_relatives")
                    coor |= {rels[1], rels[2], rels[3]}
                elif info.get("sub_type") == "real":
                    rels = info.get("flattened_relatives")
                    coor |= {rels[0], rels[1], rels[2]}
                elif info.get("sub_type") == "imag":
                    rels = info.get("flattened_relatives")
                    coor |= {rels[0], rels[1], rels[3]}
                else:
                    coor.add(variable)
        self._minimal_coordinate_space = coor
        return self._minimal_coordinate_space

    def _require_cached_system(self, system_label: str) -> Tuple[Any, ...]:
        cached = self._variable_spaces.get(system_label, None)
        if isinstance(cached, tuple):
            self._system_index_cache.setdefault(
                system_label, {v: i for i, v in enumerate(cached)}
            )
            return cached
        if isinstance(cached, list):
            vs = tuple(cached)
            self._variable_spaces[system_label] = vs
            self._system_index_cache[system_label] = {v: i for i, v in enumerate(vs)}
            return vs
        raise KeyError(_missing_system_msg(system_label))

    @property
    def variable_spaces_types(self):
        if self._variable_spaces_types is None:
            self._variable_spaces_types = _variable_spaces_types_algo(
                getattr(self, "_variable_spaces", {})
            )
        return self._variable_spaces_types

    def infer_varSpace(
        self,
        formatting: Literal["real", "complex", "any"] = "complex",
        *,
        return_dict: bool = False,
    ):
        cache = getattr(self, "_infer_varSpace_cache", None)
        if cache is None:
            cache = self._infer_varSpace_cache = {}

        key = (formatting, bool(return_dict))
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        if formatting not in ("real", "complex", "any"):
            raise ValueError("formatting must be one of: 'real', 'complex', 'any'")

        systems = getattr(self, "_variable_spaces", None)
        if not isinstance(systems, dict) or not systems:
            out = (tuple(), {}) if return_dict else tuple()
            cache[key] = out
            return out

        std_bucket = []
        holo_bucket = []
        anti_bucket = []
        real_bucket = []
        imag_bucket = []

        sys_info_cache: Dict[Any, dict] = {}
        for syslbl in sorted(systems.keys(), key=lambda x: str(x)):
            info = vmf_lookup(
                syslbl, path=True, relatives=True, flattened_relatives=True
            )
            sys_info_cache[syslbl] = info

            if info.get("type") != "coordinate":
                if syslbl is None:
                    atoms = tuple(sorted(systems.get(None) or (), key=str))
                    if atoms:
                        std_bucket.append((syslbl, atoms))
                continue

            sub = info.get("sub_type", None)
            rel = info.get("relatives") or {}

            if sub != "complex":
                st = rel.get("standard", None)
                if isinstance(st, tuple) and st:
                    std_bucket.append((syslbl, st))
                continue

            h = rel.get("holo", None)
            a = rel.get("anti", None)
            r = rel.get("real", None)
            i = rel.get("imag", None)

            if isinstance(h, tuple) and h:
                holo_bucket.append((syslbl, h))
            if isinstance(a, tuple) and a:
                anti_bucket.append((syslbl, a))
            if isinstance(r, tuple) and r:
                real_bucket.append((syslbl, r))
            if isinstance(i, tuple) and i:
                imag_bucket.append((syslbl, i))

        out_vars = []

        if formatting == "any":
            for _, t in std_bucket:
                out_vars.extend(t)
            for _, t in holo_bucket:
                out_vars.extend(t)
            for _, t in anti_bucket:
                out_vars.extend(t)
            for _, t in real_bucket:
                out_vars.extend(t)
            for _, t in imag_bucket:
                out_vars.extend(t)

        elif formatting == "complex":
            for _, t in std_bucket:
                out_vars.extend(t)
            for _, t in holo_bucket:
                out_vars.extend(t)
            for _, t in anti_bucket:
                out_vars.extend(t)

        else:  # "real"
            for _, t in std_bucket:
                out_vars.extend(t)
            for _, t in real_bucket:
                out_vars.extend(t)
            for _, t in imag_bucket:
                out_vars.extend(t)

        out_t = tuple(out_vars)

        if not return_dict:
            cache[key] = out_t
            return out_t

        loc: Dict[Any, Tuple[Any, int]] = {}
        for syslbl, info in sys_info_cache.items():
            if info.get("type") != "coordinate":
                if syslbl is None:
                    for atom in systems.get(None) or ():
                        loc[atom] = (syslbl, atom)
                continue
            flat = info.get("flattened_relatives", None)
            if not isinstance(flat, tuple) or not flat:
                continue

            rel = info.get("relatives") or {}
            sub = info.get("sub_type", None)

            if sub == "complex":
                h = rel.get("holo") or tuple()
                a = rel.get("anti") or tuple()
                r = rel.get("real") or tuple()
                i = rel.get("imag") or tuple()

                off_h = 0
                off_a = off_h + len(h)
                off_r = off_a + len(a)
                off_i = off_r + len(r)

                for j, v in enumerate(h):
                    loc[v] = (syslbl, off_h + j)
                for j, v in enumerate(a):
                    loc[v] = (syslbl, off_a + j)
                for j, v in enumerate(r):
                    loc[v] = (syslbl, off_r + j)
                for j, v in enumerate(i):
                    loc[v] = (syslbl, off_i + j)
            else:
                st = rel.get("standard") or flat
                if isinstance(st, tuple) and st:
                    for j, v in enumerate(st):
                        loc[v] = (syslbl, j)

        loc_out = {v: loc[v] for v in out_t if v in loc}
        out = (out_t, loc_out)
        cache[key] = out
        return out

    def infer_minimal_varSpace(
        self,
        *,
        return_dict: bool = False,
    ):
        vs_all, loc = self.infer_varSpace(formatting="any", return_dict=True)

        present = set()
        cd = getattr(self, "coeff_dict", None)
        if isinstance(cd, dict):
            for k in cd.keys():
                if not isinstance(k, tuple):
                    continue
                kl = len(k)
                if kl == 0 or kl % 3 != 0:
                    continue
                deg = kl // 3
                idxs = k[:deg]
                syslbls = k[2 * deg :]
                for idx, sys in zip(idxs, syslbls):
                    present.add((sys, idx))

        out = tuple(
            v
            for v in vs_all
            if (loc.get(v) in (None,)) is False and (loc[v][0], loc[v][1]) in present
        )

        if not return_dict:
            return out

        out_loc = {v: loc[v] for v in out if v in loc}
        return out, out_loc

    def _legacy_system_label(self):
        syslbl = None
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            if not isinstance(k, tuple) or len(k) == 0 or len(k) % 3 != 0:
                return None
            deg = len(k) // 3
            sys = k[2 * deg :]
            if not sys:
                continue
            s0 = sys[0]
            if any(s != s0 for s in sys):
                return None
            if syslbl is None:
                syslbl = s0
            elif syslbl != s0:
                return None
        return syslbl

    def _legacy_varSpace(self):
        syslbl = self._legacy_system_label()
        if syslbl is None:
            return None
        vs = self._variable_spaces.get(syslbl, None)
        if not isinstance(vs, tuple):
            return None
        if self.total_degree == 0:
            return vs
        n = len(vs)
        if n == 0:
            return None
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            deg = len(k) // 3
            idxs = k[:deg]
            if any((not isinstance(i, Integral)) or i < 0 or i >= n for i in idxs):
                return None
        return vs

    def _legacy_coeff_dict(self):
        vs = self._legacy_varSpace()
        if vs is None:
            return None
        if self.total_degree == 0:
            return {tuple(): self.coeff_dict.get(tuple(), 0)}
        out = {}
        deg = self.total_degree
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            idxs = k[:deg]
            out[tuple(idxs)] = v
        if not out:
            return {(0,) * deg: 0}
        return out

    def _require_legacy_view(self, what: str):
        if self._legacy_varSpace() is None:
            raise ValueError(
                f"{what} is only available when the tensor is supported on a single cached coordinate system."
            )

    @property
    def realVarSpace(self):
        if self.dgcvType == "standard":
            return self._realVarSpace
        if self._realVarSpace is None or self._imVarSpace is None:
            self.cd_formats
        if self._realVarSpace is None or self._imVarSpace is None:
            return None
        return self._realVarSpace + self._imVarSpace

    @property
    def holVarSpace(self):
        if self.dgcvType == "standard":
            return self._holVarSpace
        if self._holVarSpace is None:
            self.cd_formats
        return self._holVarSpace

    @property
    def antiholVarSpace(self):
        if self.dgcvType == "standard":
            return self._antiholVarSpace
        if self._antiholVarSpace is None:
            self.cd_formats
        return self._antiholVarSpace

    @property
    def compVarSpace(self):
        if self.dgcvType == "standard":
            if self._holVarSpace is None or self._antiholVarSpace is None:
                return None
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace is None or self._antiholVarSpace is None:
            self.cd_formats
        if self._holVarSpace is None or self._antiholVarSpace is None:
            return None
        return self._holVarSpace + self._antiholVarSpace

    def _merged_variable_spaces(self, other):
        out = dict(self._variable_spaces)
        ov = getattr(other, "_variable_spaces", {})
        for k, v in ov.items():
            if k in out:
                if k is None:
                    out[k] = out[None] | v
                elif out[k] != v:
                    raise ValueError(
                        f"Incompatible cached variable spaces {out[k]} and {v} for system '{k}'."
                    )
                continue
            out[k] = v
        return out
