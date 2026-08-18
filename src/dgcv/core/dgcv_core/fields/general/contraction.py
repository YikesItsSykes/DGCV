from __future__ import annotations

from ....._aux._backends._types_and_constants import check_dgcv_scalar
from ....._aux._utilities._config import get_dgcv_settings_registry
from ....._aux._vmf._safeguards import check_dgcv_category, get_dgcv_category


class _tensor_field_contraction:
    def __call__(self, *args, strict_left_to_right: bool = False):
        from . import tensor_field_class

        if len(args) == 0:
            return self
        if len(args) > 1:
            contracted = self.__call__(args[0])
            if contracted == 0:
                return 0
            if get_dgcv_category(contracted) == "tensor_field":
                coerce = getattr(contracted, "as_tensor_field", None)
                if callable(coerce):
                    contracted = coerce()
                return contracted(*args[1:])
            first_tf = None
            for idx, arg in enumerate(args[1:]):
                if get_dgcv_category(arg) == "tensor_field":
                    first_tf = arg
                    new_args = args[idx + 2 :]
                    break
                elif check_dgcv_scalar(arg):
                    contracted *= arg
                else:
                    raise TypeError(
                        "tensor_field_class only contracts with dgcv tensor_field classes and scalars, "
                        f"not {type(arg).__name__}."
                    )
            if first_tf:
                tail = first_tf(*new_args) if new_args else first_tf
                return contracted * tail
            return contracted

        other = args[0]
        if check_dgcv_category(other):
            coerce = getattr(other, "as_tensor_field", None)
            if callable(coerce):
                other = coerce()

        if get_dgcv_category(other) == "tensor_field":
            if self._validated_format == "complex":
                if other._validated_format in {"mixed", "real"}:
                    return self.__call__(other._to_complex())
            elif self._validated_format == "real":
                if other._validated_format in {"mixed", "complex"}:
                    return self.__call__(other._to_real())
            elif self._validated_format == "mixed":
                if other._validated_format == "real":
                    return self.__call__(other._to_real())
                if other._validated_format == "complex":
                    return self.__call__(other._to_complex())
                if other._validated_format == "mixed":
                    pref = get_dgcv_settings_registry()["preferred_variable_format"]
                    if pref == "real":
                        return self._to_real().__call__(other._to_real())
                    else:
                        return self._to_complex().__call__(other._to_complex())
            if self.data_shape == "symmetric" or self.data_shape == "skew":
                scale = -1 if self.data_shape == "skew" else 1

                shape_cd = {}
                abort = False

                for k2, v2 in other.coeff_dict.items():
                    if not v2:
                        continue

                    lk2 = len(k2)
                    if lk2 not in (0, 3):
                        abort = True
                        break

                    if lk2 == 0:
                        for k1, v1 in self.coeff_dict.items():
                            if not v1:
                                continue
                            shape_cd[k1] = shape_cd.get(k1, 0) + v1 * v2
                        continue

                    a, b, c = k2
                    for k1, v1 in self.coeff_dict.items():
                        if not v1:
                            continue
                        sign = 1
                        deg = len(k1) // 3
                        for idx in range(deg):
                            idx2, idx3 = deg + idx, 2 * deg + idx
                            if a == k1[idx] and b + k1[idx2] == 1 and c == k1[idx3]:
                                new_key = tuple(
                                    elem
                                    for count, elem in enumerate(k1)
                                    if count not in (idx, idx2, idx3)
                                )
                                shape_cd[new_key] = (
                                    shape_cd.get(new_key, 0) + sign * v1 * v2
                                )
                                break
                            sign *= scale

                if not abort:
                    if not shape_cd:
                        return 0
                    if tuple() in shape_cd and len(shape_cd) == 1:
                        return shape_cd[tuple()]

                    return tensor_field_class(
                        coeff_dict=shape_cd,
                        data_shape=self.data_shape,
                        dgcvType=self.dgcvType,
                        _simplifyKW=self._simplifyKW,
                        variable_spaces=self._merged_variable_spaces(other),
                        parameters=self.parameters
                        | getattr(other, "parameters", set()),
                    )
            if self.data_shape == "general":
                gen_cd = {}
                abort = False

                for k2, v2 in other.coeff_dict.items():
                    if not v2:
                        continue

                    lk2 = len(k2)
                    if lk2 not in (0, 3):
                        abort = True
                        break

                    if lk2 == 0:
                        for k1, v1 in self.coeff_dict.items():
                            if not v1:
                                continue
                            gen_cd[k1] = gen_cd.get(k1, 0) + v1 * v2
                        continue

                    a, b, c = k2
                    for k1, v1 in self.coeff_dict.items():
                        if not v1:
                            continue
                        deg = len(k1) // 3
                        idx, idx2, idx3 = 0, deg, 2 * deg
                        if a == k1[idx] and b + k1[idx2] == 1 and c == k1[idx3]:
                            new_key = tuple(
                                elem
                                for count, elem in enumerate(k1)
                                if count not in (idx, idx2, idx3)
                            )
                            gen_cd[new_key] = gen_cd.get(new_key, 0) + v1 * v2

                if not abort:
                    if not gen_cd:
                        return 0
                    if tuple() in gen_cd and len(gen_cd) == 1:
                        return gen_cd[tuple()]

                    return tensor_field_class(
                        coeff_dict=gen_cd,
                        data_shape="general",
                        dgcvType=self.dgcvType,
                        _simplifyKW=self._simplifyKW,
                        variable_spaces=self._merged_variable_spaces(other),
                        parameters=self.parameters
                        | getattr(other, "parameters", set()),
                    )

        if get_dgcv_category(other) != "tensor_field":
            raise TypeError(
                "tensor_field_class.__call__ only supports contraction against tensor_field instances."
            )

        shape_a = getattr(self, "data_shape", "general")
        shape_b = getattr(other, "data_shape", "general")
        if shape_a != shape_b and shape_a != "all" and shape_b != "all":
            ###!!! optimize later: shape-aware contraction can avoid general expansion by sorting argument keys
            self = self._coerce_to_general()
            other = other._coerce_to_general()

        def _split_tripled(k):
            d = len(k) // 3
            return k[:d], k[d : 2 * d], k[2 * d :]

        def _join_tripled(idxs, valence_tuple, syslbls):
            return tuple(idxs) + tuple(valence_tuple) + tuple(syslbls)

        def _complementary(vb1, vb2):
            for a, b in zip(vb1, vb2):
                if a + b != 1:
                    return False
            return True

        vst = self.variable_spaces_types

        def _profile_from_parts(idxs, syslbls):
            out = []
            for idx, sys in zip(idxs, syslbls):
                sys_data = vst.get(sys)
                if sys_data is None or sys_data.get("type") != "complex":
                    out.append("s")
                    continue
                b0, b1, b2 = sys_data["breaks"]
                if idx < b1:
                    out.append("c")
                elif idx < b2:
                    out.append("r")
                else:
                    out.append("s")
            return tuple(out)

        def _expand_to_profile(k, v, want_profile):
            have = self._profile_key(k)
            have0 = tuple(t[0] for t in have)
            want0 = tuple(t[0] for t in want_profile)

            for a0, b0 in zip(have0, want0):
                if (a0 == "s") != (b0 == "s"):
                    return None

            swap_slots = [i for i, (a0, b0) in enumerate(zip(have0, want0)) if a0 != b0]
            if not swap_slots:
                return {k: v}

            flip = getattr(self, "_flip_format", None)
            if not callable(flip):
                return None

            terms = {k: v}
            for slot in swap_slots:
                kind0 = want0[slot]
                if kind0 not in ("r", "c"):
                    return None

                new_terms = {}
                for kk, vv in terms.items():
                    out = flip(slot, kk, vv, to_kind=kind0)
                    if not out:
                        continue
                    for nk, nv in out.items():
                        if nv:
                            new_terms[nk] = new_terms.get(nk, 0) + nv

                terms = new_terms
                if not terms:
                    return None

            return terms

        def _pick_by_idxs(terms, want_idxs):
            out = {}
            for k, v in terms.items():
                if not v:
                    continue
                idxs, _, _ = _split_tripled(k)
                if idxs == want_idxs:
                    out[k] = out.get(k, 0) + v
            return out or None

        new_cd = {}

        for k1, v1 in self.coeff_dict.items():
            if not v1:
                continue

            i1, val1, s1 = _split_tripled(k1)
            d1 = len(i1)

            for k2, v2 in other.coeff_dict.items():
                if not v2:
                    continue

                i2, val2, s2 = _split_tripled(k2)
                d2 = len(i2)

                strict_left_to_right = (
                    True  # hard override -- releasing this has not been decided
                )
                if d2 > d1:
                    if strict_left_to_right:
                        continue
                    lead_i2, tail_i2 = i2[:d1], i2[d1:]
                    lead_val2, tail_val2 = val2[:d1], val2[d1:]
                    lead_s2, tail_s2 = s2[:d1], s2[d1:]

                    if lead_s2 != s1:
                        continue
                    if not _complementary(val1, lead_val2):
                        continue

                    want_profile = _profile_from_parts(lead_i2, lead_s2)

                    terms = _expand_to_profile(k1, v1, want_profile)
                    if not terms:
                        continue
                    terms = _pick_by_idxs(terms, lead_i2)
                    if not terms:
                        continue

                    nk = _join_tripled(tail_i2, tail_val2, tail_s2)
                    acc = new_cd.get(nk, 0)
                    for v1a in terms.values():
                        acc += v1a * v2
                    if acc:
                        new_cd[nk] = acc
                    continue

                lead_i1, tail_i1 = i1[:d2], i1[d2:]
                lead_val1, tail_val1 = val1[:d2], val1[d2:]
                lead_s1, tail_s1 = s1[:d2], s1[d2:]

                if lead_s1 != s2:
                    continue
                if not _complementary(lead_val1, val2):
                    continue

                want_profile = _profile_from_parts(lead_i1, lead_s1)

                terms = _expand_to_profile(k2, v2, want_profile)
                if not terms:
                    continue
                terms = _pick_by_idxs(terms, lead_i1)
                if not terms:
                    continue

                nk = _join_tripled(tail_i1, tail_val1, tail_s1)
                acc = new_cd.get(nk, 0)
                for v2a in terms.values():
                    acc += v1 * v2a
                if acc:
                    new_cd[nk] = acc

        if not new_cd:
            return 0

        if tuple() in new_cd and len(new_cd) == 1:
            return new_cd[tuple()]

        return tensor_field_class(
            coeff_dict=new_cd,
            data_shape="general",
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._merged_variable_spaces(other),
        )
