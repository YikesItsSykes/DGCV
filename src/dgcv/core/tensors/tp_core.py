from typing import List

from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
)
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import retrieve_passkey
from ..dgcv_core.spaces.spaces import _vs_card
from .coefficients import _process_coeffs_dict, _process_coeffs_dict_trusted
from .promotion import _card_root_map, _promote_keys


class _tp_core:
    def __init__(
        self,
        coeff_dict,
        shape: List[str] = None,
        _process_shape_with_accumulation=True,
        _amb_prom=False,
        _hom_id=None,
        _hom_decomp=None,
        _trusted=False,
        _prebuilt=None,
    ):
        if _prebuilt is not None:
            result = _prebuilt
            (
                processed_coeff_dict,
                max_degree,
                min_degree,
                vs_id,
            ) = result
            self._init_state(
                processed_coeff_dict,
                max_degree,
                min_degree,
                vs_id,
                shape,
                _hom_id,
                _hom_decomp,
            )
            return
        if not isinstance(coeff_dict, dict):
            raise ValueError("Coefficient dictionary must be a dictionary.")
        processor = _process_coeffs_dict_trusted if _trusted else _process_coeffs_dict
        try:
            result = processor(
                coeff_dict,
                shape=shape,
                _amb_prom=_amb_prom,
                _process_shape_with_accumulation=_process_shape_with_accumulation,
            )
        except ValueError as ve:
            print(f"ValueError: {ve}")
            dgcv_warning(
                "Check the return statement of _process_coeffs_dict",
                wc_label="debug_log",
            )
            raise
        except TypeError as te:
            print(f"ValueError: {te}")
            dgcv_warning(
                "Check that _process_coeffs_dict returned a tuple",
                wc_label="debug_log",
            )
            raise
        except Exception as e:
            print(f"ValueError: {e}")
            raise
        (
            processed_coeff_dict,
            max_degree,
            min_degree,
            vs_id,
        ) = result
        self._init_state(
            processed_coeff_dict,
            max_degree,
            min_degree,
            vs_id,
            shape,
            _hom_id,
            _hom_decomp,
        )

    def _init_state(
        self,
        processed_coeff_dict,
        max_degree,
        min_degree,
        vs_id,
        shape,
        _hom_id,
        _hom_decomp,
    ):
        shape = (
            "custom"
            if callable(shape)
            else "all"
            if max_degree < 2
            else {
                "skew": "skew",
                "symmetric": "symmetric",
                "general": "general",
                "all": "all",
            }.get(shape, "general")
        )  # custom shapes are not fully supported yet
        self.shape = shape
        self.vs_id = vs_id
        self.vector_space = (
            vs_id[0].space if len(vs_id) > 0 else None
        )  ### deprecate soon
        self.coeff_dict = processed_coeff_dict
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.homogeneous = min_degree == max_degree
        self._card_map = None
        self._homogeneous_dicts = None
        self._weights = None
        self._homogeneous_components = None
        self._free_symbols = None
        self._leading_valence = None
        self._trailing_valence = None
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "tensorProduct"
        self._terms = [self] if len(self.coeff_dict) == 1 else None
        self._properties = {}
        if _hom_id:
            self._properties["_hom_id"] = (
                _hom_id  # _hom_id format: [{source_label,{y:x for x,y in zip(coeffs,target_labels)}}, label]
            )
        if _hom_decomp:
            self._properties["_hom_decomp"] = _hom_decomp

    def _promoted(self, targets):
        if not targets:
            return self
        return tensorProduct(
            dict(_promote_keys(self.coeff_dict, targets)),
            shape=self.shape,
            _trusted=True,
        )

    @property
    def _card_by_root(self):
        if self._card_map is None:
            self._card_map = _card_root_map(self.vs_id)
        return self._card_map

    @property
    def _vs_spring(self):
        return self._card_by_root.keys()

    @property
    def _unpromoted_spring(self):
        return {
            card: root for root, card in self._card_by_root.items() if card is not root
        }

    @property
    def homogeneous_dicts(self):
        if self._homogeneous_dicts is None:
            hd = {}
            for k, v in self.coeff_dict.items():
                deg = len(k)
                d = hd.get(deg)
                if d is None:
                    hd[deg] = {k: v}
                else:
                    d[k] = v
            self._homogeneous_dicts = hd
        return self._homogeneous_dicts

    @property
    def leading_valence(self):
        if self._leading_valence is not None:
            return self._leading_valence
        lv = set()
        for k in self.coeff_dict:
            if k:
                lv = lv | {k[0][1]}
        if len(lv) == 1:
            self._leading_valence = list(lv)[0]
        else:
            self._leading_valence = -1  # denoting exceptional cases
        return self._leading_valence

    @property
    def trailing_valence(self):
        if self._trailing_valence is not None:
            return self._trailing_valence
        lv = set()
        for k in self.coeff_dict:
            if k:
                lv = lv | {k[-1][1]}
        if len(lv) == 1:
            self._trailing_valence = list(lv)[0]
        else:
            self._trailing_valence = -1  # denoting exceptional cases
        return self._trailing_valence

    @property
    def homogeneous_components(self):
        if self._homogeneous_components is None:
            self._homogeneous_components = [
                tensorProduct(cd) for cd in self.homogeneous_dicts.values()
            ]
        return self._homogeneous_components

    @property
    def free_vectors(self):
        vec_idx = set()
        for key in self.coeff_dict:
            for factor in key:
                vec_idx.add((factor[0], factor[2]))
        return set(
            card.space.basis[idx]
            for idx, card in vec_idx
            if type(card) is _vs_card and card.space is not None
        )

    @property
    def coeffs(self):
        return tuple(self.coeff_dict.values())

    @property
    def terms(self):
        if self._terms is None:
            tList = []
            for k, v in self.coeff_dict.items():
                if _scalar_is_zero(v):
                    continue
                tList.append(tensorProduct({k: v}))
            self._terms = tList if len(tList) > 0 else [self]
        return self._terms

    def _convert_to_tp(self):
        return self

    @property
    def is_zero(self):
        for j in self.coeff_dict.values():
            if not _scalar_is_zero(simplify(j)):
                return False
        return True

    @property
    def is_literal_zero(self):
        for j in self.coeff_dict.values():
            if not _scalar_is_zero(j):
                return False
        return True

    def dual(self):
        def keyflip(key):
            return tuple((idx, 1 - valence, card) for idx, valence, card in key)

        return tensorProduct({keyflip(k): v for k, v in self.coeff_dict.items()})

    def __xor__(self, other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    @property
    def ambient_rep(self):
        return tensorProduct(self.coeff_dict, _amb_prom=True)

    @property
    def free_symbols(self):
        if self._free_symbols is None:
            fs = set()
            for c in self.coeff_dict.values():
                fs |= get_free_symbols(c)
            self._free_symbols = fs
        return self._free_symbols
