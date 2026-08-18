from types import MappingProxyType


class _frozen_array:
    def __init__(
        self, array_data=None, *, shape=None, entry_rule=None, null_return=None
    ):
        super().__init__(
            array_data=array_data,
            shape=shape,
            entry_rule=entry_rule,
            null_return=null_return,
        )
        self._freeze()

    def _freeze(self):
        self._data = MappingProxyType(dict(self._data))
        self._is_frozen = True

    def __setitem__(self, key, value):
        raise TypeError(f"{self.__class__.__name__} is immutable")

    def __delitem__(self, key):
        raise TypeError(f"{self.__class__.__name__} is immutable")

    def __setattr__(self, name, value):
        if getattr(self, "_is_frozen", False):
            raise TypeError(f"{self.__class__.__name__} is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self):
        nr = (
            f",\n null_return={repr(self.null_return)}" if self.null_return != 0 else ""
        )
        return f"frozen_array_dgcv(\n {repr(dict(self._data))},\n shape = {self.shape}{nr}\n )"

    def apply(self, func, *, in_place=False, skip_none=True, default=None, **kwargs):
        if in_place:
            raise TypeError(f"{self.__class__.__name__} does not support in_place=True")
        return super().apply(
            func,
            in_place=False,
            skip_none=skip_none,
            default=default,
            **kwargs,
        )

    @property
    def _data_unspooled(self):
        cache = getattr(self, "_data_unspooled_cache", None)
        if cache is None:
            object.__setattr__(
                self,
                "_data_unspooled_cache",
                {self._unspool(key): value for key, value in self._data.items()},
            )
            cache = self._data_unspooled_cache
        return MappingProxyType(cache)

    def __getstate__(self):
        return {
            "_data": dict(self._data),
            "_is_frozen": self._is_frozen,
            "shape": self.shape,
            "ndim": self.ndim,
            "null_return": self.null_return,
            "_dgcv_class_check": self._dgcv_class_check,
            "_data_unspooled_cache": (
                dict(self._data_unspooled_cache)
                if self._data_unspooled_cache is not None
                else None
            ),
        }

    def __setstate__(self, state):
        object.__setattr__(self, "_is_frozen", False)
        object.__setattr__(self, "shape", state["shape"])
        object.__setattr__(self, "ndim", state["ndim"])
        object.__setattr__(self, "null_return", state["null_return"])
        object.__setattr__(self, "_dgcv_class_check", state["_dgcv_class_check"])
        object.__setattr__(self, "_data", MappingProxyType(state["_data"]))
        object.__setattr__(
            self, "_data_unspooled_cache", state["_data_unspooled_cache"]
        )
        object.__setattr__(self, "_is_frozen", state["_is_frozen"])


def freeze_array(array):
    if isinstance(array, frozen_array_dgcv):
        return array
    if not isinstance(array, array_dgcv):
        raise TypeError("freeze_array expects an array_dgcv instance")
    return frozen_array_dgcv(
        dict(array._data),
        shape=tuple(array.shape),
        null_return=array.null_return,
    )
