from numbers import Integral

from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import retrieve_passkey
from ..combinatorics.combinatorics import carProd
from ._indexing import _spool, _unspool


def _entry_equal(a, b):
    if a is b:
        return True
    if not (isinstance(a, _array_core) or isinstance(b, _array_core)):
        try:
            return _scalar_is_zero(a - b)
        except Exception:
            pass
    try:
        return bool(a == b)
    except Exception:
        return False


class _array_core:
    def __init__(
        self, array_data=None, *, shape=None, entry_rule=None, null_return=None
    ):
        if shape is not None:
            if isinstance(shape, (list, tuple)):
                shape = tuple(shape)
            else:
                raise TypeError(
                    "shape must be tuple-like (list/tuple) of non-negative ints"
                )

            if not all(isinstance(s, Integral) and s >= 0 for s in shape):
                raise TypeError(
                    "shape must be tuple-like (list/tuple) of non-negative ints"
                )

            self.shape = shape
        if array_data is None:
            if entry_rule is None:
                array_data = dict()
                self.shape = (0,)
            else:
                try:
                    assert shape is not None
                    array_dict = dict()
                    for idx in carProd(*[range(j) for j in shape]):
                        val = entry_rule(*idx)
                        if not _scalar_is_zero(val):
                            array_dict[tuple(idx)] = val
                    array_data = array_dict
                except Exception:
                    dgcv_warning(
                        "The given `entry_rule` and `shape` combination given to `array_dgcv` could not be processed. A trivial array was constructed instead."
                    )
                    array_data = dict()
                    self.shape = (0,)
        self._data, self.shape = self._normalize(array_data)
        self._data_unspooled_cache = None
        self.ndim = len(self.shape)
        self.null_return = 0 if null_return is None else null_return
        self._dgcv_class_check = retrieve_passkey()

    def _normalize(self, array_data):
        if isinstance(array_data, dict):
            shp = getattr(self, "shape", None)
            if not isinstance(shp, tuple):
                raise TypeError("dict input requires shape=... in array_dgcv(...)")

            flat = {}
            for k, v in array_data.items():
                if isinstance(k, tuple):
                    idx = _spool(k, shp)
                else:
                    idx = k

                if v is None:
                    flat[idx] = None
                    continue
                if _scalar_is_zero(v):
                    continue
                flat[idx] = v

            return flat, tuple(shp)
        if callable(array_data):
            shp = getattr(self, "shape", None)
            if not isinstance(shp, tuple):
                raise TypeError(
                    "callable entry rule requires shape=... in array_dgcv(...)"
                )
            flat = {}
            if not shp:
                v = array_data()
                if v is None:
                    flat[0] = None
                else:
                    if not _scalar_is_zero(v):
                        flat[0] = v
                return flat, tuple(shp)

            def _walk(prefix, rest):
                if not rest:
                    try:
                        v = array_data(*prefix)
                    except TypeError:  # expecting array_data to operate on a indices directly rather than indices packed into a tuple.
                        v = array_data(prefix)

                    idx = _spool(prefix, shp)

                    if v is None:
                        flat[idx] = None
                        return

                    if _scalar_is_zero(v):
                        return
                    flat[idx] = v
                    return

                for i in range(rest[0]):
                    _walk(prefix + (i,), rest[1:])

            _walk((), shp)
            return flat, tuple(shp)

        if isinstance(array_data, (list, tuple)):
            shape = self._infer_shape(array_data)
            flat = {}
            self._fill(flat, array_data, shape, shape_full=shape)
            return flat, shape

        if isinstance(array_data, array_dgcv):
            return dict(array_data._data), tuple(array_data.shape)

        rows = getattr(array_data, "rows", None)
        cols = getattr(array_data, "cols", None)
        if isinstance(rows, Integral) and isinstance(cols, Integral):
            shape = (rows, cols)
            flat = {}
            for i in range(rows):
                base = i * cols
                for j in range(cols):
                    flat[base + j] = array_data[i, j]
            return flat, shape

        nrows = getattr(array_data, "nrows", None)
        ncols = getattr(array_data, "ncols", None)
        if callable(nrows) and callable(ncols):
            r = int(nrows())
            c = int(ncols())
            shape = (r, c)
            flat = {}
            for i in range(r):
                base = i * c
                for j in range(c):
                    flat[base + j] = array_data[i, j]
            return flat, shape

        raise TypeError(f"Unsupported array_data type: {type(array_data)}")

    def slice(self, index_value_pairs):
        """
        The index_value_pairs should be a dict of whose
        keys are indices, and whose values is the index value. The
        pairs [(i_1,v_1), (i_2,v_2),...] indicate taking the cross
        section of the array where entry index i_j is fixed at
        i_j=v_j for all j.
        """
        former_shape = tuple(self.shape)
        remaining_indices = [
            j for j in range(len(former_shape)) if j not in index_value_pairs
        ]
        new_shape = tuple(former_shape[j] for j in remaining_indices)
        newarray = array_dgcv({}, shape=new_shape)
        for k, v in self._data.items():
            uk = self._unspool(k)
            nk = []
            for count, idx in enumerate(uk):
                if count not in index_value_pairs:
                    nk.append(idx)
                elif index_value_pairs[count] != idx:
                    nk = None
                    break
            if nk is not None:
                newarray[tuple(nk)] = v
        return newarray

    def __bool__(self):
        return bool(self._data)

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, _array_core):
            return NotImplemented
        if tuple(self.shape) != tuple(other.shape):
            return False
        for k in set(self._data) | set(other._data):
            if not _entry_equal(
                self._data.get(k, self.null_return),
                other._data.get(k, other.null_return),
            ):
                return False
        return True

    def _infer_shape(self, data):
        shape = []
        cur = data
        while isinstance(cur, (list, tuple)):
            shape.append(len(cur))
            children = [x for x in cur if isinstance(x, (list, tuple))]
            if not children:
                break
            cur = max(children, key=len)
        return tuple(shape)

    def _fill(self, flat, data, shape, shape_full, prefix=()):
        if not shape:
            flat[_spool(prefix, shape_full)] = data
            return
        if not isinstance(data, (list, tuple)):
            flat[_spool(prefix, shape_full)] = data
            return
        for i, val in enumerate(data):
            if not _scalar_is_zero(val) and val is not None:
                self._fill(flat, val, shape[1:], shape_full, prefix + (i,))

    def __getitem__(self, key):
        if isinstance(key, slice):
            rng = range(*key.indices(len(self)))
            new_data = {}

            for new_i, flat_i in enumerate(rng):
                if flat_i in self._data:
                    new_data[new_i] = self._data[flat_i]

            return self.__class__(
                new_data,
                shape=(len(rng),),
                null_return=self.null_return,
            )
        if isinstance(key, tuple) and any(isinstance(k, slice) for k in key):
            if len(key) == 1:
                k0 = key[0]
                if not isinstance(k0, slice):
                    raise TypeError("single-entry tuple slice key must contain a slice")

                rng = range(*k0.indices(len(self)))
                new_data = {}

                for new_i, flat_i in enumerate(rng):
                    if flat_i in self._data:
                        new_data[new_i] = self._data[flat_i]

                return self.__class__(
                    new_data,
                    shape=(len(rng),),
                    null_return=self.null_return,
                )

            if len(key) != self.ndim:
                raise IndexError(
                    f"slice tuple must have length 1 or array dimension = {len(self.shape)}"
                )
            axis_ranges = []
            new_shape = []

            for k, dim in zip(key, self.shape):
                if isinstance(k, slice):
                    rng = list(range(*k.indices(dim)))
                else:
                    if not isinstance(k, Integral):
                        raise TypeError(
                            "slice tuple entries must be integers or slice objects"
                        )
                    j = int(k)
                    if j < 0:
                        j += dim
                    if j < 0 or j >= dim:
                        raise IndexError("array index out of range")
                    rng = [j]

                axis_ranges.append(rng)
                new_shape.append(len(rng))

            axis_pos_maps = [
                {old: new for new, old in enumerate(rng)} for rng in axis_ranges
            ]

            new_data = {}
            for flat_idx, val in self._data.items():
                old_idx = _unspool(flat_idx, self.shape)

                if all(old_idx[d] in axis_pos_maps[d] for d in range(self.ndim)):
                    new_idx = tuple(
                        axis_pos_maps[d][old_idx[d]] for d in range(self.ndim)
                    )
                    new_flat = _spool(new_idx, tuple(new_shape))
                    new_data[new_flat] = val

            return self.__class__(
                new_data,
                shape=tuple(new_shape),
                null_return=self.null_return,
            )
        idx = _spool(key, self.shape) if isinstance(key, tuple) else key
        return self._data.get(idx, self.null_return)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            idx, tkey = _spool(key, self.shape), key
        else:
            idx, tkey = key, None
        self._data[idx] = value
        if self._data_unspooled_cache is not None:
            if tkey is None:
                tkey = _unspool(idx, self.shape)
            self._data_unspooled_cache[tkey] = value

    def _spool(self, key):
        return _spool(key, self.shape)

    def _unspool(self, key):
        return _unspool(key, self.shape)

    @property
    def _data_unspooled(self):
        if self._data_unspooled_cache is None:
            self._data_unspooled_cache = {
                _unspool(key, self.shape): value for key, value in self._data.items()
            }
        return self._data_unspooled_cache

    def __len__(self):
        total = 1
        for s in self.shape:
            total *= s
        return total

    def __iter__(self):
        shp = getattr(self, "shape", None)
        if not isinstance(shp, tuple):
            raise TypeError("array_dgcv is missing a valid shape")

        if not shp:
            yield self._data.get(0, self.null_return)
            return

        n = 1
        for s in shp:
            n *= s

        d = self._data
        for k in range(n):
            yield d.get(k, self.null_return)

    def iter_nonzero_items(self, *, include_zeros=False, include_none=False):
        shp = getattr(self, "shape", None)
        if not isinstance(shp, tuple):
            raise TypeError("array_dgcv is missing a valid shape")

        for k, v in self._data.items():
            if v is None:
                if include_none:
                    yield (_unspool(k, shp), v)
                continue

            if (not include_zeros) and _scalar_is_zero(v):
                continue

            yield (_unspool(k, shp), v)
