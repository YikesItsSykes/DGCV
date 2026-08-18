from ..._aux._backends._engine import engine_kind, engine_module
from ..._aux._utilities._config import dgcv_warning


class _matrix_engine:
    def _to_engine_matrix(self, kind: str | None = None):
        if kind is None:
            kind = engine_kind()
        if kind not in ("sage", "sympy"):
            raise RuntimeError(f"Unsupported engine kind {kind!r}")

        rep = self._engine_representation.get(kind, None)
        if rep is not None:
            return rep

        mod = engine_module()
        if mod is None:
            raise RuntimeError("No symbolic engine is available.")

        rows = [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]

        if kind == "sage":
            M = mod.matrix(rows)
        else:
            M = mod.Matrix(rows)

        self._engine_representation[kind] = M
        return M

    def _eigenvals_dict_by_engine(self, *, kind: str | None = None) -> dict:
        if kind is None:
            kind = engine_kind()
        M = self._to_engine_matrix(kind=kind)

        if kind == "sage":
            vals = M.eigenvalues()
            out = {}
            for v in vals:
                out[v] = out.get(v, 0) + 1
            return out

        ev = M.eigenvals()
        if isinstance(ev, dict):
            return dict(ev)
        out = {}
        try:
            for v in list(ev):
                out[v] = out.get(v, 0) + 1
        except Exception:
            pass
        return out

    def _eigenvects_by_engine(self, *, kind: str | None = None):
        if kind is None:
            kind = engine_kind()
        M = self._to_engine_matrix(kind=kind)

        if kind == "sage":
            data = M.eigenvectors_right()
            out = []
            for lam, vecs, mult in data:
                vv = []
                for v in vecs:
                    vv.append(matrix_dgcv.col_vector(list(v)))
                out.append((lam, int(mult), vv))
            return out

        data = M.eigenvects()
        out = []
        for lam, mult, vecs in data:
            vv = []
            for v in vecs:
                vv.append(matrix_dgcv(v))
            out.append((lam, int(mult), vv))
        return out

    def symbolic_engine_method(
        self,
        method: str,
        method_arguments=None,
        method_keywords=None,
        return_anything: bool = False,
    ):
        """
        Accepts a method name for matrix classes from whichever symbolic engine is active (set in the dgcv settings registry). Attempts to apply that method and if the result is matrix-like then it is converted back into a `matrix_dgcv` class, which is returned. Will return unaltered matrix if the attempt fails, along with a warning that this happened.

        Setting `return_anything` to True will return exactly what the method evaluates to rather than trying to coerce results into dgcv classes. If the given name points to an attribute or property rather than a method, then the result is infered from that property value.

        As dgcv is compatible with multiple symbolic engines, of which each have their respective names for matrix class methods, supported method names is fully reliant on whichever symbolic engine is selected via dgcv settings (use `set_dgcv_settings()` to change it).
        """
        value = getattr(self._to_engine_matrix(), method)
        args = [] if method_arguments is None else method_arguments
        kwds = {} if method_keywords is None else method_keywords
        if callable(value):
            value = value(*args, **kwds)
        if return_anything:
            return value
        try:
            return matrix_dgcv(value)
        except Exception:
            dgcv_warning(
                "Requested method name either does not exist for the current symbolic engine's matrix class or it does not return something `dgcv` supports as a matrix-like value."
            )
            return self

    def to_symbolic_engine_class(self):
        return self._to_engine_matrix()
