from dgcv.core.base import dgcv_class

# --- helpers -----------------------------------------------------------------


class plain_dgcv(dgcv_class):
    pass


class latex_only_dgcv(dgcv_class):
    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return "x+y"


class repr_latex_only_dgcv(dgcv_class):
    def _repr_latex_(self, raw: bool = False, **kwargs):
        s = "x^2"
        return s if raw else f"$\\displaystyle {s}$"


class simplifiable_dgcv(dgcv_class):
    def _eval_simplify(self, **kwargs):
        return "simplified"


# --- simplify ----------------------------------------------------------------


def test_dgcv_class_default_simplify_returns_self():
    obj = plain_dgcv()
    assert obj.__dgcv_simplify__() is obj


def test_dgcv_class_delegates_simplify_to_eval_simplify():
    obj = simplifiable_dgcv()
    assert obj.__dgcv_simplify__() == "simplified"


# --- string / repr -----------------------------------------------------------


def test_dgcv_class_default_str_uses_class_name():
    obj = plain_dgcv()
    assert str(obj) == "<plain_dgcv>"


def test_dgcv_class_repr_uses_str_when_print_style_readable(monkeypatch):
    monkeypatch.setattr(
        "dgcv.core.base.get_dgcv_settings_registry",
        lambda: {"print_style": "readable"},
    )
    obj = plain_dgcv()
    assert repr(obj) == "<plain_dgcv>"


def test_dgcv_class_repr_falls_back_to_object_repr_when_not_readable(monkeypatch):
    monkeypatch.setattr(
        "dgcv.core.base.get_dgcv_settings_registry",
        lambda: {"print_style": "plain"},
    )
    obj = plain_dgcv()
    r = repr(obj)
    assert isinstance(r, str)
    assert "plain_dgcv" in r


# --- latex behavior ----------------------------------------------------------


def test_dgcv_class_latex_falls_back_to_str_when_no_latex_methods_are_overridden():
    obj = plain_dgcv()
    assert obj._latex() == "<plain_dgcv>"


def test_dgcv_class_repr_latex_falls_back_to_str_when_no_latex_methods_are_overridden():
    obj = plain_dgcv()
    assert obj._repr_latex_() == "<plain_dgcv>"


def test_dgcv_class_repr_latex_wraps_latex_from_overridden_latex_method():
    obj = latex_only_dgcv()
    assert obj._repr_latex_(raw=True) == "x+y"
    assert obj._repr_latex_() == "$\\displaystyle x+y$"


def test_dgcv_class_latex_uses_overridden_repr_latex_when_available():
    obj = repr_latex_only_dgcv()
    assert obj._latex(raw=True) == "x^2"


# --- mimebundle --------------------------------------------------------------


def test_dgcv_class_repr_mimebundle_includes_plain_text():
    obj = plain_dgcv()
    bundle = obj._repr_mimebundle_()

    assert "text/plain" in bundle
    assert bundle["text/plain"] == repr(obj)


def test_dgcv_class_repr_mimebundle_includes_latex_when_available():
    obj = latex_only_dgcv()
    bundle = obj._repr_mimebundle_()

    assert "text/plain" in bundle
    assert "text/latex" in bundle
    assert bundle["text/latex"] == "$\\displaystyle x+y$"
