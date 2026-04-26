import uuid

import pytest  # type: ignore

from dgcv._aux._backends._types_and_constants import imag_unit
from dgcv._aux._utilities._config import get_globals, get_variable_registry
from dgcv._aux._vmf.vmf import clear_vmf
from dgcv.core.dgcv_core.dgcv_core import createVariables


@pytest.fixture
def fresh_label():
    def make(prefix="test"):
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    return make


@pytest.fixture
def clean_vmf():
    clear_vmf(report=False)
    yield
    clear_vmf(report=False)


@pytest.fixture
def two_standard_systems(clean_vmf, fresh_label):
    x_label = fresh_label("x")
    y_label = fresh_label("y")

    createVariables(x_label, 2, withVF=True)
    createVariables(y_label, 2, withVF=True)

    g = get_globals()

    return {
        "x_label": x_label,
        "y_label": y_label,
        "x1": g[f"{x_label}1"],
        "x2": g[f"{x_label}2"],
        "y1": g[f"{y_label}1"],
        "y2": g[f"{y_label}2"],
        "Dx1": g[f"D_{x_label}1"],
        "Dx2": g[f"D_{x_label}2"],
        "Dy1": g[f"D_{y_label}1"],
        "Dy2": g[f"D_{y_label}2"],
        "dx1": g[f"d_{x_label}1"],
        "dx2": g[f"d_{x_label}2"],
        "dy1": g[f"d_{y_label}1"],
        "dy2": g[f"d_{y_label}2"],
    }


@pytest.fixture
def complex_system_factory(clean_vmf, fresh_label):
    def build_system(n=1, withVF=True):
        z_label = fresh_label("z")
        x_label = fresh_label("x")
        y_label = fresh_label("y")

        if n == 1:
            createVariables(z_label, x_label, y_label, withVF=withVF)
        else:
            createVariables(z_label, x_label, y_label, n, withVF=withVF)

        g = get_globals()

        def as_tuple(base_label, *, bar=False, D=False, d=False):
            prefix = ""
            if D:
                prefix = "D_"
            elif d:
                prefix = "d_"

            stem = f"BAR{base_label}" if bar else base_label

            if n == 1:
                return (g[f"{prefix}{stem}"],)
            return tuple(g[f"{prefix}{stem}{j}"] for j in range(1, n + 1))

        return {
            "imag": imag_unit(),
            "labels": {
                "z": z_label,
                "x": x_label,
                "y": y_label,
            },
            "z": as_tuple(z_label),
            "BARz": as_tuple(z_label, bar=True),
            "x": as_tuple(x_label),
            "y": as_tuple(y_label),
            "D_z": as_tuple(z_label, D=True),
            "D_BARz": as_tuple(z_label, bar=True, D=True),
            "D_x": as_tuple(x_label, D=True),
            "D_y": as_tuple(y_label, D=True),
            "d_z": as_tuple(z_label, d=True),
            "d_BARz": as_tuple(z_label, bar=True, d=True),
            "d_x": as_tuple(x_label, d=True),
            "d_y": as_tuple(y_label, d=True),
        }

    return build_system
