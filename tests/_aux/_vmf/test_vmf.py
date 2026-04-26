from dgcv._aux._utilities._config import get_variable_registry
from dgcv._aux._vmf.vmf import clear_vmf, clearVar, listVar, vmf_lookup
from dgcv.core.dgcv_core.dgcv_core import createVariables


def assert_registry_empty():
    registry = get_variable_registry()

    assert registry["standard_variable_systems"] == {}
    assert registry["complex_variable_systems"] == {}
    assert registry["finite_algebra_systems"] == {}
    assert registry["misc"] == {}

    assert registry["eds"]["atoms"] == {}
    assert registry["eds"]["coframes"] == {}

    assert registry["protected_variables"] == set()
    assert registry["temporary_variables"] == set()
    assert registry["obscure_variables"] == set()

    assert registry["dgcv_enforced_real_atoms"] == {}
    assert registry["_labels"] == {}
    assert registry["paths"] == {}

    for name, mapping in registry["conversion_dictionaries"].items():
        assert dict(mapping) == {}, f"conversion_dictionaries[{name}] was not empty"


# --- baseline / cleanup -----------------------------------------------------


def test_clean_vmf_starts_empty(clean_vmf):
    assert_registry_empty()


def test_clear_vmf_resets_registry_for_standard_variables(clean_vmf, fresh_label):
    label = fresh_label("std")

    createVariables(label, number_of_variables=2, withVF=False)

    registry = get_variable_registry()
    assert label in registry["standard_variable_systems"]

    family_names = registry["standard_variable_systems"][label]["family_names"]
    assert len(family_names) == 2
    assert family_names == (f"{label}1", f"{label}2")

    clear_vmf(report=False)

    assert_registry_empty()


# --- listing ----------------------------------------------------------------


def test_listVar_includes_created_standard_system(clean_vmf, fresh_label):
    label = fresh_label("std")

    createVariables(label, number_of_variables=2, withVF=False)

    labels = listVar()
    assert label in labels

    standard_labels = listVar(standard_only=True)
    assert label in standard_labels


def test_listVar_temporary_only_filters_standard_systems(clean_vmf, fresh_label):
    temp_label = fresh_label("tmp")
    regular_label = fresh_label("std")

    createVariables(temp_label, withVF=False, temporary_variables=True)
    createVariables(regular_label, withVF=False)

    temp_labels = listVar(temporary_only=True)

    assert temp_label in temp_labels
    assert regular_label not in temp_labels


# --- lookup -----------------------------------------------------------------


def test_vmf_lookup_unregistered_string_returns_unregistered(clean_vmf, fresh_label):
    label = fresh_label("missing")

    info = vmf_lookup(label)

    assert info["type"] == "unregistered"
    assert info["sub_type"] is None


def test_vmf_lookup_standard_parent_returns_expected_metadata(clean_vmf, fresh_label):
    label = fresh_label("std")

    createVariables(label, number_of_variables=2, withVF=False)

    info = vmf_lookup(label, path=True, relatives=True)

    assert info["type"] == "coordinate"
    assert info["sub_type"] == "standard"
    assert info["path"] == ("standard_variable_systems", label)
    assert info["relatives"]["system_label"] == label


def test_vmf_lookup_standard_child_reports_system_index(clean_vmf, fresh_label):
    label = fresh_label("std")

    createVariables(label, number_of_variables=3, withVF=False)

    registry = get_variable_registry()
    system = registry["standard_variable_systems"][label]
    child_obj = system["family_values"][1]

    info = vmf_lookup(child_obj, path=True, relatives=True, system_index=True)

    assert info["type"] == "coordinate"
    assert info["sub_type"] == "standard"
    assert info["path"] == ("standard_variable_systems", label, "family_values", 1)
    assert info["system_index"] == 1
    assert info["relatives"]["system_label"] == label


# --- clearing ----------------------------------------------------------------


def test_clearVar_removes_standard_system_by_parent_label(clean_vmf, fresh_label):
    label = fresh_label("std")

    createVariables(label, number_of_variables=2, withVF=False)

    registry = get_variable_registry()
    assert label in registry["standard_variable_systems"]

    clearVar(label, report=False)

    registry = get_variable_registry()
    assert label not in registry["standard_variable_systems"]
    assert label not in registry["_labels"]
    assert label not in registry["paths"]


def test_clearVar_removes_standard_system_by_child_label(clean_vmf, fresh_label):
    label = fresh_label("std")

    createVariables(label, number_of_variables=2, withVF=False)

    registry = get_variable_registry()
    child_name = registry["standard_variable_systems"][label]["family_names"][0]

    clearVar(child_name, report=False)

    registry = get_variable_registry()
    assert label not in registry["standard_variable_systems"]
    assert child_name not in registry["paths"]
    assert label not in listVar()
