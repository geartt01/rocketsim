"""Stage 7: RocketSim against OpenRocket's exported reference data (reference/openrocket/)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("compare", ROOT / "validation" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


@pytest.fixture(scope="module")
def results():
    calm, wind = compare.load_reference("calm"), compare.load_reference("wind2")
    cd = compare.cd_table(calm)
    rocket, table = compare.fly(cd)
    _, own = compare.fly()
    _, table_wind = compare.fly(cd, wind_speed=2.0)
    return {"table": compare.errors(table.summary, compare.reference_summary(calm)),
            "own": compare.errors(own.summary, compare.reference_summary(calm)),
            "table_wind": compare.errors(table_wind.summary, compare.reference_summary(wind)),
            "aero": compare.aero_comparison(rocket, calm)}


def test_apogee_within_5_percent_with_openrocket_cd(results):  # PLAN.md acceptance target
    assert abs(results["table"]["apogee"]["percent"]) < 5
    assert abs(results["table"]["apogee"]["percent"]) < 0.5  # what it actually achieves


def test_apogee_within_10_percent_with_own_aero(results):  # PLAN.md acceptance target
    assert abs(results["own"]["apogee"]["percent"]) < 10


def test_dynamics_match_openrocket_with_its_cd(results):
    for key in ("max_velocity", "time_from_ignition", "max_accel_g"):
        assert abs(results["table"][key]["percent"]) < 1


def test_wind_drift_matches_openrocket(results):
    assert abs(results["table_wind"]["drift"]["percent"]) < 5


def test_cp_within_2_percent_of_openrocket(results):  # PLAN.md Stage 2 test
    for row in results["aero"]:
        assert row["cp_ours"] == pytest.approx(row["cp_openrocket"], rel=0.02)
