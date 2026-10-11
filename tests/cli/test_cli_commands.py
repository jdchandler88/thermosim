"""Command-line interface."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from thermosim.cli import main
from thermosim.config.io import example_scenarios, save_scenario, scenario_to_toml
from thermosim.recording import load_run


@pytest.fixture(scope="module")
def pot_toml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("sc") / "pot.toml"
    save_scenario(example_scenarios()["small_pot_stove"], path)
    return path


@pytest.fixture(scope="module")
def run_dir(pot_toml: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("run") / "out"
    code = main(["run", str(pot_toml), "--out", str(out), "--cell-size", "0.02", "--quiet"])
    assert code == 0
    return out


@pytest.mark.parametrize("cmd", [[], ["run"], ["validate"], ["examples"], ["compare"], ["gui"]])
def test_help_works(cmd: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*cmd, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_run_writes_all_files_and_prints_us_summary(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = {p.name for p in run_dir.iterdir()}
    assert names == {
        "summary.json",
        "timeseries.csv",
        "samples.npz",
        "snapshots.npz",
        "scenario.toml",
    }
    assert load_run(run_dir).summary.stop_reason.value == "target_reached"


def test_run_prints_us_units_by_default(
    pot_toml: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", str(pot_toml), "--out", str(tmp_path / "o"), "--cell-size", "0.02"])
    cap = capsys.readouterr()
    assert code == 0
    assert "°F" in cap.out and "BTU" in cap.out
    assert "target_reached" in cap.out
    assert "100.0%" in cap.err or "%" in cap.err  # progress bar goes to stderr


def test_run_metric_units(
    pot_toml: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "run",
            str(pot_toml),
            "--out",
            str(tmp_path / "o"),
            "--cell-size",
            "0.02",
            "--units",
            "metric",
            "--quiet",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "°C" in out and "MJ" in out


def test_run_exit_1_when_target_not_reached(pot_toml: Path, tmp_path: Path) -> None:
    code = main(
        [
            "run",
            str(pot_toml),
            "--out",
            str(tmp_path / "o"),
            "--cell-size",
            "0.02",
            "--max-time",
            "30",
            "--quiet",
        ]
    )
    assert code == 1


def test_validate_good_and_broken(
    pot_toml: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate", str(pot_toml)]) == 0
    out = capsys.readouterr().out
    assert "valid" in out and "Liquid cells" in out and "memory" in out
    broken = tmp_path / "broken.toml"
    broken.write_text("schema_version = 1\nname = 'x'\n[vessel]\ninner_diameter_m = -1\n")
    assert main(["validate", str(broken)]) == 2
    err = capsys.readouterr().err
    assert "error" in err.lower() and len(err.strip()) > 10
    assert main(["validate", str(tmp_path / "missing.toml")]) == 2


def test_validate_geometry_error_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = scenario_to_toml(example_scenarios()["small_pot_stove"])
    assert "position_m" in text
    broken = re.sub(r"position_m = \[[^\]]*\]", "position_m = [5.0, 0.0, 0.1]", text, count=1)
    path = tmp_path / "bad_probe.toml"
    path.write_text(broken)
    assert main(["validate", str(path)]) == 2
    assert capsys.readouterr().err.strip()


def test_examples_list_and_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["examples"]) == 0
    out = capsys.readouterr().out
    for name in example_scenarios():
        assert name in out
    assert main(["examples", "--write", str(tmp_path / "ex")]) == 0
    written = sorted(p.name for p in (tmp_path / "ex").glob("*.toml"))
    assert written == sorted(f"{n}.toml" for n in example_scenarios())


def test_compare(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = load_run(run_dir)
    t = np.array([s.t for s in run.samples])
    T = np.array([s.T_mean for s in run.samples])
    csv = tmp_path / "meas.csv"
    lines = ["time_s,temp_C"] + [
        f"{ti:.1f},{Ti - 273.15 + 1.0:.3f}" for ti, Ti in zip(t[::10], T[::10], strict=True)
    ]
    csv.write_text("\n".join(lines) + "\n")
    code = main(["compare", str(run_dir), str(csv), "--probe", "mean"])
    out = capsys.readouterr().out
    assert code == 0
    assert "RMSE" in out and "Bias" in out
    assert "+1.00 K" in out or "-1.00 K" in out


def test_gui_launches_lazily(monkeypatch: pytest.MonkeyPatch, pot_toml: Path) -> None:
    calls: list[list[str]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        calls.append(list(argv or []))
        return 7

    monkeypatch.setattr("thermosim.gui.app.main", fake_main)
    assert main(["gui", str(pot_toml)]) == 7
    assert main(["gui"]) == 7
    assert calls == [[str(pot_toml)], []]


def test_solver_error_exit_3(
    pot_toml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thermosim.core.errors import SolverError
    from thermosim.solver.thermal import ThermalSolver

    def boom(*args: object, **kwargs: object) -> None:
        raise SolverError("synthetic")

    monkeypatch.setattr(ThermalSolver, "step", boom)
    code = main(
        ["run", str(pot_toml), "--out", str(tmp_path / "o"), "--cell-size", "0.02", "--quiet"]
    )
    assert code == 3
