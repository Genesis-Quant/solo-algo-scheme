import json
from importlib import import_module
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from scheme.execute.results import save_run
from scheme.execute.schema import Environment, Task
from scheme.manage import main
from scheme.report.base import Report

KINDS = ("factor", "model", "optimize", "control", "execution", "strategy")


@pytest.mark.parametrize("kind", KINDS)
def test_cli_preserves_task_kind_and_dispatches_own_entry(kind, tmp_path, monkeypatch):
    data = {
        "kind": kind,
        "environment": {"lockfile": "environment/uv.lock"},
        "output": "report",
        **({"factor": {"wheel": "factor.whl"}} if kind == "factor" else {"algos": {}}),
    }
    source = tmp_path / "input.json"
    source.write_text(json.dumps(data))
    handler = Mock(return_value=0)
    monkeypatch.setattr(import_module(f"scheme.execute.{kind}.task"), "run", handler)
    assert main(["run", "--input", str(source), "--output", str(tmp_path / "report")]) == 0
    assert handler.call_args.args[0]["kind"] == kind


@pytest.mark.parametrize("kind", KINDS[1:])
def test_algo_task_uses_own_schema(kind, tmp_path, monkeypatch):
    entry = import_module(f"scheme.execute.{kind}.task")
    handler = Mock(return_value=0)
    monkeypatch.setattr(entry, "run_research", handler)
    component = dict(package="example", version="1.0.0", wheel="example.whl",
                     sha256="0" * 64, entry="example:Algo")
    data = dict(kind=kind, environment={"lockfile": tmp_path / "uv.lock"},
                output=tmp_path / "report", algos={name: component for name in KINDS[1:5]},
                backtest={"start": "2026-06-01", "end": "2026-06-03"})
    assert entry.run(data, input_sha256="a" * 64) == 0
    assert handler.call_args.args[0].kind == kind
    with pytest.raises(ValidationError):
        entry.run({**data, "kind": "backtest"}, input_sha256="a" * 64)
    if kind != "strategy":
        with pytest.raises(ValidationError):
            entry.run({**data, "algos": {**data["algos"], kind: None}}, input_sha256="a" * 64)


@pytest.mark.parametrize("kind", KINDS)
def test_manifest_separates_task_and_report_kinds(kind, tmp_path):
    class Result(Report):
        report_kind = "factor" if kind == "factor" else "backtest"
        filenames = {"data": "data.parquet"}
        data = pd.DataFrame({"value": [1]})

    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1")
    task = Task(kind=kind, environment=Environment(lockfile=lockfile), output=tmp_path / "report")
    save_run(task, Result(), {"scheme": "1.0.0"}, input_sha256="a" * 64)
    manifest = json.loads((task.output / "run.json").read_text())
    assert manifest["input"]["kind"] == kind
    assert manifest["report_kind"] == Result.report_kind


def test_backtest_is_not_a_task_kind(tmp_path):
    with pytest.raises(ValidationError):
        Task(kind="backtest", environment={"lockfile": tmp_path / "uv.lock"}, output=tmp_path)
