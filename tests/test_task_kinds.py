import json
from importlib import import_module
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from scheme.execute.results import check_output, save_run
from scheme.execute.schema import Environment, ModelTask, Package, Task
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
    assert handler.call_args.args == ({**data, "output": str(tmp_path / "report")},)
    assert handler.call_args.kwargs == {}


@pytest.mark.parametrize("kind", KINDS[1:])
def test_algo_task_uses_own_schema(kind, tmp_path, monkeypatch):
    entry = import_module(f"scheme.execute.{kind}.task")
    handler = Mock(return_value=0)
    monkeypatch.setattr(entry, "run_research", handler)
    component = dict(package="example", version="1.0.0", wheel="example.whl",
                     entry="example:Algo")
    data = dict(kind=kind, environment={"lockfile": tmp_path / "uv.lock"},
                output=tmp_path / "report", algos={name: component for name in KINDS[1:5]},
                backtest={"start": "2026-06-01", "end": "2026-06-03"})
    assert entry.run(data) == 0
    assert handler.call_args.args[0].kind == kind
    assert handler.call_args.kwargs == {}
    with pytest.raises(ValidationError):
        entry.run({**data, "kind": "backtest"})
    if kind != "strategy":
        with pytest.raises(ValidationError):
            entry.run({**data, "algos": {**data["algos"], kind: None}})


@pytest.mark.parametrize("kind", KINDS)
def test_manifest_separates_task_and_report_kinds(kind, tmp_path):
    class Result(Report):
        report_kind = "factor" if kind == "factor" else "backtest"
        filenames = {"data": "data.parquet"}
        data = pd.DataFrame({"value": [1]})

    task = Task(kind=kind, environment=Environment(lockfile=tmp_path / "uv.lock"),
                output=tmp_path / "report")
    save_run(task, Result(), {"scheme": "1.0.0"})
    manifest = json.loads((task.output / "run.json").read_text())
    assert set(manifest) == {"protocol", "status", "input", "report_kind", "versions", "reports"}
    assert manifest["protocol"] == 1
    assert manifest["status"] == "success"
    assert manifest["input"]["kind"] == kind
    assert manifest["report_kind"] == Result.report_kind
    assert manifest["versions"] == {"scheme": "1.0.0"}
    assert manifest["reports"] == Result.filenames
    assert all((task.output / filename).is_file() for filename in manifest["reports"].values())
    assert not (task.output / "run.json.tmp").exists()


def test_algo_research_runs_and_saves_without_reading_task_files(tmp_path, monkeypatch):
    from scheme.base import Algo, ModelAlgo, ModelParams, ResearchContext
    from scheme.execute import research

    class Context(ResearchContext[float]):
        pass

    class Model(ModelAlgo[ModelParams, Context]):
        pass

    task = ModelTask.model_validate({
        "environment": {"lockfile": tmp_path / "uv.lock"}, "output": tmp_path / "report",
        "algos": {"model": {
            "package": "model-test", "version": "1.0.0", "wheel": "model.whl", "entry": "model:Model",
        }},
        "backtest": {"start": "2026-06-01", "end": "2026-06-03"},
    })
    versions = {"scheme": "1.0.0", "model-test": "1.0.0"}
    result = Mock(spec=Report)
    loaded = Mock(return_value=Model)
    backtest = Mock(return_value=result)
    saved = Mock()
    monkeypatch.setattr(research, "installed_versions", lambda: versions)
    monkeypatch.setattr(research, "load_component", loaded)
    monkeypatch.setattr(research, "run_backtest", backtest)
    monkeypatch.setattr(research, "save_run", saved)

    assert research.run(task) == 0
    loaded.assert_called_once_with(task.algos.model, Algo)
    algos, context, parameters = backtest.call_args.args
    assert type(algos[0]) is Model
    assert type(context) is Context
    assert parameters is task.backtest
    saved.assert_called_once_with(task, result, versions)
    assert not task.environment.lockfile.exists()


def test_component_contract_has_only_package_identity_and_entry():
    assert set(Package.model_fields) == {"package", "version", "wheel", "entry"}


def test_incomplete_report_does_not_publish_run_manifest(tmp_path):
    task = Task(kind="factor", environment={"lockfile": tmp_path / "uv.lock"}, output=tmp_path)
    result = Mock(spec=Report)

    def fail(output):
        (output / "partial.parquet").touch()
        raise OSError("report save failed")

    result.save.side_effect = fail
    with pytest.raises(OSError, match="report save failed"):
        save_run(task, result, {"scheme": "1.0.0"})
    assert not (task.output / "run.json").exists()
    assert not (task.output / "run.json.tmp").exists()


def test_completed_output_is_rejected_before_task_execution(tmp_path):
    task = Task(kind="factor", environment={"lockfile": tmp_path / "uv.lock"}, output=tmp_path)
    (task.output / "run.json").touch()
    with pytest.raises(FileExistsError, match="新的 Run/Attempt"):
        check_output(task)


def test_backtest_is_not_a_task_kind(tmp_path):
    with pytest.raises(ValidationError):
        Task(kind="backtest", environment={"lockfile": tmp_path / "uv.lock"}, output=tmp_path)
