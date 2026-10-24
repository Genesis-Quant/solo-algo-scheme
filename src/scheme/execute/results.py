import json
from collections.abc import Mapping
from pathlib import Path

from runtime.utils import logger as log

from scheme.report.base import Report

from .checkpoints import Checkpoints
from .schema import Task


def check_output(task: Task) -> None:
    if (task.output / "run.json").exists():
        raise FileExistsError("输出目录已有成功运行记录，请使用新的 Run/Attempt 目录")


def save_run(
    task: Task,
    result: Report,
    versions: dict[str, str],
    checkpoints: Checkpoints | None = None,
) -> None:
    if checkpoints is None:
        paths = result.save(task.output)
    else:
        directory = checkpoints.commit("report", result.save)
        paths = checkpoints.publish(directory, result.filenames.values(), task.output)
    publish_manifest(task, result.report_kind, result.filenames, versions, paths)
    if checkpoints is not None:
        checkpoints.clear()


def resume_run(
    task: Task,
    report: type[Report],
    versions: dict[str, str],
    checkpoints: Checkpoints | None,
) -> bool:
    """报告数据已有完整检查点时直接发布，不再计算任何阶段。"""
    if checkpoints is None:
        return False
    directory = checkpoints.complete("report", report.filenames.values())
    if directory is None:
        return False
    log.info("[检查点] 报告数据已完整保存，跳过全部计算阶段，直接发布报告")
    paths = checkpoints.publish(directory, report.filenames.values(), task.output)
    publish_manifest(task, report.report_kind, report.filenames, versions, paths)
    checkpoints.clear()
    return True


def publish_manifest(
    task: Task,
    report_kind: str,
    filenames: Mapping[str, str],
    versions: dict[str, str],
    paths: list[Path],
) -> None:
    # 成功清单最后发布，避免把部分写入的 Parquet 视为完整报告。
    temporary = task.output / "run.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "protocol": 1,
                "status": "success",
                "input": task.model_dump(mode="json"),
                "report_kind": report_kind,
                "versions": versions,
                "reports": dict(filenames),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(task.output / "run.json")
    log.success(
        f"报告已保存：{len(paths)} 个文件，"
        f"共 {sum(path.stat().st_size for path in paths)} 字节，目录 {task.output}"
    )
