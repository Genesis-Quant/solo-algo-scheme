import hashlib
import json

from scheme.report.base import Report

from .schema import Task


def check_output(task: Task) -> None:
    if (task.output / "run.json").exists():
        raise FileExistsError("输出目录已有成功运行记录，请使用新的 Run/Attempt 目录")


def save_run(task: Task, result: Report, versions: dict[str, str], *, input_sha256: str) -> None:
    check_output(task)
    paths = result.save(task.output)
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    # 最后写成功清单，部分失败的 Parquet 不会被视为完整报告。
    temporary = task.output / "run.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "protocol": 1,
                "status": "success",
                "input_sha256": input_sha256,
                "input": task.model_dump(mode="json"),
                "report_kind": result.report_kind,
                "versions": versions,
                "reports": result.filenames,
                "report_sha256": hashes,
                "lock_sha256": hashlib.sha256(task.environment.lockfile.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(task.output / "run.json")
