import json

from scheme.log import get_logger
from scheme.report.base import Report

from .schema import Task

log = get_logger("scheme.task")


def check_output(task: Task) -> None:
    if (task.output / "run.json").exists():
        raise FileExistsError("输出目录已有成功运行记录，请使用新的 Run/Attempt 目录")


def save_run(task: Task, result: Report, versions: dict[str, str]) -> None:
    paths = result.save(task.output)
    # 最后写成功清单，部分失败的 Parquet 不会被视为完整报告。
    temporary = task.output / "run.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "protocol": 1,
                "status": "success",
                "input": task.model_dump(mode="json"),
                "report_kind": result.report_kind,
                "versions": versions,
                "reports": result.filenames,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(task.output / "run.json")
    log.info(
        "报告已保存：{} 个文件（{}），共 {:.1f} MB → {}",
        len(paths), ", ".join(path.name for path in paths),
        sum(path.stat().st_size for path in paths) / 2**20, task.output,
    )
