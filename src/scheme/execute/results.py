import json

from runtime.utils import logger as log

from scheme.report.base import Report

from .schema import Task


def check_output(task: Task) -> None:
    if (task.output / "run.json").exists():
        raise FileExistsError("输出目录已有成功运行记录，请使用新的 Run/Attempt 目录")


def save_run(task: Task, result: Report, versions: dict[str, str]) -> None:
    paths = result.save(task.output)
    # 成功清单最后发布，避免把部分写入的 Parquet 视为完整报告。
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
    log.success(
        f"报告已保存：{len(paths)} 个文件，"
        f"共 {sum(path.stat().st_size for path in paths)} 字节，目录 {task.output}"
    )
