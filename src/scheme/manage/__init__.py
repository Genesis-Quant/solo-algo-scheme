"""任务环境中的研究入口，接口由对应版本的 scheme 负责。"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from time import perf_counter

from scheme.log import duration, get_logger

TASK_KINDS = ("factor", "model", "optimize", "control", "execution", "strategy")

log = get_logger("scheme.task")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scheme")
    parser.add_argument("command", choices=["run", "parameters"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "parameters":
            from .parameters import inspect_project

            if arguments.project is None:
                raise ValueError("parameters 需要 --project")
            values = json.load(sys.stdin) if arguments.validate else None
            print(json.dumps(inspect_project(arguments.project.resolve(), values), ensure_ascii=False))
            return 0
        if arguments.input is None or arguments.output is None:
            raise ValueError("run 需要 --input 和 --output")
        input_file = arguments.input.resolve()
        data = json.loads(input_file.read_text(encoding="utf-8"))
        kind = data["kind"]
        if kind not in TASK_KINDS:
            raise ValueError(f"不支持的研究类型：{kind}")
        output = (input_file.parent / data["output"]).resolve()
        if output != arguments.output.resolve():
            raise ValueError("输出目录与 input.output 不一致")
        data["output"] = str(output)
        started = perf_counter()
        log.info("正式任务开始：{}，输入 {}", kind, input_file)
        code = import_module(f"scheme.execute.{kind}.task").run(data)
        log.info("正式任务完成：{}，用时 {}", kind, duration(perf_counter() - started))
        return code
    except Exception as error:
        if arguments.command == "run":
            # 正式任务的错误连同 traceback 进入调度日志；parameters 的 stderr 仍是单行错误协议。
            log.opt(exception=error).error("正式任务失败：{}: {}", type(error).__name__, error)
        else:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
