"""任务环境中的研究入口，接口由对应版本的 scheme 负责。"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

from runtime.utils import logger as log

from scheme._errors import error_text

from .parameters import inspect_project

TASK_KINDS = ("factor", "model", "optimize", "control", "execution", "strategy")


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
            if arguments.project is None:
                raise ValueError("parameters 需要 --project")
            values = json.load(sys.stdin) if arguments.validate else None
            result = inspect_project(arguments.project.resolve(), values)
            print(json.dumps(result, ensure_ascii=False))
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
        components = (
            [data["factor"]]
            if kind == "factor"
            else [item for item in data["algos"].values() if item]
        )
        descriptions = "; ".join(
            f"{item['package']} {item['version']} ({item['entry']})" for item in components
        )
        log.info(f"任务 {input_file.parent.name}：类型={kind}，组件={descriptions}，输出={output}")
        from scheme.execute.checkpoints import Checkpoints

        environment = data.get("environment")
        lockfile = (
            input_file.parent / environment["lockfile"]
            if isinstance(environment, dict) and isinstance(environment.get("lockfile"), str)
            else None
        )
        checkpoints = Checkpoints.for_input(input_file, lockfile)
        code = import_module(f"scheme.execute.{kind}.task").run(data, checkpoints=checkpoints)
        log.success(f"正式任务完成：{kind}，输出 {output}")
        return code
    except Exception as error:
        if arguments.command == "run":
            # 正式任务的错误连同 traceback 进入调度日志；parameters 的 stderr 仍是单行错误协议。
            log.error(f"正式任务失败：{error_text(error, traceback=True)}")
        else:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
