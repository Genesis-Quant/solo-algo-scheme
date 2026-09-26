"""任务环境中的研究入口，接口由对应版本的 scheme 负责。"""

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path

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
            from .parameters import inspect_project

            if arguments.project is None:
                raise ValueError("parameters 需要 --project")
            values = json.load(sys.stdin) if arguments.validate else None
            print(json.dumps(inspect_project(arguments.project.resolve(), values), ensure_ascii=False))
            return 0
        if arguments.input is None or arguments.output is None:
            raise ValueError("run 需要 --input 和 --output")
        input_file = arguments.input.resolve()
        source = input_file.read_bytes()
        data = json.loads(source)
        kind = data["kind"]
        if kind not in TASK_KINDS:
            raise ValueError(f"不支持的研究类型：{kind}")
        output = (input_file.parent / data["output"]).resolve()
        if output != arguments.output.resolve():
            raise ValueError("输出目录与 input.output 不一致")
        data["output"] = str(output)
        data["environment"]["lockfile"] = str(
            (input_file.parent / data["environment"]["lockfile"]).resolve()
        )
        components = [data["factor"]] if kind == "factor" else data["algos"].values()
        for component in components:
            if component is not None and not component["wheel"].startswith("https://"):
                component["wheel"] = str((input_file.parent / component["wheel"]).resolve())
        return import_module(f"scheme.execute.{kind}.task").run(
            data, input_sha256=hashlib.sha256(source).hexdigest()
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
