"""任务环境中的研究入口，接口由对应版本的 scheme 负责。"""

import argparse
import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scheme")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        input_file = arguments.input.resolve()
        source = input_file.read_bytes()
        data = json.loads(source)
        kind = data["kind"]
        if kind not in {"factor", "backtest"}:
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
        return import_module(f"scheme.manage.apps.{kind}").run(
            data, input_sha256=hashlib.sha256(source).hexdigest()
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
