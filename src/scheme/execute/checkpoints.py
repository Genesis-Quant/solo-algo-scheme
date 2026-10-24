"""正式任务的阶段检查点：失败或终止后重试时复用已完成阶段，报告发布成功后全部删除。"""

import hashlib
import json
import os
import pickle
import shutil
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.utils import logger as log

from scheme._errors import error_text

__all__ = ["Checkpoints"]

PROTOCOL = 1
MANIFEST = "checkpoint.json"
VALUE = "value.pkl"
LABELS = {"factor": "因子计算", "backtest": "回测回放", "report": "报告数据"}


def readable(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class Checkpoints:
    """同一 Run 目录内按阶段保存的结果，只在冻结输入和锁定环境完全一致时复用。"""

    def __init__(self, directory: Path, fingerprint: str) -> None:
        self.directory = directory
        self.fingerprint = fingerprint

    @classmethod
    def for_input(cls, input_file: Path, lockfile: Path | None) -> "Checkpoints":
        digest = hashlib.sha256(input_file.read_bytes())
        if lockfile is not None and lockfile.is_file():
            digest.update(b"\0")
            digest.update(lockfile.read_bytes())
        return cls(input_file.parent / "checkpoints", digest.hexdigest())

    def complete(self, stage: str, files: Iterable[str] = ()) -> Path | None:
        """返回完整且属于当前输入的阶段目录；不完整或过期的检查点会被删除。"""
        directory = self.directory / stage
        if not directory.is_dir():
            return None
        try:
            manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
            if (
                manifest.get("protocol") != PROTOCOL
                or manifest.get("stage") != stage
                or manifest.get("fingerprint") != self.fingerprint
            ):
                raise ValueError("与当前冻结输入或锁定环境不一致")
            recorded = manifest["files"]
            if missing := [name for name in files if name not in recorded]:
                raise ValueError(f"缺少文件 {missing}")
            for name, size in recorded.items():
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"文件名无效：{name!r}")
                if (directory / name).stat().st_size != size:
                    raise ValueError(f"文件不完整：{name}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            log.warning(f"[检查点] 丢弃{LABELS.get(stage, stage)}：{error_text(error)}")
            shutil.rmtree(directory, ignore_errors=True)
            return None
        return directory

    def load(self, stage: str) -> Any | None:
        directory = self.complete(stage, [VALUE])
        if directory is None:
            return None
        try:
            with (directory / VALUE).open("rb") as file:
                value = pickle.load(file)
        except Exception as error:
            log.warning(f"[检查点] {LABELS[stage]}无法读取，重新计算：{error_text(error)}")
            shutil.rmtree(directory, ignore_errors=True)
            return None
        log.info(
            f"[检查点] 复用{LABELS[stage]}结果，跳过该阶段"
            f"（{readable((directory / VALUE).stat().st_size)}）"
        )
        return value

    def save(self, stage: str, value: Any) -> None:
        """中间结果保存失败不影响本次运行，只是重试时不能跳过该阶段。"""

        def write(directory: Path) -> None:
            with (directory / VALUE).open("wb") as file:
                pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            self.commit(stage, write)
        except Exception as error:
            log.warning(f"[检查点] 保存{LABELS[stage]}失败，本次运行继续：{error_text(error)}")

    def commit(self, stage: str, write: Callable[[Path], object]) -> Path:
        """先写入临时目录，清单最后写入，再整体替换为阶段目录。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob(f".{stage}-*.tmp"):
            shutil.rmtree(stale, ignore_errors=True)
        staging = self.directory / f".{stage}-{uuid4().hex}.tmp"
        staging.mkdir()
        target = self.directory / stage
        try:
            write(staging)
            files = {path.name: path.stat().st_size for path in staging.iterdir() if path.is_file()}
            (staging / MANIFEST).write_text(
                json.dumps(
                    {
                        "protocol": PROTOCOL,
                        "stage": stage,
                        "fingerprint": self.fingerprint,
                        "created_at": datetime.now(UTC).isoformat(),
                        "files": files,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            shutil.rmtree(target, ignore_errors=True)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        log.info(
            f"[检查点] 已保存{LABELS[stage]}（{readable(sum(files.values()))}），"
            "失败或终止后重试将直接复用"
        )
        return target

    def publish(self, directory: Path, filenames: Iterable[str], output: Path) -> list[Path]:
        """复制而不移动，发布中断时检查点仍然完整。"""
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for filename in filenames:
            destination = output / filename
            temporary = destination.with_name(f"{destination.name}.tmp")
            shutil.copyfile(directory / filename, temporary)
            temporary.replace(destination)
            paths.append(destination)
        return paths

    def clear(self) -> None:
        if not self.directory.exists():
            return
        try:
            shutil.rmtree(self.directory)
        except OSError as error:
            log.warning(f"[检查点] 运行已成功，但删除检查点失败：{error_text(error)}")
            return
        log.info("[检查点] 运行成功，已删除全部检查点")
