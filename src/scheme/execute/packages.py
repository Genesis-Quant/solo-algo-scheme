"""包身份、锁定版本和类入口校验；不在已运行的 Kernel 中安装依赖。"""

import hashlib
import importlib
import importlib.metadata as metadata
import json
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scheme.execute.strategy.assembly import parameter_type

from .schema import Component, Environment, Package


def validate_environment(environment: Environment) -> dict[str, str]:
    lock = tomllib.loads(environment.lockfile.read_text(encoding="utf-8"))
    packages = lock.get("package", [])
    if any({"editable", "directory"} & p.get("source", {}).keys() for p in packages):
        raise ValueError("正式任务锁文件不能包含可变源码目录")
    candidates: dict[str, set[str]] = {}
    for package in packages:
        candidates.setdefault(canonicalize_name(package["name"]), set()).add(package["version"])
    actual = {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}
    for name, version in actual.items():
        if version not in candidates.get(name, set()):
            raise ValueError(f"环境中的 {name}=={version} 不在任务锁文件中")
    # 检查实际生效的传递依赖，不要求安装锁文件中其他平台或未启用 extra 的包。
    for distribution in metadata.distributions():
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            name = canonicalize_name(requirement.name)
            if name not in actual or (
                requirement.specifier and actual[name] not in requirement.specifier
            ):
                raise ValueError(f"{distribution.metadata['Name']} 的依赖不满足：{text}")
            if requirement.url and requirement.url.startswith("git+"):
                direct = json.loads(
                    metadata.distribution(name).read_text("direct_url.json") or "{}"
                )
                revision = requirement.url.rsplit("@", 1)[-1]
                vcs = direct.get("vcs_info", {})
                commits = {
                    p.get("source", {}).get("git", "").rsplit("#", 1)[-1]
                    for p in packages
                    if canonicalize_name(p["name"]) == name
                }
                if vcs.get("commit_id") not in commits or revision not in {
                    vcs.get("commit_id"),
                    vcs.get("requested_revision"),
                }:
                    raise ValueError(f"{name} Git commit 与依赖声明不一致")
    return actual


def validate_scheme_requirement(requirements: list[str], version: Version) -> None:
    """仅接受同主版本的最低正式版本范围，并验证实际 Scheme 在范围内。"""
    parsed = [(text, Requirement(text)) for text in requirements]
    candidates = [(text, r) for text, r in parsed if canonicalize_name(r.name) == "scheme"]
    message = (
        "研究包必须声明兼容 scheme 的唯一范围："
        f">=X.Y.Z,<{version.major + 1}.0.0（X={version.major}，无 URL、marker、extras）"
    )
    if len(candidates) != 1:
        raise ValueError(message)
    text, requirement = candidates[0]
    bounds = {specifier.operator: specifier.version for specifier in requirement.specifier}
    minimum = bounds.get(">=", "")
    # SpecifierSet 会去重；原始声明也只能包含两个边界。
    if (requirement.url or requirement.marker or requirement.extras or text.count(",") != 1
            or len(requirement.specifier) != 2 or set(bounds) != {">=", "<"}
            or not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", value) for value in bounds.values())
            or Version(minimum).major != version.major
            or Version(bounds["<"]) != Version(f"{version.major + 1}.0.0")):
        raise ValueError(message)
    if version not in requirement.specifier:
        raise ValueError(f"实际 scheme {version} 不满足研究包依赖：{requirement}")


def verify_component(component: Package) -> metadata.Distribution:
    distribution = metadata.distribution(component.package)
    if distribution.version != component.version:
        raise ValueError(f"{component.package} 实际安装版本不是 {component.version}")
    scheme = Version(metadata.version("scheme"))
    if Version(component.version).major != scheme.major:
        raise ValueError("研究包主版本必须与 scheme 一致")
    validate_scheme_requirement(distribution.requires or [], scheme)
    with tempfile.TemporaryDirectory(prefix="solo-wheel-") as temporary:
        path = Path(temporary) / "component.whl"
        if component.wheel.startswith("https://"):
            with urlopen(component.wheel, timeout=60) as response, path.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        else:
            path = Path(component.wheel)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != component.sha256.lower():
                raise ValueError("wheel SHA256 不一致")
        with ZipFile(path) as wheel:
            # 校验实际安装内容，防止同版本的不同 wheel 或被修改的 editable 源码冒充。
            records = [n for n in wheel.namelist() if n.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise ValueError("wheel RECORD 无效")
            import csv

            for name, checksum, _ in csv.reader(wheel.read(records[0]).decode().splitlines()):
                if not checksum or ".data/" in name:
                    continue
                installed = Path(distribution.locate_file(name))
                if not installed.is_file() or installed.read_bytes() != wheel.read(name):
                    raise ValueError(f"安装内容与候选 wheel 不一致：{name}")
    return distribution


def load_entry(distribution: metadata.Distribution, entry: str) -> type:
    module_name, class_name = entry.split(":")
    module_path = module_name.replace(".", "/")
    files = {str(f).replace("\\", "/") for f in distribution.files or []}
    if not {module_path + ".py", module_path + "/__init__.py"} & files:
        raise ValueError(f"入口 {entry} 不属于 {distribution.metadata['Name']}")
    value = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(value, type):
        raise TypeError(f"入口不是类：{entry}")
    return value


def load_component(component: Package, base: type) -> type:
    distribution = verify_component(component)
    cls = load_entry(distribution, component.entry)
    if not issubclass(cls, base):
        raise TypeError(f"{component.entry} 未继承当前 scheme 的 {base.__name__}")
    return cls


def instantiate(component: Component, base: type) -> Any:
    cls = load_component(component, base)
    params = parameter_type(cls).model_validate(component.params)
    return cls(params)
