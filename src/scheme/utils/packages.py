"""包身份、锁定版本和类入口校验；不在已运行的 Kernel 中安装依赖。"""

import hashlib
import importlib
import importlib.metadata as metadata
import json
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version
from pydantic import BaseModel

from scheme import generic_model, parameter_type

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


def verify_component(component: Package) -> metadata.Distribution:
    distribution = metadata.distribution(component.package)
    if distribution.version != component.version:
        raise ValueError(f"{component.package} 实际安装版本不是 {component.version}")
    if Version(component.version).major != Version(metadata.version("scheme")).major:
        raise ValueError("研究包主版本必须与 scheme 一致")
    requirements = [Requirement(r) for r in distribution.requires or []]
    major = Version(metadata.version("scheme")).major
    scheme_requirements = [r for r in requirements if canonicalize_name(r.name) == "scheme"]
    required = SpecifierSet(f">={major}.0.0,<{major + 1}.0.0")
    if (len(scheme_requirements) != 1 or scheme_requirements[0].url
            or scheme_requirements[0].marker or scheme_requirements[0].specifier != required):
        raise ValueError(f"研究包必须声明兼容整个大版本的 scheme 范围：{required}")
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


def validate_context(algo: Any, ctx: BaseModel) -> None:
    expected = generic_model(type(algo), 1)
    generic = getattr(expected, "__pydantic_generic_metadata__", {})
    if generic.get("args") == (Any,):
        expected = generic["origin"]
    if not isinstance(ctx, expected):
        raise TypeError(
            f"{type(algo).__name__} 需要 {expected.__name__}，收到 {type(ctx).__name__}"
        )


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
