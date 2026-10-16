"""包身份、锁定版本和类入口校验；不在已运行的 Kernel 中安装依赖。"""

import importlib
import importlib.metadata as metadata
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scheme.log import get_logger

from .schema import Package

log = get_logger("scheme.task")


def installed_versions() -> dict[str, str]:
    versions = {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}
    log.debug("任务环境：{}", "，".join(f"{name} {version}" for name, version in sorted(versions.items())))
    return versions


def validate_scheme_requirement(
    requirements: list[str], version: Version, package_version: Version | None = None,
) -> None:
    """主、次版本一致，依赖范围包含系列起点、排除下一系列且满足当前 scheme。"""
    if any(
        not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", str(value))
        for value in (version, package_version) if value is not None
    ):
        raise ValueError("研究包和 scheme 必须使用 1–3 段正式数字版本")
    if package_version is not None and (
        package_version.major, package_version.minor
    ) != (version.major, version.minor):
        raise ValueError("研究包主版本和次版本必须与 scheme 一致")
    parsed = [Requirement(text) for text in requirements]
    candidates = [r for r in parsed if canonicalize_name(r.name) == "scheme"]
    minimum = Version(f"{version.major}.{version.minor}.0")
    maximum = Version(f"{version.major}.{version.minor + 1}.0")
    message = f"研究包必须声明兼容 scheme 的唯一范围：包含 {minimum} 且不包含 {maximum}"
    if len(candidates) != 1:
        raise ValueError(message)
    requirement = candidates[0]
    if minimum not in requirement.specifier or maximum in requirement.specifier:
        raise ValueError(message)
    if version not in requirement.specifier:
        raise ValueError(f"当前 scheme {version} 不满足研究包依赖：{requirement}")


def verify_component(component: Package) -> metadata.Distribution:
    distribution = metadata.distribution(component.package)
    if distribution.version != component.version:
        raise ValueError(f"{component.package} 实际安装版本不是 {component.version}")
    scheme = Version(metadata.version("scheme"))
    validate_scheme_requirement(distribution.requires or [], scheme, Version(component.version))
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
