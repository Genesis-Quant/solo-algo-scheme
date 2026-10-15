"""包身份、锁定版本和类入口校验；不在已运行的 Kernel 中安装依赖。"""

import importlib
import importlib.metadata as metadata
import re
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scheme.execute.strategy.assembly import parameter_type
from scheme.log import get_logger

from .schema import Component, Package

log = get_logger("scheme.task")


def installed_versions() -> dict[str, str]:
    versions = {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}
    log.debug("任务环境：{}", "，".join(f"{name} {version}" for name, version in sorted(versions.items())))
    return versions


def validate_scheme_requirement(
    requirements: list[str], version: Version, package_version: Version | None = None,
) -> None:
    """仅接受同 major.minor 的完整 patch 范围；版本限 1–3 段正式数字版本。"""
    if any(
        not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", str(value))
        for value in (version, package_version) if value is not None
    ):
        raise ValueError("研究包和 scheme 必须使用 1–3 段正式数字版本")
    if package_version is not None and (
        package_version.major, package_version.minor
    ) != (version.major, version.minor):
        raise ValueError("研究包主版本和次版本必须与 scheme 一致")
    parsed = [(text, Requirement(text)) for text in requirements]
    candidates = [(text, r) for text, r in parsed if canonicalize_name(r.name) == "scheme"]
    message = (
        "研究包必须声明兼容 scheme 的唯一范围："
        f">={version.major}.{version.minor}.0,<{version.major}.{version.minor + 1}.0"
        "（无 URL、marker、extras）"
    )
    if len(candidates) != 1:
        raise ValueError(message)
    text, requirement = candidates[0]
    bounds = {specifier.operator: specifier.version for specifier in requirement.specifier}
    # SpecifierSet 会去重；原始声明也只能包含两个边界。
    if (requirement.url or requirement.marker or requirement.extras or text.count(",") != 1
            or len(requirement.specifier) != 2 or set(bounds) != {">=", "<"}
            or not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", value) for value in bounds.values())
            or Version(bounds[">="]) != Version(f"{version.major}.{version.minor}.0")
            or Version(bounds["<"]) != Version(f"{version.major}.{version.minor + 1}.0")):
        raise ValueError(message)


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


def instantiate(component: Component, base: type) -> Any:
    cls = load_component(component, base)
    params = parameter_type(cls).model_validate(component.params)
    return cls(params)
