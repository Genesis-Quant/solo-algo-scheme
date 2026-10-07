import hashlib
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pydantic import BaseModel

from scheme import Algo, ResearchContext
from scheme.execute.packages import validate_environment, verify_component
from scheme.execute.schema import Component, Environment
from scheme.execute.strategy.assembly import validate_context


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    installed.mkdir()
    source = installed / "algo.py"
    source.write_bytes(b"class Algo: pass\n")
    wheel = tmp_path / "candidate.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("algo.py", source.read_bytes())
        archive.writestr("demo-1.0.0.dist-info/RECORD", "algo.py,sha256=unused,17\n")
    dist = SimpleNamespace(
        version="1.0.0", requires=["scheme>=1,<1.1"], locate_file=lambda p: installed / p
    )
    monkeypatch.setattr("scheme.execute.packages.metadata.distribution", lambda _: dist)
    monkeypatch.setattr("scheme.execute.packages.metadata.version", lambda _: "1.0.0")
    component = Component(
        package="demo",
        version="1.0.0",
        wheel=str(wheel),
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        entry="algo:Algo",
    )
    return component, source, dist


def test_wheel_contents_match_installed(candidate):
    component, _, dist = candidate
    assert verify_component(component) is dist


def test_wrong_wheel_hash(candidate):
    component, _, _ = candidate
    component.sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        verify_component(component)


def test_same_version_modified_source(candidate):
    component, source, _ = candidate
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="安装内容"):
        verify_component(component)


def test_wrong_installed_version(candidate):
    component, _, dist = candidate
    dist.version = "1.0.7"
    with pytest.raises(ValueError, match="实际安装版本"):
        verify_component(component)


def test_wrong_scheme_major(candidate):
    component, _, dist = candidate
    component.version = dist.version = "2.0.0"
    with pytest.raises(ValueError, match="主版本"):
        verify_component(component)


def test_missing_scheme_requirement(candidate):
    component, _, dist = candidate
    dist.requires = []
    with pytest.raises(ValueError, match="声明兼容"):
        verify_component(component)


@pytest.mark.parametrize("requirements,scheme_version,package_version,compatible", [
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.0", "1.1.7", True, id="new-package-on-old-patch"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.7", "1.1.0", True, id="old-package-on-new-patch"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", True, id="minor-minimum"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.99", "1.1.123", True, id="arbitrary-patches"),
    pytest.param(["scheme>=1.0,<1.1"], "1.0.2", "1.0.9", True, id="old-minor"),
    pytest.param(["scheme>=2.3.0,<2.4.0"], "2.3.0", "2.3.7", True, id="another-major-old-patch"),
    pytest.param(["scheme>=2.3.0,<2.4.0"], "2.3.7", "2.3.0", True, id="another-major-new-patch"),
    pytest.param(["pydantic>=2,<3", "ScHeMe (<1.2.0, >=1.1.0)"], "1.1.0", "1.1.7", True, id="order-case-and-other-dependency"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.0", "1.0.7", False, id="package-older-minor"),
    pytest.param(["scheme>=1.0.0,<1.1.0"], "1.0.7", "1.1.0", False, id="package-newer-minor"),
    pytest.param(["scheme>=1.0.0,<1.1.0"], "1.1.0", "1.0.7", False, id="scheme-newer-minor"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.0.7", "1.1.0", False, id="scheme-older-minor"),
    pytest.param(["scheme>=1.0.0,<2.0.0"], "1.1.7", "1.0.7", False, id="same-major-old-range-old-package"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.0.7", "1.1.7", False, id="same-major-old-range-new-package"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.1.0", "1.1.7", False, id="same-minor-old-range"),
    pytest.param(["scheme>=1,<2"], "1.0.1", "1.0.1", False, id="old-minor-old-range"),
    pytest.param(["scheme>=1.1.1,<1.2.0"], "1.1.0", "1.1.7", False, id="below-patch-minimum"),
    pytest.param(["scheme>=1.1.1,<1.2.0"], "1.1.7", "1.1.0", False, id="above-patch-minimum"),
    pytest.param(["scheme>=1.1.7,<1.2.0"], "1.1.7", "1.1.7", False, id="matching-patch-minimum"),
    pytest.param(["scheme>=1.1.0,<1.1.8"], "1.1.7", "1.1.0", False, id="patch-upper-bound"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "2.1.0", "1.1.0", False, id="scheme-major-mismatch"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.0", "2.1.0", False, id="package-major-mismatch"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.2.0", "1.2.0", False, id="outside-upper-bound"),
    pytest.param([], "1.1.0", "1.1.0", False, id="missing"),
    pytest.param(["scheme>=1.1.0,<1.2.0", "Scheme>=1.1,<1.2"], "1.1.0", "1.1.0", False, id="multiple"),
    pytest.param(["scheme @ https://example.invalid/scheme.whl"], "1.1.0", "1.1.0", False, id="url"),
    pytest.param(["scheme @ git+https://example.invalid/scheme.git@deadbeef"], "1.1.0", "1.1.0", False, id="git-url"),
    pytest.param(['scheme>=1.1.0,<1.2.0; python_version >= "3.12"'], "1.1.0", "1.1.0", False, id="marker"),
    pytest.param(["scheme[extra]>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="extras"),
    pytest.param(["scheme==1.1.0"], "1.1.0", "1.1.0", False, id="exact"),
    pytest.param(["scheme>=1.1.0,<1.2.0,==1.1.0"], "1.1.0", "1.1.0", False, id="range-and-pin"),
    pytest.param(["scheme>=1.1.0"], "1.1.0", "1.1.0", False, id="missing-upper"),
    pytest.param(["scheme<1.2.0"], "1.1.0", "1.1.0", False, id="missing-lower"),
    pytest.param(["scheme>=0.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="cross-major-lower"),
    pytest.param(["scheme>=1.1.0,<3.0.0"], "1.1.0", "1.1.0", False, id="cross-major-upper"),
    pytest.param(["scheme>=1.0.0,<1.2.0"], "1.1.0", "1.1.0", False, id="cross-minor-lower"),
    pytest.param(["scheme>=1.1.0,<1.3.0"], "1.1.0", "1.1.0", False, id="cross-minor-upper"),
    pytest.param(["scheme>=1.1.0,<=1.2.0"], "1.1.0", "1.1.0", False, id="inclusive-upper"),
    pytest.param(["scheme>1.1.0,<1.2.0"], "1.1.7", "1.1.0", False, id="exclusive-lower"),
    pytest.param(["scheme>=1.1.0,<1.2.0,!=1.1.2"], "1.1.0", "1.1.0", False, id="excluded-version"),
    pytest.param(["scheme~=1.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="compatible-operator"),
    pytest.param(["scheme>=1.0.0,>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="multiple-lower"),
    pytest.param(["scheme>=1.1.0,<1.2.0,<1.3.0"], "1.1.0", "1.1.0", False, id="multiple-upper"),
    pytest.param(["scheme>=1.1.0,<1.2.0,<1.2.0"], "1.1.0", "1.1.0", False, id="duplicate-upper"),
    pytest.param(["scheme>=1.1.0,>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="duplicate-lower"),
    pytest.param(["scheme>=1.1.0,>=1.1,<1.2.0"], "1.1.0", "1.1.0", False, id="equivalent-duplicate-lower"),
    pytest.param(["scheme>=1,<1.1"], "1.0.1", "1.0.1", True, id="one-part-lower"),
    pytest.param(["scheme>=1.1,<1.2"], "1.1.0", "1.1.7", True, id="shorthand"),
    pytest.param(["scheme>=1.1,<1.2.0"], "1.1.0", "1.1.7", True, id="shorthand-lower"),
    pytest.param(["scheme>=1.1.0,<1.2"], "1.1.7", "1.1.0", True, id="shorthand-upper"),
    pytest.param(["scheme>=1.1.0rc1,<1.2.0"], "1.1.0", "1.1.0", False, id="prerelease-lower"),
    pytest.param(["scheme>=1.1.0.dev1,<1.2.0"], "1.1.0", "1.1.0", False, id="dev-lower"),
    pytest.param(["scheme>=1.1.0.post1,<1.2.0"], "1.1.1", "1.1.0", False, id="post-lower"),
    pytest.param(["scheme>=1.1.0+local,<1.2.0"], "1.1.0", "1.1.0", False, id="local-lower"),
    pytest.param(["scheme>=1!1.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="epoch-lower"),
    pytest.param(["scheme>=1.1.0.0,<1.2.0"], "1.1.0", "1.1.0", False, id="four-part-lower"),
    pytest.param(["scheme>=1.1.0,<1.2.0.0"], "1.1.0", "1.1.0", False, id="four-part-upper"),
    pytest.param(["scheme>=1.1.0,<1.2.0rc1"], "1.1.0", "1.1.0", False, id="prerelease-upper"),
    pytest.param(["scheme>=not-a-version,<1.2.0"], "1.1.0", "1.1.0", False, id="malformed-requirement"),
])
def test_wheel_scheme_compatibility(candidate, monkeypatch, requirements, scheme_version, package_version, compatible):
    component, _, dist = candidate
    component.version = dist.version = package_version
    dist.requires = requirements
    monkeypatch.setattr("scheme.execute.packages.metadata.version", lambda _: scheme_version)
    if compatible:
        assert verify_component(component) is dist
    else:
        with pytest.raises(ValueError):
            verify_component(component)


@pytest.mark.parametrize("version", [
    "not-a-version", "1!1.1.0", "1.1.0rc1", "1.1.0.dev1",
    "1.1.0.post1", "1.1.0+local", "1.1.0.0", "1.1.0.1",
])
@pytest.mark.parametrize("target", ["scheme", "package"])
def test_wheel_rejects_non_release_versions(candidate, monkeypatch, target, version):
    component, _, dist = candidate
    component.version = dist.version = version if target == "package" else "1.1.0"
    dist.requires = ["scheme>=1.1.0,<1.2.0"]
    scheme_version = version if target == "scheme" else "1.1.0"
    monkeypatch.setattr("scheme.execute.packages.metadata.version", lambda _: scheme_version)
    with pytest.raises(ValueError):
        verify_component(component)


@pytest.mark.parametrize("scheme_version,package_version", [
    ("1", "1.0.7"), ("1.0.7", "1"), ("1.1", "1.1.7"), ("1.1.7", "1.1"),
])
def test_wheel_accepts_equivalent_release_version_segments(
    candidate, monkeypatch, scheme_version, package_version,
):
    component, _, dist = candidate
    component.version = dist.version = package_version
    dist.requires = ["scheme>=1,<1.1"] if scheme_version == "1" or package_version == "1" else [
        "scheme>=1.1,<1.2"
    ]
    monkeypatch.setattr("scheme.execute.packages.metadata.version", lambda _: scheme_version)
    assert verify_component(component) is dist


@pytest.mark.parametrize("actual_version,locked_version", [("1.1.0", "1.1.7"), ("1.1.7", "1.1.0")])
def test_environment_lock_keeps_exact_scheme_patch(tmp_path, monkeypatch, actual_version, locked_version):
    lock = tmp_path / "uv.lock"
    lock.write_text(f'[[package]]\nname="scheme"\nversion="{locked_version}"\n', encoding="utf-8")
    dist = SimpleNamespace(metadata={"Name": "scheme"}, version=actual_version, requires=[])
    monkeypatch.setattr("scheme.execute.packages.metadata.distributions", lambda: [dist])
    with pytest.raises(ValueError, match="不在任务锁文件"):
        validate_environment(Environment(lockfile=lock))


def test_context_mismatch():
    class Params(BaseModel):
        pass

    class A(ResearchContext[float]):
        pass

    class B(ResearchContext[str]):
        pass

    class Example(Algo[Params, A]):
        pass

    with pytest.raises(TypeError, match="需要"):
        validate_context(Example(Params()), B())
    validate_context(Example(Params()), A())
