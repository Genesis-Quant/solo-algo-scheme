import hashlib
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pydantic import BaseModel

from scheme import Algo, ResearchContext
from scheme.execute.packages import verify_component
from scheme.execute.schema import Component
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
        version="1.0.0", requires=["scheme>=1,<2"], locate_file=lambda p: installed / p
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
    dist.version = "1.1.0"
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
    pytest.param(["scheme>=1.0.0,<2.0.0"], "1.0.0", "1.0.1", True, id="old-minimum"),
    pytest.param(["scheme>=1.0.0,<2.0.0"], "1.0.1", "1.0.1", True, id="old-patch"),
    pytest.param(["scheme>=1.0.0,<2.0.0"], "1.1.0", "1.0.1", True, id="old-on-new"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.0.0", "1.1.0", False, id="new-on-old"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.0.1", "1.1.0", False, id="new-on-old-patch"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.1.0", "1.1.0", True, id="new-minimum"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.9.9", "1.1.0", True, id="later-minor"),
    pytest.param(["scheme>=1.1.1,<2.0.0"], "1.1.0", "1.1.0", False, id="below-patch-minimum"),
    pytest.param(["scheme>=1.1.1,<2.0.0"], "1.1.1", "1.1.0", True, id="patch-minimum"),
    pytest.param(["pydantic>=2,<3", "ScHeMe (<2.0.0, >=1.1.0)"], "1.1.0", "1.2.0", True, id="order-case-and-other-dependency"),
    pytest.param(["scheme>=2.1.0,<3.0.0"], "2.1.0", "2.0.0", True, id="another-major"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "2.0.0", "1.1.0", False, id="scheme-major-mismatch"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "1.1.0", "2.0.0", False, id="package-major-mismatch"),
    pytest.param(["scheme>=1.1.0,<2.0.0"], "2.0.0", "2.0.0", False, id="outside-upper-bound"),
    pytest.param([], "1.1.0", "1.1.0", False, id="missing"),
    pytest.param(["scheme>=1.1.0,<2.0.0", "Scheme>=1.0.0,<2.0.0"], "1.1.0", "1.1.0", False, id="multiple"),
    pytest.param(["scheme @ https://example.invalid/scheme.whl"], "1.1.0", "1.1.0", False, id="url"),
    pytest.param(["scheme @ git+https://example.invalid/scheme.git@deadbeef"], "1.1.0", "1.1.0", False, id="git-url"),
    pytest.param(['scheme>=1.1.0,<2.0.0; python_version >= "3.12"'], "1.1.0", "1.1.0", False, id="marker"),
    pytest.param(["scheme[extra]>=1.1.0,<2.0.0"], "1.1.0", "1.1.0", False, id="extras"),
    pytest.param(["scheme==1.1.0"], "1.1.0", "1.1.0", False, id="exact"),
    pytest.param(["scheme>=1.1.0"], "1.1.0", "1.1.0", False, id="missing-upper"),
    pytest.param(["scheme<2.0.0"], "1.1.0", "1.1.0", False, id="missing-lower"),
    pytest.param(["scheme>=0.1.0,<2.0.0"], "1.1.0", "1.1.0", False, id="cross-major-lower"),
    pytest.param(["scheme>=1.1.0,<3.0.0"], "1.1.0", "1.1.0", False, id="cross-major-upper"),
    pytest.param(["scheme>=1.1.0,<1.9.0"], "1.1.0", "1.1.0", False, id="narrow-upper"),
    pytest.param(["scheme>=1.1.0,<=2.0.0"], "1.1.0", "1.1.0", False, id="inclusive-upper"),
    pytest.param(["scheme>1.0.0,<2.0.0"], "1.1.0", "1.1.0", False, id="exclusive-lower"),
    pytest.param(["scheme>=1.1.0,<2.0.0,!=1.2.0"], "1.1.0", "1.1.0", False, id="excluded-version"),
    pytest.param(["scheme~=1.1.0,<2.0.0"], "1.1.0", "1.1.0", False, id="compatible-operator"),
    pytest.param(["scheme>=1.0.0,>=1.1.0,<2.0.0"], "1.1.0", "1.1.0", False, id="multiple-lower"),
    pytest.param(["scheme>=1.1.0,<2.0.0,<3.0.0"], "1.1.0", "1.1.0", False, id="multiple-upper"),
    pytest.param(["scheme>=1.1.0,<2.0.0,<2.0.0"], "1.1.0", "1.1.0", False, id="duplicate-upper"),
    pytest.param(["scheme>=1,<2"], "1.0.1", "1.0.1", True, id="old-shorthand"),
    pytest.param(["scheme>=1.1,<2"], "1.1.0", "1.1.0", True, id="shorthand"),
    pytest.param(["scheme>=1.1,<2"], "1.0.1", "1.1.0", False, id="shorthand-below-minimum"),
    pytest.param(["scheme>=1.1,<2.0.0"], "1.1.0", "1.1.0", True, id="shorthand-lower"),
    pytest.param(["scheme>=1.1.0,<2"], "1.1.0", "1.1.0", True, id="shorthand-upper"),
    pytest.param(["scheme>=1.1.0rc1,<2.0.0"], "1.1.0", "1.1.0", False, id="prerelease-lower"),
    pytest.param(["scheme>=1.1.0.dev1,<2.0.0"], "1.1.0", "1.1.0", False, id="dev-lower"),
    pytest.param(["scheme>=1.1.0.post1,<2.0.0"], "1.1.1", "1.1.0", False, id="post-lower"),
    pytest.param(["scheme>=1.1.0+local,<2.0.0"], "1.1.0", "1.1.0", False, id="local-lower"),
    pytest.param(["scheme>=1!1.1.0,<2.0.0"], "1.1.0", "1.1.0", False, id="epoch-lower"),
    pytest.param(["scheme>=1.1.0.0,<2.0.0"], "1.1.0", "1.1.0", False, id="four-part-lower"),
    pytest.param(["scheme>=1.1.0,<2.0.0rc1"], "1.1.0", "1.1.0", False, id="prerelease-upper"),
    pytest.param(["scheme>=not-a-version,<2.0.0"], "1.1.0", "1.1.0", False, id="malformed"),
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
