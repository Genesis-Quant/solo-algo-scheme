import hashlib
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from pydantic import BaseModel

from scheme import Algo, ResearchContext
from scheme.utils.packages import validate_context, verify_component
from scheme.utils.schema import Component


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
    monkeypatch.setattr("scheme.utils.packages.metadata.distribution", lambda _: dist)
    monkeypatch.setattr("scheme.utils.packages.metadata.version", lambda _: "1.0.0")
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
