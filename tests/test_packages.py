from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from scheme import Algo, ResearchContext
from scheme.execute.packages import installed_versions, verify_component
from scheme.execute.schema import Component
from scheme.execute.strategy.assembly import validate_context


@pytest.fixture
def candidate(monkeypatch):
    dist = SimpleNamespace(version="1.0.0", requires=["scheme>=1,<1.1"])
    monkeypatch.setattr("scheme.execute.packages.metadata.distribution", lambda _: dist)
    monkeypatch.setattr("scheme.execute.packages.metadata.version", lambda _: "1.0.0")
    component = Component(
        package="demo", version="1.0.0", wheel="candidate.whl", entry="algo:Algo",
    )
    return component, dist


def test_component_uses_installed_distribution(candidate):
    component, dist = candidate
    assert verify_component(component) is dist


def test_wrong_installed_version(candidate):
    component, dist = candidate
    dist.version = "1.0.7"
    with pytest.raises(ValueError, match="实际安装版本"):
        verify_component(component)


def test_wrong_scheme_major(candidate):
    component, dist = candidate
    component.version = dist.version = "2.0.0"
    with pytest.raises(ValueError, match="主版本"):
        verify_component(component)


def test_missing_scheme_requirement(candidate):
    component, dist = candidate
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
    pytest.param(["scheme>=1.1.0,<1.1.8"], "1.1.7", "1.1.0", True, id="patch-upper-bound"),
    pytest.param(["scheme>=1.1.0,<1.1.8"], "1.1.8", "1.1.0", False, id="current-above-upper"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "2.1.0", "1.1.0", False, id="scheme-major-mismatch"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.1.0", "2.1.0", False, id="package-major-mismatch"),
    pytest.param(["scheme>=1.1.0,<1.2.0"], "1.2.0", "1.2.0", False, id="outside-upper-bound"),
    pytest.param([], "1.1.0", "1.1.0", False, id="missing"),
    pytest.param(["scheme>=1.1.0,<1.2.0", "Scheme>=1.1,<1.2"], "1.1.0", "1.1.0", False, id="multiple"),
    pytest.param(["scheme @ https://example.invalid/scheme.whl"], "1.1.0", "1.1.0", False, id="url"),
    pytest.param(["scheme @ git+https://example.invalid/scheme.git@deadbeef"], "1.1.0", "1.1.0", False, id="git-url"),
    pytest.param(['scheme>=1.1.0,<1.2.0; python_version >= "3.12"'], "1.1.0", "1.1.0", True, id="marker"),
    pytest.param(["scheme[extra]>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", True, id="extras"),
    pytest.param(["scheme==1.1.0"], "1.1.0", "1.1.0", True, id="exact"),
    pytest.param(["scheme==1.1.0"], "1.1.7", "1.1.0", False, id="exact-excludes-current"),
    pytest.param(["scheme>=1.1.0,<1.2.0,==1.1.0"], "1.1.0", "1.1.0", True, id="range-and-pin"),
    pytest.param(["scheme>=1.1.0"], "1.1.0", "1.1.0", False, id="missing-upper"),
    pytest.param(["scheme<1.2.0"], "1.1.0", "1.1.0", True, id="missing-lower"),
    pytest.param(["scheme>=0.1.0,<1.2.0"], "1.1.0", "1.1.0", True, id="cross-major-lower"),
    pytest.param(["scheme>=1.1.0,<3.0.0"], "1.1.0", "1.1.0", False, id="cross-major-upper"),
    pytest.param(["scheme>=1.0.0,<1.2.0"], "1.1.0", "1.1.0", True, id="cross-minor-lower"),
    pytest.param(["scheme>=1.1.0,<1.3.0"], "1.1.0", "1.1.0", False, id="cross-minor-upper"),
    pytest.param(["scheme>=1.1.0,<=1.2.0"], "1.1.0", "1.1.0", False, id="inclusive-upper"),
    pytest.param(["scheme>1.1.0,<1.2.0"], "1.1.7", "1.1.0", False, id="exclusive-lower"),
    pytest.param(["scheme>=1.1.0,<1.2.0,!=1.1.2"], "1.1.0", "1.1.0", True, id="excluded-other-version"),
    pytest.param(["scheme>=1.1.0,<1.2.0,!=1.1.2"], "1.1.2", "1.1.0", False, id="excluded-current-version"),
    pytest.param(["scheme>=1.1.0,<1.2.0,!=1.1.0"], "1.1.7", "1.1.0", False, id="excluded-series-minimum"),
    pytest.param(["scheme~=1.1.0,<1.2.0"], "1.1.0", "1.1.0", True, id="compatible-with-upper"),
    pytest.param(["scheme~=1.6.0"], "1.6.1", "1.6.0", True, id="compatible-operator"),
    pytest.param(["scheme==1.6.*"], "1.6.1", "1.6.0", True, id="wildcard"),
    pytest.param(["scheme>=1.0.0,>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", True, id="multiple-lower"),
    pytest.param(["scheme>=1.1.0,<1.2.0,<1.3.0"], "1.1.0", "1.1.0", True, id="multiple-upper"),
    pytest.param(["scheme>=1.1.0,<1.2.0,<1.2.0"], "1.1.0", "1.1.0", True, id="duplicate-upper"),
    pytest.param(["scheme>=1.1.0,>=1.1.0,<1.2.0"], "1.1.0", "1.1.0", True, id="duplicate-lower"),
    pytest.param(["scheme>=1.1.0,>=1.1,<1.2.0"], "1.1.0", "1.1.0", True, id="equivalent-duplicate-lower"),
    pytest.param(["scheme>=1,<1.1"], "1.0.1", "1.0.1", True, id="one-part-lower"),
    pytest.param(["scheme>=1.1,<1.2"], "1.1.0", "1.1.7", True, id="shorthand"),
    pytest.param(["scheme>=1.1,<1.2.0"], "1.1.0", "1.1.7", True, id="shorthand-lower"),
    pytest.param(["scheme>=1.1.0,<1.2"], "1.1.7", "1.1.0", True, id="shorthand-upper"),
    pytest.param(["scheme>=1.1.0rc1,<1.2.0"], "1.1.0", "1.1.0", True, id="prerelease-lower"),
    pytest.param(["scheme>=1.1.0.dev1,<1.2.0"], "1.1.0", "1.1.0", True, id="dev-lower"),
    pytest.param(["scheme>=1.1.0.post1,<1.2.0"], "1.1.1", "1.1.0", False, id="post-lower"),
    pytest.param(["scheme>=1.1.0+local,<1.2.0"], "1.1.0", "1.1.0", False, id="local-lower"),
    pytest.param(["scheme>=1!1.1.0,<1.2.0"], "1.1.0", "1.1.0", False, id="epoch-lower"),
    pytest.param(["scheme>=1.1.0.0,<1.2.0"], "1.1.0", "1.1.0", True, id="four-part-lower"),
    pytest.param(["scheme>=1.1.0,<1.2.0.0"], "1.1.0", "1.1.0", True, id="four-part-upper"),
    pytest.param(["scheme>=1.1.0,<1.2.0rc1"], "1.1.0", "1.1.0", True, id="prerelease-upper"),
    pytest.param(["scheme>=not-a-version,<1.2.0"], "1.1.0", "1.1.0", False, id="malformed-requirement"),
])
def test_wheel_scheme_compatibility(candidate, monkeypatch, requirements, scheme_version, package_version, compatible):
    component, dist = candidate
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
    component, dist = candidate
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
    component, dist = candidate
    component.version = dist.version = package_version
    dist.requires = ["scheme>=1,<1.1"] if scheme_version == "1" or package_version == "1" else [
        "scheme>=1.1,<1.2"
    ]
    monkeypatch.setattr("scheme.execute.packages.metadata.version", lambda _: scheme_version)
    assert verify_component(component) is dist


def test_installed_versions_records_distribution_metadata(monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": "Scheme"}, version="1.1.0"),
        SimpleNamespace(metadata={"Name": "Arena_Runtime"}, version="2.0.0"),
    ]
    monkeypatch.setattr("scheme.execute.packages.metadata.distributions", lambda: distributions)
    assert installed_versions() == {"scheme": "1.1.0", "arena-runtime": "2.0.0"}


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
