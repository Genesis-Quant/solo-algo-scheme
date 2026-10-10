"""Offline contracts for the Solo Jupyter provider and standalone HTML export."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scheme.report.base import Report


@pytest.fixture
def legacy_notebook_html(monkeypatch):
    render = Mock(side_effect=AssertionError("show must not use notebook_html/srcdoc"))
    monkeypatch.setattr("scheme.report.notebook_html", render)
    yield render
    render.assert_not_called()


def mock_ipython(monkeypatch, shell):
    module = ModuleType("IPython")
    module.get_ipython = Mock(return_value=shell)
    monkeypatch.setitem(sys.modules, "IPython", module)
    # Providers own display; Report.show must not import or display a second time.
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    return module.get_ipython


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, {"height": 1000, "theme": "light", "annual_trading_days": 252,
              "risk_free_rate": 0.0}),
        ({"height": 720, "theme": "dark", "annual_trading_days": 250,
          "risk_free_rate": 0.025},
         {"height": 720, "theme": "dark", "annual_trading_days": 250,
          "risk_free_rate": 0.025}),
    ],
    ids=["defaults", "custom-ui-and-statistics"],
)
def test_show_delegates_same_report_and_returns_none(
    monkeypatch, legacy_notebook_html, options, expected,
):
    report = Report()
    provider = Mock(return_value=object())
    get_ipython = mock_ipython(monkeypatch, SimpleNamespace(solo_report_display=provider))

    assert report.show(**options) is None

    get_ipython.assert_called_once_with()
    provider.assert_called_once_with(report, **expected)
    assert provider.call_args.args[0] is report


@pytest.mark.parametrize(
    "shell",
    [None, SimpleNamespace(), SimpleNamespace(solo_report_display=None),
     SimpleNamespace(solo_report_display="not-callable")],
    ids=["no-shell", "missing-provider", "none-provider", "noncallable-provider"],
)
def test_show_without_provider_has_actionable_error(
    monkeypatch, legacy_notebook_html, shell,
):
    mock_ipython(monkeypatch, shell)

    with pytest.raises(RuntimeError) as error:
        Report().show()

    message = str(error.value)
    assert "Solo" in message
    assert "Jupyter Server" in message
    assert "启动桥接" in message
    assert "solo_report_display" in message
    assert "report.to_html()" in message


def test_show_without_ipython_has_actionable_error(monkeypatch, legacy_notebook_html):
    monkeypatch.setitem(sys.modules, "IPython", None)

    with pytest.raises(RuntimeError, match=r"Solo.*Jupyter Server.*启动桥接.*to_html") as error:
        Report().show()

    assert isinstance(error.value.__cause__, ImportError)


def test_show_leaves_validation_and_errors_to_provider(monkeypatch, legacy_notebook_html):
    invalid = {"height": True, "theme": "invalid", "annual_trading_days": False,
               "risk_free_rate": float("inf")}
    failure = ValueError("provider validation failed")
    provider = Mock(side_effect=failure)
    mock_ipython(monkeypatch, SimpleNamespace(solo_report_display=provider))
    report = Report()

    with pytest.raises(ValueError) as error:
        report.show(**invalid)

    assert error.value is failure
    provider.assert_called_once_with(report, **invalid)


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, {"theme": "light", "annual_trading_days": 252, "risk_free_rate": 0.0}),
        ({"theme": "dark", "annual_trading_days": 250, "risk_free_rate": 0.025},
         {"theme": "dark", "annual_trading_days": 250, "risk_free_rate": 0.025}),
    ],
    ids=["defaults", "custom-statistics"],
)
def test_to_html_still_uses_standalone_html_without_ipython(monkeypatch, options, expected):
    monkeypatch.setitem(sys.modules, "IPython", None)
    document = "<!doctype html><html>standalone export</html>"
    render = Mock(return_value=document)
    monkeypatch.setattr("scheme.report.report_html", render)
    report = Report()

    assert report.to_html(**options) is document

    render.assert_called_once_with(report, **expected)
    assert render.call_args.args[0] is report
