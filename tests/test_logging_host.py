"""独立进程验证 Scheme 不接管宿主的标准库日志配置。"""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"


def run_logging(tmp_path, code, *args):
    script = tmp_path / "logging_case.py"
    script.write_text(dedent(code), encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("SCHEME_LOG_LEVEL", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SOURCE), environment.get("PYTHONPATH")))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), *args], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.parametrize("host_level", ["DEBUG", "INFO"])
def test_preexisting_file_handler_survives_import_and_reconfiguration(tmp_path, host_level):
    run_logging(tmp_path, """
        import logging
        import sys
        from io import StringIO
        from pathlib import Path

        root = logging.getLogger()
        assert not root.handlers
        host = logging.FileHandler("host.log", encoding="utf-8")
        host.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        host.setFormatter(formatter)
        root.addHandler(host)
        root.setLevel(sys.argv[1])
        host_level = root.level
        original_stream = host.stream
        expected = []

        def emit(stage):
            for name in ("httpx", "backtest"):
                for level in (logging.INFO, logging.WARNING):
                    message = f"{stage}-{name}-{logging.getLevelName(level)}"
                    logging.getLogger(name).log(level, message)
                    expected.append(message)

        def preserved():
            assert root.handlers == [host]
            assert root.level == host_level
            assert host.level == logging.DEBUG
            assert host.formatter is formatter
            assert host.stream is original_stream
            assert not original_stream.closed

        emit("before-import")
        import scheme.log as scheme_log
        preserved()
        emit("after-import")
        output = StringIO()
        for index, level in enumerate(("INFO", "INFO", "DEBUG", "WARNING", "ERROR", "INFO")):
            scheme_log.configure(level, stream=output)
            preserved()
            emit(f"configured-{index}-{level}")

        host.flush()
        assert Path("host.log").read_text(encoding="utf-8").splitlines() == expected
        assert output.getvalue() == ""
        preserved()
    """, host_level)


@pytest.mark.parametrize("host_level", ["DEBUG", "INFO"])
def test_file_handler_added_after_scheme_is_preserved(tmp_path, host_level):
    run_logging(tmp_path, """
        import logging
        import sys
        from io import StringIO
        from pathlib import Path

        import scheme.log as scheme_log

        root = logging.getLogger()
        assert len(root.handlers) == 1
        bridge = root.handlers[0]
        assert isinstance(bridge, scheme_log._Intercept)
        output = StringIO()
        scheme_log.configure("INFO", stream=output)
        host = logging.FileHandler("host.log", encoding="utf-8")
        host.setLevel(logging.DEBUG)
        host.setFormatter(logging.Formatter("%(message)s"))
        original_stream = host.stream
        root.addHandler(host)
        root.setLevel(sys.argv[1])
        host_level = root.level
        expected = []

        def emit(stage):
            for name in ("httpx", "backtest"):
                for level in (logging.INFO, logging.WARNING):
                    message = f"{stage}-{name}-{logging.getLevelName(level)}"
                    logging.getLogger(name).log(level, message)
                    expected.append(message)

        emit("before-reconfigure")
        for index, level in enumerate(("INFO", "INFO", "DEBUG", "ERROR", "WARNING", "INFO")):
            scheme_log.configure(level, stream=output)
            assert root.handlers == [bridge, host]
            assert root.level == host_level
            assert host.level == logging.DEBUG
            assert host.stream is original_stream
            assert not original_stream.closed
            stage = f"configured-{index}-{level}"
            emit(stage)
            text = output.getvalue()
            assert f"{stage}-httpx-INFO" not in text
            assert text.count(f"{stage}-backtest-INFO") == int(level in ("INFO", "DEBUG"))
            assert text.count(f"{stage}-httpx-WARNING") == int(level != "ERROR")

        host.flush()
        assert Path("host.log").read_text(encoding="utf-8").splitlines() == expected
        assert output.getvalue().count("before-reconfigure-backtest-INFO") == 1
        assert host.stream is original_stream and not original_stream.closed
    """, host_level)


def test_standalone_bridge_updates_levels_without_duplicates_and_keeps_source(tmp_path):
    run_logging(tmp_path, r"""
        import logging
        import re
        from io import StringIO

        root = logging.getLogger()
        assert not root.handlers
        import scheme.log as scheme_log

        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        bridge = root.handlers[0]
        assert isinstance(bridge, scheme_log._Intercept)
        output = StringIO()
        expected = []

        def emit(name, level, message):
            logging.getLogger(name).log(level, message)

        for level in ("INFO", "INFO"):
            scheme_log.configure(level, stream=output)
            assert root.handlers == [bridge]
            assert root.level == logging.INFO
        for name in ("backtest", "model_1365.algo", "__main__"):
            message = f"{name}-info"
            emit(name, logging.INFO, message)
            expected.append(message)
        emit("backtest", logging.DEBUG, "hidden-project-debug")
        emit("httpx", logging.INFO, "hidden-third-party-info")
        emit("httpx", logging.WARNING, "third-party-warning")
        expected.append("third-party-warning")

        for level in ("DEBUG", "DEBUG"):
            scheme_log.configure(level, stream=output)
            assert root.handlers == [bridge]
            assert root.level == logging.DEBUG
        emit("backtest", logging.DEBUG, "project-debug")
        emit("model_1365.algo", logging.INFO, "project-info-at-debug")
        expected.extend(("project-debug", "project-info-at-debug"))
        emit("httpx", logging.DEBUG, "hidden-third-party-debug")
        emit("httpx", logging.INFO, "still-hidden-third-party-info")

        root.setLevel(logging.ERROR)
        scheme_log.configure("DEBUG", stream=output)
        assert root.handlers == [bridge]
        assert root.level == logging.DEBUG
        emit("backtest", logging.DEBUG, "debug-after-repeat")
        expected.append("debug-after-repeat")
        scheme_log.configure("WARNING", stream=output)
        assert root.handlers == [bridge]
        assert root.level == logging.WARNING
        emit("backtest", logging.INFO, "hidden-project-info-at-warning")
        emit("backtest", logging.WARNING, "project-warning")
        expected.append("project-warning")

        arena_line = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\s*([A-Z]+)\s*\] "
            r"([^|]+?) \| ([^|]+?): (\d+)行\| (.*)$"
        )
        parsed = []
        for line in output.getvalue().splitlines():
            match = arena_line.fullmatch(line)
            assert match is not None, line
            parsed.append(match.groups())
        assert [row[-1] for row in parsed] == expected
        assert all(row[2:4] == ("logging_case.py", "emit") for row in parsed)
        assert all(int(row[4]) > 0 for row in parsed)
    """)


def test_is_enabled_follows_scheme_threshold_not_host_root(tmp_path):
    run_logging(tmp_path, """
        import logging
        import os
        from io import StringIO

        root = logging.getLogger()
        host = logging.NullHandler()
        root.addHandler(host)
        root.setLevel(logging.CRITICAL)
        import scheme.log as scheme_log
        from loguru import logger

        output = StringIO()
        logger.level("MONTHLY", no=25)
        cases = (
            ("INFO", ("INFO", "MONTHLY", "WARN", "ERROR", "FATAL"), ("DEBUG",)),
            ("warn", ("WARN", "ERROR", "FATAL"), ("DEBUG", "INFO", "MONTHLY")),
            ("DEBUG", ("DEBUG", "INFO", "MONTHLY", "WARN", "ERROR", "FATAL"), ()),
            ("MONTHLY", ("MONTHLY", "WARN", "ERROR", "FATAL"), ("DEBUG", "INFO")),
            ("ERROR", ("ERROR", "FATAL"), ("DEBUG", "INFO", "MONTHLY", "WARN")),
        )
        for level, enabled, disabled in cases:
            for _ in range(2):
                scheme_log.configure(level, stream=output)
                for name in enabled:
                    assert scheme_log.is_enabled(f" {name.lower()} ") is True
                for name in disabled:
                    assert scheme_log.is_enabled(name) is False
                assert root.handlers == [host]
                assert root.level == logging.CRITICAL

        try:
            scheme_log.configure("invalid", stream=output)
        except ValueError:
            pass
        else:
            raise AssertionError("explicit invalid levels must fail")
        assert scheme_log.is_enabled("WARN") is False
        os.environ[scheme_log.LEVEL_VARIABLE] = "debug"
        scheme_log.configure(stream=output)
        assert scheme_log.is_enabled("DEBUG") is True
        os.environ[scheme_log.LEVEL_VARIABLE] = "invalid"
        scheme_log.configure(stream=output)
        assert scheme_log.is_enabled("DEBUG") is False
        assert scheme_log.is_enabled("INFO") is True
        assert "已按 INFO 输出日志" in output.getvalue()
        assert root.handlers == [host] and root.level == logging.CRITICAL
    """)


@pytest.mark.parametrize("mode", ["cli", "notebook"])
def test_default_stream_remains_cli_stderr_and_notebook_stdout(tmp_path, mode):
    run_logging(tmp_path, """
        import sys
        from io import StringIO

        class Output(StringIO):
            pass

        if sys.argv[1] == "notebook":
            Output.__module__ = "ipykernel.iostream"
        stdout, stderr = Output(), StringIO()
        original_streams = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout, stderr
            import scheme.log as scheme_log

            scheme_log.configure("INFO")
            scheme_log.get_logger("scheme").info("default-stream-marker")
            expected, other = (stdout, stderr) if sys.argv[1] == "notebook" else (stderr, stdout)
            assert "default-stream-marker" in expected.getvalue()
            assert "default-stream-marker" not in other.getvalue()
        finally:
            sys.stdout, sys.stderr = original_streams
    """, mode)
