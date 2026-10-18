"""离线验证公开回测 wrapper；不导入真实 Runtime 配置或连接 DolphinDB。"""

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_DATA = Path(__file__).resolve().parents[1] / "src" / "scheme" / "data"
_SOURCE = _DATA / "dolphindb"


class FakeSession:
    def __init__(self):
        self.calls = []
        self.uploads = []
        self.close_count = 0
        self.msg_logger = object()
        self.fail = False

    def run(self, script, *args, **kwargs):
        self.calls.append((script, args, kwargs))
        if self.fail:
            raise RuntimeError("session failed")
        return script

    def upload(self, variables):
        self.uploads.append(variables)

    def close(self):
        self.close_count += 1


class FakeResult:
    def __init__(self, session):
        self.session = session
        self.closed = False

    @property
    def return_summary(self):
        return self.session.run(
            "backtest::standardize_return_summary("
            "Backtest::getReturnSummary(coreBacktestEngine), "
            "Backtest::getDailyTotalPortfolios(coreBacktestEngine), "
            "coreBacktestAnnualTradingDays, coreBacktestRiskFreeRate)"
        )

    def close(self):
        if not self.closed:
            self.session.run("Backtest::dropBacktestEngine(coreBacktestEngine)")
            self.session.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _load(monkeypatch, name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inline(monkeypatch):
    runtime = ModuleType("runtime.apps.backtest.api")
    runtime.BACKTEST_SESSION_MAX_TIME = 3600
    runtime.created = []
    runtime.fail = False

    def create_session(*, max_time):
        session = FakeSession()
        runtime.created.append((session, max_time))
        return session

    # 保留 Runtime 当前 public API 形状；fake 仅模拟本 wrapper 使用的协议。
    def run_backtest(
        dataset_query, callbacks, *, session=None, market_source="daily", codes_query=None,
        utils="", params=None, name=None, config=None, adj=None, risk_free_rate=0.04,
        annual_trading_days=250, source_ref="coreBacktestSourceData",
        message_ref="coreBacktestMessage",
    ):
        runtime.last_call = locals().copy()
        assert session is not None
        session.run('loadPlugin("MatchingEngineSimulator"); loadPlugin("Backtest")')
        session.run("use query")
        for _ in range(2):
            session.run("use factor")
            session.run("use backtest")
        if utils:
            session.run(utils)
        session.run("\n".join(callbacks.values()))
        if market_source == "snapshot":
            session.run("""
                coreBacktestSnapshotState = dict(STRING,ANY)
                coreBacktestSnapshotState["enabled"] = false
                coreBacktestEngine = backtest::run_snapshot_backtest(
                    coreBacktestName, coreBacktestConfig, coreBacktestSnapshotTable,
                    coreBacktestSnapshotReference, coreBacktestCodes, coreBacktestTradeDates,
                    coreBacktestBenchmarkMessage,
                    initialize, beforeTrading, onBar, onSnapshot, onOrder, onTrade,
                    afterTrading, finalize
                )
            """)
        else:
            session.run(f"""
                coreBacktestTradeDates = coreBacktestAvailableTradeDates
                coreBacktestEngine = backtest::run_backtest(
                    coreBacktestName, coreBacktestConfig, {message_ref},
                    initialize, beforeTrading, onBar, onSnapshot, onOrder, onTrade,
                    afterTrading, finalize
                )
            """)
        if runtime.fail:
            raise RuntimeError("runtime failed")
        return FakeResult(session)

    runtime.create_session = create_session
    runtime.run_backtest = run_backtest
    packages = ("runtime", "runtime.apps", "runtime.apps.backtest", "runtime.apps.query")
    for name in packages:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "runtime.apps.backtest.api", runtime)
    sys.modules["runtime.apps.backtest"].api = runtime
    def query(*args, **kwargs):
        return None

    query_api = ModuleType("runtime.apps.query.api")
    query_api.execute_query = query
    query_result = ModuleType("runtime.apps.query.result")
    query_result.QueryResult = type("QueryResult", (), {})
    monkeypatch.setitem(sys.modules, "runtime.apps.query.api", query_api)
    monkeypatch.setitem(sys.modules, "runtime.apps.query.result", query_result)
    sys.modules["runtime.apps.query"].api = query_api
    sys.modules["runtime.apps.query"].FactorQuery = type("FactorQuery", (), {})
    sys.modules["runtime.apps.query"].execute_query = query

    # 独立包名避免触发 scheme.__init__ 和真实配置；实际包源码原样加载。
    package_name = "_scheme_inline_backtest_test"
    _load(monkeypatch, package_name, _SOURCE / "__init__.py", package=True)
    module = _load(monkeypatch, f"{package_name}.backtest", _SOURCE / "backtest.py")
    api = _load(monkeypatch, f"{package_name}.api", _SOURCE / "api.py")
    data_name = "_scheme_inline_data_test"
    data_package = ModuleType(data_name)
    data_package.__path__ = [str(_DATA)]
    monkeypatch.setitem(sys.modules, data_name, data_package)
    monkeypatch.setitem(sys.modules, f"{data_name}.dolphindb", sys.modules[package_name])
    monkeypatch.setitem(sys.modules, f"{data_name}.dolphindb.api", api)
    data = _load(monkeypatch, data_name, _DATA / "__init__.py", package=True)
    return SimpleNamespace(module=module, runtime=runtime, api=api, data=data, query=query)


def _scripts(session):
    return [call[0] for call in session.calls]


def _inline_loads(session):
    return [script for script in _scripts(session) if "def getParams()" in script]


def test_public_api_signature_and_binding(inline):
    assert inline.data.query is inline.api.query
    assert inline.api.backtest is inline.module.backtest
    assert inline.data.backtest is inline.module.backtest
    assert inline.module.backtest.__wrapped__ is inline.runtime.run_backtest
    assert inspect.signature(inline.module.backtest) == inspect.signature(inline.runtime.run_backtest)
    borrowed = FakeSession()
    callbacks = {"initialize": "def initialize(mutable context) { getParam(`size) }"}
    result = inline.data.backtest(
        {"factors": ["close"]}, callbacks, session=borrowed, market_source="snapshot",
        params={"size": 100}, name="test", source_ref="source", message_ref="message",
    )
    assert inline.runtime.last_call["callbacks"] is callbacks
    assert inline.runtime.last_call["market_source"] == "snapshot"
    assert inline.runtime.last_call["source_ref"] == "source"
    assert inline.runtime.last_call["message_ref"] == "message"
    assert result.session.raw_session is borrowed
    assert not inline.runtime.created
    with pytest.raises(TypeError):
        inline.data.backtest({}, {}, None)
    assert not inline.runtime.created


def test_bootstrap_loads_once_and_delegates(inline):
    raw = FakeSession()
    proxy = inline.module._InlineBacktestSession(raw)
    assert proxy.run("use factor") is None
    proxy.run("use backtest")
    proxy.run(" use backtest; ")
    proxy.run("use factor;")
    proxy.run("use query", 1, clearMemory=True)
    proxy.upload({"coreBacktestParams": {"size": 100}})
    assert len(_inline_loads(raw)) == 1
    assert raw.calls[-1] == ("use query", (1,), {"clearMemory": True})
    assert raw.uploads == [{"coreBacktestParams": {"size": 100}}]
    assert proxy.msg_logger is raw.msg_logger
    assert not any(script.strip() in {"use factor", "use backtest"} for script in _scripts(raw))


def test_failed_inline_load_is_not_marked_loaded(inline):
    raw = FakeSession()
    proxy = inline.module._InlineBacktestSession(raw)
    raw.fail = True
    with pytest.raises(RuntimeError, match="session failed"):
        proxy.run("use backtest")
    raw.fail = False
    proxy.run("use backtest")
    proxy.run("use backtest")
    assert len(_inline_loads(raw)) == 2


@pytest.mark.parametrize("name", [
    "getParams", "getParam", "getIndustry", "getTradeDates", "getHistoryData",
    "getLastData", "getMinuteHistory", "order_target", "order_target_value",
    "backtest_order_quote",
    "read_stock_snapshots", "build_minute_bars", "build_stock_snapshot_message",
    "snapshot_callback",
])
def test_known_qualified_calls_function_objects_and_partials(inline, name):
    source = f"backtest::{name}(x); f=backtest::{name}; g=backtest::{name}{{,x}}"
    expected = f"{name}(x); f={name}; g={name}{{,x}}"
    assert inline.module._rewrite_script(source) == expected


def test_literals_comments_symbols_and_plugins_unchanged(inline):
    protected = r'''// backtest::getParam(x); use factor
/* backtest::order_target(ctx); use ta */
a = "backtest::getParam(\"name\") use factor"
b = 'backtest::getParam(\'name\') use common'
c = `backtest::getParam`backtest::order_target
columns = `use`factor`query::submodule
p = Backtest::getPosition(engine, code)
loadPlugin("MatchingEngineSimulator")
loadPlugin("Backtest")
'''
    assert inline.module._rewrite_script(protected) == protected
    source = protected + "f=backtest::getParam; getParam(`size)"
    assert inline.module._rewrite_script(source) == protected + "f=getParam; getParam(`size)"


@pytest.mark.parametrize("script", [
    "use unknown", "use common", "use ta", "use factor; x=1", "use backtest; x=1",
    "x=1; use factor", "use /* comment */ factor", "use", "use `query", 'use "query"',
    "use query::submodule",
])
def test_proxy_rejects_other_module_uses(inline, script):
    raw = FakeSession()
    proxy = inline.module._InlineBacktestSession(raw)
    with pytest.raises(ValueError, match="仅允许 use query"):
        proxy.run(script)
    assert not raw.calls


def test_unknown_qualified_function_is_not_forwarded(inline):
    raw = FakeSession()
    with pytest.raises(ValueError, match="未内联回测函数"):
        inline.module._InlineBacktestSession(raw).run("f=backtest::unknown")
    assert not raw.calls


@pytest.mark.parametrize("script", ["use factor", "use backtest", "use ta", "use common"])
@pytest.mark.parametrize("location", ["utils", "callbacks"])
def test_user_module_uses_rejected_before_session_creation(inline, script, location):
    callbacks = {"initialize": "def initialize(mutable context) { return NULL }"}
    kwargs = {}
    if location == "utils":
        kwargs["utils"] = script
    else:
        callbacks["initialize"] = f"def initialize(mutable context) {{ {script} }}"
    with pytest.raises(ValueError, match="仅允许 use query"):
        inline.data.backtest({}, callbacks, **kwargs)
    assert not inline.runtime.created


def test_user_query_and_bare_helpers_remain_available(inline):
    callback = "def initialize(mutable context) { getParams(); getParam(`size); getIndustry() }"
    result = inline.data.backtest({}, {"initialize": callback}, utils="use query")
    raw = result.session.raw_session
    assert callback in _scripts(raw)
    assert len(_inline_loads(raw)) == 1
    assert "def getParams()" in _inline_loads(raw)[0]
    assert "def getParam(key)" in _inline_loads(raw)[0]
    assert _scripts(raw).count("use query") == 2


def test_lazy_result_keeps_proxy_and_existing_close_semantics(inline):
    borrowed = FakeSession()
    with inline.data.backtest({}, {}, session=borrowed) as result:
        assert isinstance(result.session, inline.module._InlineBacktestSession)
        summary = result.return_summary
        assert "backtest::" not in summary
        assert "standardize_return_summary(" not in summary
        assert "if (size(values) > 0)" in summary
        assert summary.strip().endswith("result")
        assert "Backtest::getReturnSummary(" in summary
        assert any('loadPlugin("MatchingEngineSimulator")' in s for s in _scripts(borrowed))
        # 已有结果的代理传回公开入口不增加包裹或重复内联。
        again = inline.data.backtest({}, {}, session=result.session)
        assert again.session is result.session
        assert len(_inline_loads(borrowed)) == 1
    result.close()
    assert borrowed.close_count == 1
    assert _scripts(borrowed)[-1] == "Backtest::dropBacktestEngine(coreBacktestEngine)"


@pytest.mark.parametrize("owned", [True, False])
def test_error_closes_only_wrapper_owned_session(inline, owned):
    inline.runtime.fail = True
    raw = FakeSession()
    kwargs = {} if owned else {"session": raw}
    with pytest.raises(RuntimeError, match="runtime failed"):
        inline.data.backtest({}, {}, **kwargs)
    if owned:
        raw, max_time = inline.runtime.created[0]
        assert max_time == inline.runtime.BACKTEST_SESSION_MAX_TIME
    else:
        assert not inline.runtime.created
    assert raw.close_count == int(owned)


def test_helpers_are_literals_at_session_initialization_and_roots_are_not_defs(inline):
    tree = ast.parse((_SOURCE / "backtest.py").read_text(encoding="utf-8"))
    run = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    literals = [
        node.args[0].value for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run" and node.args
        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
    ]
    helper_script = next(script for script in literals if "def getParams()" in script)
    names = set(re.findall(r"(?m)^\s*def (\w+)\(", helper_script))
    assert names == inline.module._FUNCTIONS
    assert {"getHistoryData", "order_target", "getMinuteHistory", "snapshot_callback"} <= names
    all_text = "\n".join(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    assert not re.search(
        r"\bdef (?:run_backtest|run_snapshot_backtest|build_backtest_message|"
        r"standardize_return_summary|create_backtest_engine)\(", all_text,
    )
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "importlib.resources"
        for node in ast.walk(tree)
    )
    assert "arena_backtest.dos" not in all_text


@pytest.mark.parametrize("target,data,adj,spread", [
    ("customMessage", "coreBacktestData", "coreBacktestAdj", "coreBacktestSyntheticSpread"),
    ("coreBacktestBenchmarkMessage", "coreBacktestBenchmarkData", "NULL", "0.0"),
])
def test_message_root_is_expanded_in_place(inline, target, data, adj, spread):
    raw = FakeSession()
    proxy = inline.module._InlineBacktestSession(raw)
    proxy.run(
        f"before=1; {target} = backtest::build_backtest_message({data}, {adj}, {spread}); after=2",
        1, clearMemory=True,
    )
    script, args, kwargs = raw.calls[-1]
    assert script.startswith("before=1;") and script.endswith("; after=2")
    assert f"market_data = {data}" in script
    assert f"adj = {adj}" in script and f"synthetic_spread = {spread}" in script
    assert f"{target} = message" in script
    assert "backtest::" not in script and "def build_backtest_message(" not in script
    assert 'endsWith(codes, ".XSHE")' in script
    assert args == (1,) and kwargs == {"clearMemory": True}


def test_two_message_expansions_preserve_benchmark_filter_and_union(inline):
    raw = FakeSession()
    inline.module._InlineBacktestSession(raw).run("""
        custom = backtest::build_backtest_message(
            coreBacktestData, coreBacktestAdj, coreBacktestSyntheticSpread)
        custom = select * from custom where string(symbol) != coreBacktestBenchmark
        if (custom.rows() == 0) { throw "empty" }
        coreBacktestBenchmarkMessage = backtest::build_backtest_message(
            coreBacktestBenchmarkData, NULL, 0.0)
        custom = unionAll(custom, coreBacktestBenchmarkMessage)
        custom.sortBy!(`timestamp`symbol)
    """)
    script = _scripts(raw)[-1]
    assert script.index("market_data = coreBacktestData") < script.index("custom = select")
    assert script.index("custom = select") < script.index("market_data = coreBacktestBenchmarkData")
    assert script.index("market_data = coreBacktestBenchmarkData") < script.index("custom = unionAll")
    assert "backtest::" not in script


@pytest.mark.parametrize("market_source", ["daily", "snapshot"])
def test_run_roots_and_engine_creation_are_statements(inline, market_source):
    result = inline.data.backtest({}, {}, market_source=market_source, message_ref="customMessage")
    script = _scripts(result.session.raw_session)[-1]
    assert "coreBacktestEngine = engine" in script
    assert "Backtest::createBacktestEngine(" in script
    assert "Backtest::dropBacktestEngine(engine)" in script and "throw error" in script
    assert "def run_backtest(" not in script and "def run_snapshot_backtest(" not in script
    assert "create_backtest_engine(" not in script
    assert "backtest::" not in script
    assert 'config["orderBookMatchingRatio"] = 1.0' in script
    if market_source == "snapshot":
        assert 'coreBacktestSnapshotState["enabled"] = false' in script
        assert "wrapped = snapshot_callback{,,,on_snapshot_callback}" in script
        assert "for (day in days)" in script and "for (block in firstBlock..lastBlock)" in script
        assert "read_stock_snapshots(source,day,codes" in script
        assert "build_stock_snapshot_message(raw,dayReference)" in script
        assert 'state["enabled"] = false' in script
        create = script[script.index("engine = Backtest::createBacktestEngine("):]
        assert re.search(r"on_bar_callback,\s*wrapped,", create)
    else:
        assert "message = customMessage" in script
        assert "lastRow = imax(message.timestamp)" in script
        assert 'update endMessage set symbol="END", timestamp=timestamp+1' in script


def test_root_names_inside_literals_comments_and_symbols_are_not_expanded(inline):
    raw = FakeSession()
    protected = r'''// backtest::run_backtest(x)
/* backtest::build_backtest_message(x) */
a = "backtest::standardize_return_summary(x)"
b = `backtest::run_snapshot_backtest
Backtest::createBacktestEngine(name, config)
'''
    inline.module._InlineBacktestSession(raw).run(protected)
    assert _scripts(raw) == [protected]


@pytest.mark.parametrize("script", [
    "x = backtest::build_backtest_message(other, NULL, 0.0)",
    "coreBacktestEngine = backtest::run_backtest(x, y, z)",
    "coreBacktestEngine = backtest::run_snapshot_backtest(x, y, z)",
    "backtest::standardize_return_summary(x, y, z)",
])
def test_unrecognized_root_call_is_rejected_without_sending_dos(inline, script):
    raw = FakeSession()
    with pytest.raises(ValueError, match="不支持的 Runtime"):
        inline.module._InlineBacktestSession(raw).run(script)
    assert not raw.calls


def test_pinned_runtime_call_sites_are_supported_without_importing_runtime(inline):
    root = Path(sys.prefix) / "Lib/site-packages/runtime/apps/backtest"
    if not root.is_dir():
        pytest.skip("当前解释器没有锁定版本的 Runtime 源码")
    # AST 只提取 Runtime 实际 session.run/download 字面量，不执行其配置或数据库代码。
    namespace = {
        "DATA_REF": "coreBacktestData", "BENCHMARK_DATA_REF": "coreBacktestBenchmarkData",
        "BENCHMARK_MESSAGE_REF": "coreBacktestBenchmarkMessage", "message_ref": "customMessage",
        "run_message_ref": "coreBacktestRunMessage", "message_statement": "prefix = 1",
    }
    count = 0
    for filename in ("api.py", "result.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"run", "download"} and node.args):
                continue
            value = node.args[0]
            if not isinstance(value, (ast.Constant, ast.JoinedStr)):
                continue
            if "backtest::" not in ast.unparse(value):
                continue
            script = eval(compile(ast.Expression(value), filename, "eval"), {}, namespace)
            raw = FakeSession()
            inline.module._InlineBacktestSession(raw).run(script)
            assert "backtest::" not in _scripts(raw)[-1]
            assert not re.search(r"\bdef (?:run_backtest|run_snapshot_backtest|build_backtest_message|standardize_return_summary|create_backtest_engine)\(", _scripts(raw)[-1])
            count += 1
    assert count == 6
