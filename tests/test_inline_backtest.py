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
            "Backtest::getReturnSummary(coreBacktestEngine), portfolios, 250, 0.04)"
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
        session.run("backtest::run_backtest(name, config, message)")
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
    return [script for script in _scripts(session) if "def build_backtest_message(" in script]


def test_public_api_signature_and_binding(inline):
    assert inline.api.query is inline.query
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
    "build_backtest_message", "run_backtest", "run_snapshot_backtest",
    "standardize_return_summary", "backtest_order_quote", "create_backtest_engine",
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
        assert "standardize_return_summary(" in summary
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


def test_resource_contains_exact_pinned_runtime_dependency_closure(inline):
    # 仅 AST 读取已安装源码：绝不 import Runtime 编译器或配置。
    root = Path(sys.prefix) / "Lib/site-packages/runtime/database/compile/backtest/functions"
    if not root.is_dir():
        pytest.skip("当前解释器没有锁定版本的 Runtime 源码")
    definitions = {}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Name) or node.value.func.id != "DolphinDBFunction":
                continue
            fields = {field.arg: field.value for field in node.value.keywords}
            assert ast.literal_eval(fields["module"]) == "backtest"
            dependencies = fields.get("dependencies", ast.Tuple(elts=[]))
            definitions[node.targets[0].id] = (
                inspect.cleandoc(ast.literal_eval(fields["definition"])),
                [dependency.id for dependency in dependencies.elts],
            )
    tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    roots = next(
        node.value.elts for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "BACKTEST_FUNCTIONS"
    )
    assert len(roots) == 13
    ordered = []
    seen = set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        definition, dependencies = definitions[name]
        for dependency in dependencies:
            visit(dependency)
        ordered.append(definition)

    for root in roots:
        visit(root.id)
    script = inline.module._BACKTEST_SCRIPT.read_text(encoding="utf-8")
    assert len(ordered) == 19
    assert script == "\n\n".join(ordered) + "\n"
    names = set(re.findall(r"(?m)^def (\w+)\(", script))
    assert names == inline.module._FUNCTIONS
    assert not re.search(r"(?m)^\s*(?:module|use)\b|\b(?:factor|common|ta|backtest)::", script)
    assert "Backtest::dropBacktestEngine(engine)" in script
    assert "throw error" in script
    assert "objByName(\"coreBacktestSnapshotState\")" in script
