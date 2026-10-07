# Solo Scheme

scheme 提供 Factor/Algo/Strategy 接口、参数模型、策略组装、数据源、因子分析、回测驱动及报告生成。
所有研究执行代码与接口版本一起发布；不依赖 solo-runtime。

包结构按职责划分：

```text
scheme/
├── base/
│   ├── internal/   # 共用 Algo、Context、Params、ReportForm 和类型
│   ├── projects/
│   │   ├── factor/     # algo.py、params.py、form.py
│   │   ├── model/      # algo.py、params.py、form.py
│   │   ├── optimize/   # algo.py、params.py、form.py、default.py
│   │   ├── control/    # algo.py、params.py、form.py、default.py
│   │   └── execution/  # algo.py、params.py、form.py、default.py
│   └── __init__.py # 显式导入公共接口，通过 __all__ 控制导出
├── data/
│   ├── dolphindb/  # 查询、股票池、代码转换；database/ 管理连接与会话
│   ├── tushare/    # Tushare SDK 和 Pro 客户端
│   └── __init__.py # 统一导出 query、backtest、ts_api、pro
├── report/      # 共用 Notebook/HTML 展示和前端资源
├── execute/
│   ├── factor/    # 因子分析、结果和正式任务入口
│   ├── model/     # 策略建模任务入口
│   ├── optimize/  # 组合优化任务入口
│   ├── control/   # 订单风控任务入口
│   ├── execution/ # 算法下单任务入口
│   ├── strategy/  # 策略组装、回测驱动、结果和正式任务入口
│   └── ...        # 共用任务协议、包校验、结果保存
├── manage/         # CLI 入口、项目表单模型发现
└── config.py       # 服务连接配置
```

`base` 只定义研究源码和 Notebook 使用的算法、参数、表单、Context 和类型。
回测驱动在 `execute/strategy/engine.py`；泛型解析和 Context 适配校验在 `execute/strategy/assembly.py`；
报告实现位于 `report/`，不从 `base` 导出这些执行工具。
`execute/factor` 和 `execute/strategy` 实现各自研究流程；组合、风控、执行的默认算法分别位于
`base/projects/optimize/default.py`、`base/projects/control/default.py`、`base/projects/execution/default.py`。
`execute` 校验运行环境后调用研究接口，`manage` 提供命令入口。
项目可从 `scheme.base` 或 `scheme` 导入 Factor/Algo 等公共基类；
从 `scheme.base` 导入 `FactorReportForm` 和 `FactorAnalysisParams`。

每类项目统一从 `scheme.base` 导入对应的 Algo、Params、AnalysisParams 和 `ReportForm`；
公共入口由 `base/__init__.py` 的显式导入和 `__all__` 定义，不要求调用方了解内部目录。

参数按三层职责组织：

- `FactorParams`、`StrategyParams` 是构造因子和组装策略需要的运行参数，包含 `start`、`end`、`universe`。四类 Algo 的 Params 只声明自身算法字段，从统一策略参数提取，不携带回测配置。
- `FactorAnalysisParams(FactorParams)` 增加 `columns`、`return_periods` 等分析选项，`StrategyAnalysisParams(StrategyParams)` 增加行情、基准和回测引擎设置；都实现 `run` 并返回报告。各环节的 `Model/Optimize/Control/ExecutionAnalysisParams` 直接继承对应的 Algo Params，增加独立研究所需的日期、股票池、回测设置和上游选择。
- 所有 Form 只继承 `ReportForm[具体AnalysisParams]`，显式声明 UI 输入并实现 `build()`，把 `pool/lookback` 转换为 `Universe`、把资金费率转换为引擎配置。Form 不继承 Params 或 AnalysisParams，不执行研究；跨字段校验发生在 `build()` 构造分析参数时。

项目自定义 AnalysisParams 同时继承本项目的具体 Params 与对应 Scheme AnalysisParams，保证算法字段是真正的 Pydantic 字段，而不是仅存在于额外字段中。项目 Form 的泛型和 `build()` 返回该具体 AnalysisParams。运行时根据 Factor/Algo 的参数泛型构造新的具体 Params，只传算法声明的字段，不把分析字段泄漏给算法。

各项目导出对应的算法、Params、AnalysisParams 和 ReportForm。Factor 的 `params.run(Factor)` 接受类并自动构造其泛型声明的参数；四类 Algo 的 `params.run(当前Algo)` 根据上游选择组装完整策略。内置后续选项为 risk_parity（风险平价）、no_control（不风控）、direct_execution（不拆单）。Context 从 ModelAlgo 泛型解析，也可通过 `run(..., ctx=实例)` 传入。

`run` 的结果泛型保留具体分析参数类型，报告的 `parameters` 保存该参数副本。`scheme parameters` 校验 `ReportForm` 声明的具体分析类型和 `build()` 结果，生成 JSON Schema；不会在定义或校验表单时调用 `run`。

```python
from model import ModelAlgo, ModelReportForm
from scheme import StockPool

form = ModelReportForm(
    start="2026-06-01", end="2026-06-06",
    pool=StockPool.SSE50, optimize="risk_parity",
    control="no_control", execution="direct_execution",
)
params = form.build()
report = params.run(ModelAlgo)
report.show()
```

Model 保存版本与因子共用源码快照和候选 wheel 流程；输入 kind 为 model，
algos.model 指向冻结的候选包，backtest 保存表单构建后的参数和后续 Algo 选择。
Worker 使用锁文件中的 Scheme 默认实现；显式指定的下游包优先于默认实现。

Algo 包与 Scheme 的主版本、次版本必须一致，补丁版本可独立迭代。包声明整个主次版本系列，例如 `scheme>=1.2.0,<1.3.0`，因此 Scheme 1.2.0 与 1.2.7 均可用于 1.2.x 项目，不允许与 1.1.x 或 1.3.x 混用。下界必须为该系列的补丁 0，上界必须为下一次版本的补丁 0；不接受补丁版本下界、精确版本、直接 Git/URL、条件依赖或跨次版本范围。模板选择、上游项目安装、策略组装和正式 wheel 校验使用同一兼容边界。项目和正式任务仍使用 uv source 与锁文件精确固定实际版本，兼容不等于自动升级或放宽锁文件、wheel 身份校验。其他共享依赖仍须可共同解析。

### 1.2.0 发布契约与旧版本退役

1.2.0 是完整主次版本兼容契约的首个活跃发布。历史 Scheme 0.1.0、1.0.0、1.0.1、1.1.0 已退役：旧依赖声明不覆盖新的窄系列契约，历史 1.0 补丁也存在 CLI/报告接口变化；1.1.0 的正式执行仍按旧范围要求检查，不能仅修改调用方版本声明解决。Solo Backend 的 `version-policy.json` 统一决定新使用准入，不删除或移动旧 Tag。

1.2.0 保留在 1.1.0 引入的 Param → AnalysisParams 职责和单一 ReportForm 泛型。旧的 `BacktestParameters` 使用 `StrategyAnalysisParams` 替代，日期/Settings/Components 字段混入类已移除；新模板最低要求 Scheme 1.2.0。已有研究项目保留原锁定 commit、源码和锁文件，历史报告仍可查看，退役来源不能新保存、安装为上游或组装策略。升级须迁移源码并验证，不能通过自动修改旧环境替代。

1.2.0 保留任务 JSON 字段、报告文件结构和 Runtime 协议。参数投影保留 Pydantic 验证别名、运行时嵌套类型和非序列化字段；完整策略执行按已构造 Algo 的实际参数校验分析参数，包含默认值、类型转换及与分析配置同名的算法字段。报告保存具体 AnalysisParams 的深拷贝。表单 schema、默认值、已保存值与提交均使用同一公开 validation alias；业务参数分类仍使用规范字段名，无法表示为平面 UI 字段的 AliasPath 明确拒绝。

组装时统一使用一个 Scheme 安装实例，按各 Algo 的参数模型校验统一参数，并校验 Context 类型。Context 赋值校验 Signal/Target/Order 类型；Model 的 Signal 业务结构仍须与 Optimize 一致，不能仅凭包版本推断信号含义。补丁升级必须保持该主次版本系列内的 Python 接口兼容；每次发布应验证不同补丁版本 Algo 的双向混合安装、组装及原有报告契约。需要不兼容的 Python 接口调整时升级次版本；报告结构的兼容边界独立由主版本控制。

运行完成后的 `run.json.versions.scheme` 是报告解析版本的依据，不能用项目创建时的 Scheme 版本代替。前端报告仍仅按该版本的主版本分派，1.0.x、1.1.x 等 1.x 报告共用 v1 适配器，不受项目互调的主次版本限制影响。Runtime 的 `protocol` 仅表示进程完成协议，与 Scheme 的业务主版本分别演进。

```shell
uv sync
uv run pytest
scheme run --input /shared/tasks/123/input.json --output /shared/tasks/123/report
```

正式调用发生在任务锁定的 uv 环境中。启动器 Runtime 只安装任务依赖并调用上面的固定 CLI。
输入中的 environment 只含 lockfile，研究包 wheel 的身份、哈希、scheme 兼容范围及实际环境由 scheme 校验。
kind 分别为 factor、model、optimize、control、execution、strategy，对应各自执行入口；strategy 用于完整策略组装。当前除 factor 外，各入口暂时复用回测报告实现。
报告类型由结果对象决定，单独写入 run.json 的 report_kind，任务 kind 始终保留原值。输出 Parquet 后，最后原子写入协议版本 1 的 run.json。
完成清单包含原始输入、锁文件和各报告文件的 SHA256，以及实际安装版本。
完整输入示例位于根工作区 runtime/examples，进程协议见 runtime/README.md。

Jupyter SDK 使用 `scheme.base`、`scheme.execute`、`scheme.data`。
连接变量沿用 Arena：`DOLPHIN_HOST`、`DOLPHIN_PORT`、`DOLPHIN_RUNTIME_USERNAME`、
`DOLPHIN_RUNTIME_PASSWORD`；长表默认 `dfs://CoreData/coreData`，字段为 time/code/factor/value。
通过 `DOLPHIN_CORE_DATABASE`、`DOLPHIN_CORE_TABLE` 覆盖。只读取基础数据，不采集或更新。
复制 `.env.example` 填写配置后，可使用 `uv run --env-file .env solo-manage ...` 注入环境变量。

Python 入口：

数据查询统一从 `scheme.data` 按需导入：

```python
from scheme.config import DolphinSettings
from scheme.data.dolphindb.database import create_session
from scheme.data import query, backtest
from scheme.data import ts_api, pro
```

`scheme.data` 导出 DolphinDB 的 Arena `query`、`backtest`，以及 Tushare SDK `ts_api` 和 Pro 客户端 `pro`，
导入前配置 `TUSHARE_TOKEN`；不会在导入时发起数据请求。未使用 Tushare 时无需配置其 token。

`query`、`backtest` 分别直接导出 Arena 的 `execute_query`、`run_backtest`，参数和结果保持 Arena 原样。
它们返回的结果支持 `with`：退出时关闭查询连接，回测还会销毁其引擎；传入的 session 也由结果关闭。
需要保留的 DataFrame 在 `with` 内读取。Arena DSL 类型从 `runtime.apps.query` 导入。

```python
from scheme.data import query, backtest

request = {
    "start_date": "2026-06-01", "end_date": "2026-06-02",
    "codes": ["000001.SZ", "600000.SH"], "factors": ["open", "close"],
}
with query(request) as result:
    frame = result.data

# callbacks 是 Arena 格式的完整 DOS 回调字典。
with backtest(request, callbacks, config={"cash": 100000}) as result:
    portfolios = result.daily_portfolios
    trades = result.trade_details
```

这两个 Arena 入口沿用闭区间 `[start_date, end_date]`，结果使用 `time/code` 列与 `.SH/.SZ` 证券代码，
并使用已部署的 Arena DolphinDB `query/common/backtest` 模块。Solo 自身的 Factor/Algo 研究入口仍采用下文的 `[start, end)`。
Arena 的 `backtest` 会通过 Tushare 读取股票元数据，因此还需配置 `TUSHARE_TOKEN`。

```python
from scheme import StockPool, Universe
from scheme.base import FactorAnalysisParams, FactorReportForm
from scheme.execute.strategy import StrategyAnalysisParams, run_backtest

form = FactorReportForm(
    start="2025-01-01", end="2026-01-01", columns=["momentum"],
    pool=StockPool.CSI300,
)
params = form.build()
report = params.run(Factor)  # 传入项目的 Factor 类，自动构造 FactorParams。
report.show()
report.save(output_path)

# 自定义股票池等分析参数时，可直接构造，无需经过表单。
params = FactorAnalysisParams(
    start="2025-01-01", end="2026-01-01", columns=["momentum"],
    universe=Universe(
        codes=["000001.SZ"],
        derivatives={"member": {"type": "DIRECT", "op": "nullary.true", "fields": {}}},
        filters=["member"],
    ),
)
report = params.run(Factor)

report = run_backtest(algos, ctx, StrategyAnalysisParams(
    start="2025-01-01", end="2026-01-01", symbols=["000001.XSHE"],
))
report.show()
```

`report.show()` 在 Notebook 中展示内置的完整交互页面，直接复用 Solo 前端的
`FactorAnalysisReport` 和 `BacktestReport`，包括图表、统计指标、日期筛选、收益周期切换、
回测明细表的筛选/排序/分页/导出和日夜模式。因子分析结果自动保留分析参数，不需要重复传入。
可设置 `height=1000`、`theme="dark"`；回测支持 `annual_trading_days=252` 和 `risk_free_rate=0.0`。
`report.preview(name)` 仍用于直接查看 DataFrame。

`Path("report.html").write_text(report.to_html(), encoding="utf-8")` 可导出独立交互报告。
页面、DuckDB WASM、图表组件及数据全部内嵌，不连接前端服务或 CDN；保存 Notebook 输出会包含这些资源。
Notebook 需使用可信输出，Kernel 需安装 IPython（项目的 ipykernel 已包含）。

维护报告页面时，在工作区 `frontend` 执行 `npm ci`、`npm run build:scheme`，
把同一套前端组件编译到 `scheme/report/assets`，然后构建 Scheme wheel。
运行或安装 Scheme 不需要 Node.js、前端源码或单独服务。

所有日期采用 [start, end)，回测入口转换为插件的包含结束日配置。日线合成每日开盘、收盘两份快照，
要求真实涨跌停价，不使用当日高低价推算开盘可见范围；合成盘口近似无限流动性，不用于衡量真实冲击成本。
snapshot 模式接收 Arena StockSnapshot 的 Market、SecurityID、TradeDate、TradeTime、Bid/Ask 五档字段，
转换为插件消息，并关联 CoreData 的昨收与涨跌停价。默认回放 09:30–11:30、13:00–15:00 的有效成交价快照。
可通过 `DOLPHIN_SNAPSHOT_DATABASE`、`DOLPHIN_SNAPSHOT_TABLE` 指定同结构表。
其他原始格式需继承 ResearchBacktest 实现 load_messages。

因子收益通过 Arena query DSL 在 DolphinDB 中计算：复权收盘价 shift(-N) / 当日复权收盘价 - 1，按交易日轴对齐，不重复处理 Factor 的预处理。
Factor 面板上传后，在同一 DolphinDB 会话内拼接收益率、过滤空值、生成分组并调用 Arena factor 模块计算报告；Python 只下载最终报告表。
默认等权分组，market_value 使用 circ_mv；只生成分组标签，不修改因子值。尾部不足的未来收益保留空值。
评价交易日轴由 calendar_symbol（默认沪深300）行情确定，缺少基准数据报错。
未来收益只进入评价，不能传给 Model。默认风险平价缺少历史样本或求解失败时退化等权并记录原因。

策略回测驱动初始化正序，每次事件先逆序触发回调，再正序调用各环节 process，重复直到没有可消费的消息。Context 变化不伪造事件回调。
JSON 中按 Model、Optimize、Control、Execution 组装。null 后续环节使用默认 Algo。
组件字段由 scheme.AlgoComponents 定义，顺序和默认算法由 scheme.Strategy 维护。scheme 的 CLI 仅加载已指定的包并传入 Strategy，不维护默认算法实现。默认风险平价直接使用 scheme 声明的 arena-runtime 查询依赖，scheme 不依赖 solo-runtime。

```python
from scheme import Strategy
from scheme.base import StrategyParams, StrategyAnalysisParams

parameters = StrategyParams(start="2026-06-01", end="2026-07-01", window=20)
strategy = Strategy(ctx, params=parameters, model=MyModel, optimize=MyOptimize)
analysis = StrategyAnalysisParams(**parameters.model_dump(), config={"cash": 500_000})
report = analysis.run(strategy)
```

`Strategy[P, C]` 保留具体运行参数和 Context 类型。Strategy 接收 Algo 类，构造时将同一份 StrategyParams 转成每个 Algo 泛型声明的参数模型，再实例化 Algo。
StrategyParams 允许额外字段，例如 `window=20`、`gross_exposure=0.8`；各 Algo 只读取自身参数模型声明的属性，
保留默认值、类型转换和校验，同名参数共享同一个值。缺失必填参数或校验失败会在组装时报错。
JSON 中所有 Algo 参数统一写在 `backtest`，`algos` 只描述包与入口，不再接受各自的 `params`。

`scheme.FactorParams` 和 `scheme.StrategyParams` 各自定义所属运行领域的 `start`、`end`、`universe`；
`FactorAnalysisParams`、`StrategyAnalysisParams` 分别直接继承它们，不通过额外的日期或 Settings 混入类间接拼接。`universe` 是必须包含非空 `filters` 的 Arena DSL，
日期取外层 `[start, end)`；默认使用恒真过滤条件表示全市场。沪深300示例：

```python
from scheme import FactorParams

params = FactorParams(
    start="2026-06-01", end="2026-07-01",
    universe={
        "lookback": "P40D",
        "derivatives": {
            "member": {
                "type": "DIRECT", "op": "binary.gt",
                "fields": {"left": "weight_000300SH", "right": 0.0},
            },
        },
        "filters": ["member"],
    },
)
panel = params.universe.evaluate(params.start, params.end)
```

面板索引名为 `date`，覆盖查询区间的全部交易日；列为期间至少一次入池的股票代码（`.XSHG/.XSHE`），
值为 `bool`。退池后为 `False`，某天没有成员时保留全 `False` 行；整个区间没有成员时保留日期、返回零列。
`lookback` 用于 DSL 回看和历史权重前向填充，不进入输出日期范围；应覆盖起始日前最近一次成分权重记录。
Factor 和 Strategy 的 `universe` 属性首次访问时查询并缓存这张面板。
Algo 可通过 `self.backtest.universe` 读取；回测未指定 `symbols` 时按面板列名加载行情，
每日是否选股由 Algo 根据对应日期的布尔值判断。显式 `symbols` 仅控制行情加载范围。
CLI 因子任务以 `analysis` 的这三个公共参数为准，保留 `factor.params` 的自定义算法字段，再按具体 Factor 的泛型参数模型校验。表单序列化时把具体算法字段存入 `factor`，标准分析选项存入 `analysis`；不能被标准分析器执行的自定义分析字段会明确报错，不静默丢弃。

默认 Optimize 只接受有符号数值 Signal；自定义结构必须配套 Optimize。
Solo 研究链路统一使用 `.XSHG/.XSHE`：股票池、Signal、Target、行情、持仓与回报使用相同代码。
输入兼容 `.SH/.SZ`，在股票池查询、Context 消息、行情参数及下单/撤单入口转换；
直接调用 Arena `query` 时仍保留其原始 `.SH/.SZ` 格式。Signal/Target 同时包含同一证券的两种后缀会报错，不覆盖其中一个值。
Algo 可以直接下单，也可以在任意回调中写 signal。Model → Optimize → Control → Execution 依次消费 signal、target、orders；None 表示无消息，空集合仍会被消费。基类在调用特有函数前取走并清空输入，再将消息传入 on_signal(signals)、on_target(target)、on_orders(orders)。特有函数返回 None，只负责写下游消息或下单；函数内新写入的上游消息留待下一轮消费。Execution 下单产生的回报在当前处理结束后进入下一轮逆序通知，不需要等待下一条行情。
target 为 dict[Asset, float]，表示完整目标资产权重，以账户权益为分母；正数做多，负数做空，未列出的原持仓目标为零。None 表示没有新目标，空字典表示清仓。默认 Optimize 只生成风险平价权重；NoControl 根据权重、权益、当前价格和 lot_size 转换为市价订单，全部通过；Execution 直接提交订单。
若轮到提交时已经收盘或午休，订单保留到下一个可交易回调；真正报单时将 time 设置为当前回测时间。
这是待提交订单的发送时间，不是提前指定未来执行时间，且不会改写原订单对象。

研究包版本必须与任务安装的 scheme 同主版本、次版本并声明整个补丁系列的兼容范围；wheel 哈希、实际安装内容、
参数 BaseModel、入口所属包均检查。报告写共享目录，run.json 最后写入，含展开后的参数、实际版本与报告文件名。
不同正式运行应使用不同输出目录。Python 直接调用用于当前 Kernel 调试；正式包身份及锁文件检查由 CLI 执行。

验证：`uv run pytest` 默认执行无需服务的测试；设置 `SOLO_TEST_DOLPHIN=1` 并提供以上连接变量后，
会额外执行真实 DolphinDB 的因子统计、日线合成快照与 tick 字段转换/成交记账测试。
这些确定性测试只在各自会话中构造行情，不写入基础数据表。


### 组合、风控与执行研究

`OptimizeReportForm`、`ControlReportForm`、`ExecutionReportForm` 各自生成对应的 AnalysisParams，
`params.run(当前项目 Algo)` 只替换当前环节。上游字段选择插件已安装的项目入口；后续环节使用表单中的默认算法。
`scheme.algo_options("model")` 等函数返回同 Scheme 主版本、次版本的已安装包选项，补丁版本可不同；插件也使用这些选项生成下拉框。
Control 选择 Model、Optimize；Execution 选择 Model、Optimize、Control；不会把默认算法当作已选的上游项目。

```python
from optimize import OptimizeAlgo, OptimizeReportForm
from scheme import algo_options

models = algo_options("model")  # {"model_<项目ID>:ModelAlgo": "包名 · 版本"}
form = OptimizeReportForm(
    start="2026-06-01", end="2026-06-06",
    model=next(iter(models)),  # 可替换为 models 中指定的入口
    control="no_control", execution="direct_execution",
)
report = form.build().run(OptimizeAlgo)
report.show()
```

三个项目目前均输出与 Model 相同的完整回测报告。正式保存会将当前包、所有选择的上游包及传递依赖冻结为 wheel，
记录包版本、入口和 SHA256；Worker 使用冻结环境运行，读取当前项目对应的候选包，而不是把它误当成 Model。
