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
- `FactorAnalysisParams(FactorParams)` 增加 `columns`、`max_lag` 等分析选项，`StrategyAnalysisParams(StrategyParams)` 增加行情、基准和回测引擎设置；都实现 `run` 并返回报告。各环节的 `Model/Optimize/Control/ExecutionAnalysisParams` 直接继承对应的 Algo Params，增加独立研究所需的日期、股票池、回测设置和上游选择。
- 五类公共 Form 由 Scheme 提供，项目只绑定自己的 AnalysisParams。`start/end/pool` 统一定义；四类策略环节还共享行情、基准、资金和费率。Scheme 负责 `pool → Universe`、资金费率 → `config` 以及 `build()`，不执行研究。

项目自定义 AnalysisParams 同时继承本项目的具体 Params 与对应 Scheme AnalysisParams；算法字段只在 Params 声明一次。公共 Form 会自动读取具体 AnalysisParams 的字段、默认值、约束和 JSON Schema 元数据，`build()` 返回该具体类型，并执行它的字段及跨字段校验。运行时根据 Factor/Algo 的参数泛型构造新的具体 Params，只传算法声明的字段，不把分析字段泄漏给算法。

### 项目表单的最小写法（1.3.0 起）

```python
from pydantic import Field
from scheme.base import ModelParams as BaseModelParams
from scheme.base import ModelAnalysisParams as BaseModelAnalysisParams
from scheme.base import ModelReportForm as BaseModelReportForm

class ModelParams(BaseModelParams):
    n_select: int = Field(default=10, ge=1, title="选股数量")

class ModelAnalysisParams(ModelParams, BaseModelAnalysisParams):
    pass

class ModelReportForm(BaseModelReportForm[ModelAnalysisParams]):
    pass
```

这里不需要再声明日期、股票池、手续费、选股数量，也不需要项目自己实现 `build()`。
Factor、Optimize、Control、Execution 使用对应的公共 Form，同样只绑定具体分析类型。
Factor 的项目输出列可以在项目 AnalysisParams 中声明一次，例如
`columns: list[str] = Field(default_factory=lambda: ["momentum"], min_length=1, title="因子列")`。
确实需要不同 UI 默认值时，仍可在 Form 子类覆盖对应字段。

- Form 保留既有 UI 默认：日期 `2020-01-01` 至 `2027-01-01`（不含结束日），股票池沪深 300；运行参数的日期仍必填。
- 基础分析字段未被项目覆盖时沿用 Scheme 的 UI 默认；项目显式重新声明的字段保留项目约束和默认。Factor 表单的市值加权默认不会被 Analysis 的等权默认意外覆盖。
- 算法的 Pydantic decorator validators 在 `build()` 构造具体 AnalysisParams 时执行，不要求表单提前具备 `universe/config`；`build()` 不会调用 `run()`。项目新增的业务字段保留原始输入和 Schema，将转换、约束及模型校验统一推迟到 `build()`，避免转换两次或绕过算法的 strict/字符串规范化配置。不能把单独构造 Form 当作完整研究参数已经校验通过。
- 项目在 AnalysisParams 中重新声明的平台字段（如 `start`、`benchmark`、`columns`）若只修改默认值、标题或约束，仍由表单校验并规范化，传给下游环节的公共值保持规范；带 Annotated 转换器、字段 validator 或嵌套模型时，与业务字段一样推迟到 `build()` 只执行一次。
- 自动表单支持无参 `default_factory`；依赖其他已校验字段的默认工厂无法独立生成 UI 默认值，会在绑定时明确拒绝。字段间推导请放到 AnalysisParams 的 `model_validator`，或继续使用自定义 `ReportForm`。
- `universe/config/symbols` 不直接暴露为 UI 输入，股票池通过 `pool` 选择。引擎配置可以直接构造 AnalysisParams；策略 AnalysisParams 的额外 `config` 默认项会保留，资金和费率由表单输入决定。表单不提供的内部字段（如 `symbols`）在项目中重新声明时必须有默认值，否则绑定表单时报错。
- `pool/cash/commission/tax` 等表单转换字段及表单方法名不能再作为另一个含义的算法字段；冲突明确报错，不静默覆盖。四类策略环节中，`model/optimize/control/execution` 是 Scheme 管理的上下游选择字段，项目参数不能声明同名字段。
- validation alias 的平铺输入键保持唯一。`AliasChoices` 中的 `AliasPath` 备用别名按分析参数自身语义读取嵌套值，不占用平铺键；只有纯 `AliasPath`、没有字符串别名的输入字段无法表示为扁平表单，会明确拒绝。
- 旧的 `ReportForm[具体AnalysisParams]` 加手写 `build()` 仍受支持，未参数化的 Scheme 公共 Form 仍可直接使用。

这种写法依赖 Scheme 1.3 起的公共泛型表单。Scheme 1.4 改为 Arena 股票池，Algo 模板 1.4.x 因此声明 `scheme>=1.4.0,<1.5.0`。1.2.x、1.3.x 项目继续使用各自锁定的 Scheme，不会被自动迁移。

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

源码泛型统一使用 `TypeVar` 和 `Generic`，类型别名使用 `TypeAlias`。算法参数和上下文通过泛型继承关系动态解析，支持中间继承层、参数重排、部分绑定和嵌套的 Pydantic 泛型；公共表单仍须绑定具体分析类型。现有项目的新式泛型继承仍可被解析，但新模板不再使用该语法，以兼容 Jupyter Jedi 的类型推断和定义跳转。

Algo 包与 Scheme 的主版本、次版本必须一致，补丁版本可独立迭代。包声明整个主次版本系列，例如 `scheme>=1.2.0,<1.3.0`，因此 Scheme 1.2.0 与 1.2.7 均可用于 1.2.x 项目，不允许与 1.1.x 或 1.3.x 混用。下界必须为该系列的补丁 0，上界必须为下一次版本的补丁 0；不接受补丁版本下界、精确版本、直接 Git/URL、条件依赖或跨次版本范围。模板选择、上游项目安装、策略组装和正式 wheel 校验使用同一兼容边界。项目和正式任务仍使用 uv source 与锁文件精确固定实际版本，兼容不等于自动升级或放宽锁文件、wheel 身份校验。其他共享依赖仍须可共同解析。

### 日志（1.3.2 起）

沿用 Arena 原有日志格式，不增加自定义列：时间、方括号级别、文件、函数、源码行号和消息。标准库日志接入后保留真实调用位置，不显示桥接 Handler 的位置。Notebook 中写入单元格输出，正式任务写入 stderr，进入调度日志；stdout 仍留给 CLI 的 JSON 输出。

```text
2026-10-11 01:00:00 [ INFO  ] backtest.py | run: 174行| 首个行情时点 2025-01-02 09:30:00，初始化 4 个 Algo
2026-10-11 01:00:00 [WARNING] default.py | allocate: 109行| [回测 2025-01-02 09:30] 风险平价退化为等权：历史样本不足（1 只）
```

回测模拟时间只作为消息正文的前缀，不改变日志头；现有日志组件可继续解析级别和源码位置。

- INFO（默认）：任务开始与结束、环境校验、加载的组件、回测参数、股票池规模、首个行情时点、耗时 1 秒以上的 Algo 初始化、按月进度（净值、持仓、现金、订单、成交、拒单、已用和预计剩余时间）、回测汇总（收益、基准、最大回撤、手续费和订单统计）、因子分析进度和报告保存。
- WARNING：风险平价退化为等权、拒单汇总等需要关注但不中止研究的情况；ERROR 为正式任务失败及其 traceback。
- DEBUG：Arena 每次查询的会话与建表步骤（显示时保留 Arena 原有的级别标记）、每批行情的行数与加载耗时、订单与成交回报分发、逐笔下单与拒单、默认风控和执行环节的订单数量。

通过环境变量 `SCHEME_LOG_LEVEL=DEBUG`，或在 Notebook 中调用 `scheme.log.configure("DEBUG")` 调整级别。环境变量无效时回退为 INFO 并输出警告，不影响导入；显式传入无效级别直接报错。

研究项目使用标准库日志即可接入同一格式和级别：

```python
import logging

logger = logging.getLogger(__name__)
logger.info("训练完成：%d 个样本", count)
```

独立研究进程没有配置标准库日志时，研究包（`factor`、`model_1a2b` 等平台包名及其子模块）、Notebook（`__main__`）、`backtest` 和 `scheme` 的标准库日志按所选级别输出；其他第三方库只输出 WARNING 及以上。宿主已配置标准库日志时，保留已有处理器和 root 级别，日志继续由宿主处理，不替换或关闭其日志文件。Scheme 自身日志的格式和级别设置仍然生效。

### 1.5.0 传统泛型与动态类型解析

- 泛型类和函数统一使用 `TypeVar` / `Generic`，类型别名使用 `TypeAlias`；保留原有参数顺序、约束及方差推断，Jupyter Jedi 可解析 `self.params.universe.query`。
- 动态解析同时支持传统泛型和已有项目的新式泛型：沿继承链传递具体参数，支持重排、部分绑定、嵌套 Pydantic 模型、`Annotated` 和可空联合类型；未具体化的混入基类不会遮蔽后面的有效绑定。
- 公共表单仍须绑定具体分析参数；手写 `ReportForm` 的泛型中间层也可正确解析。参数、任务和报告协议不变，命名类型别名的 Schema 引用改为内联类型，验证约束不变。
- 固定 Backtest 0.1.1 的发布提交，使底层 Algo 和回测类型同样使用传统泛型。

旧 Scheme 1.4.x 无法正确绑定新模板的传统泛型上下文，因此发布新次版本：新建项目选择 Scheme / Algo 模板 v1.5.0，模板声明 `scheme>=1.5.0,<1.6.0`。已有 1.4.x 及更早项目继续使用原锁定环境，历史任务和报告不改写，不与 1.5.x 项目互为上游或组装到同一策略。

### 1.4.1 因子收益截止日修复

- 因子收益行情只读取研究区间 `[start, end)`，不再为最长持有期读取结束日之后的行情。
- 区间最后 N 个交易日的 N 日收益为空，不补零；因子值、分组和换手率保留，缺少收益的日期不产生对应周期的 IC 与分组收益。

Python 接口、任务 JSON 和报告结构不变，Algo 模板 v1.4.0 继续兼容。新建项目选择 Scheme v1.4.1 后生效；已有项目继续使用各自锁定的版本，历史报告不改写。

### 1.4.0 股票池按 Arena 两阶段契约重写

- 股票池只保留 Arena 网页的五个选项：全市场、上证 50、沪深 300、中证 500、中证 1000。移除 `StockPool.CUSTOM`（“项目股票池”）以及 `Universe` 的 `lookback`、`codes`、`factors`、`derivatives`、`filters` 字段；`Universe` 只保存 `pool`，参数默认全市场，表单默认仍为沪深 300。
- 公共表单移除股票池的“回溯周期”（`lookback`）字段。Arena 的第一阶段查询回溯固定为 `P0D`；需要历史窗口的查询通过 `Universe.query(..., lookback=...)` 指定。
- 指数池改为两阶段查询：`codes_query` 取研究区间成员的候选并集，第二阶段只查询候选并集并注入成员节点；全市场没有第一阶段。成员条件来自 `MEMBERS` 映射（股票池 → DSL JSON），节点名由类属性 `Universe.member` 提供，值为 Arena 的 `stock_pool_member`。
- 第二阶段不按成员过滤，成员节点作为逐日 BOOL 列输出，由下游加入 `filters`、作为 `on` 或直接读取。新增 `Universe.codes_query()`、`Universe.dataset_query()` 和在同一会话执行两阶段的 `Universe.query()`。
- `Universe.evaluate()` 改由两阶段查询生成：列为候选并集（全市场为查询轴代码），值为当日成员节点；指数池逐日成分与 CoreData 中权重大于 0 的记录一致。回测未指定 `symbols` 时仍按面板列加载行情，已退出指数但仍持仓的股票照常有行情。

任务 JSON 的 `universe` 只含 `pool`，报告文件和 Runtime 协议不变。股票池字段和查询接口不兼容，因此升级次版本：新建项目选择 Scheme v1.4.0 和 Algo 模板 v1.4.0；已有 1.2.x、1.3.x 项目、版本、成果和报告继续使用各自锁定的版本，不自动迁移，也不能与 1.4.x 项目互为上游或组装到同一策略。

### 1.3.4 日志审查问题修复

- 启动日志和引擎使用同一个资金、佣金及税率数值转换，直接构造配置时的数值字符串仍可回测，不因日志格式化而提前报错。
- 导入 Scheme 不再强制替换标准库 root 处理器或关闭宿主日志文件；仅在 root 没有处理器时安装研究桥接，重复配置不追加处理器。宿主添加处理器后保留其级别与输出。
- 汇总最大回撤包含初始净值 1.0，首日亏损或费用不再遗漏。
- INFO 不可见时不执行月报专用查询；可见时服务端只返回上一月末的一条账户记录，不反复传输、转换完整历史。重新打开月报后，缺失的月初基线显示为不可用，不冒用累计收益。
- 月报持仓数量同时统计多头和空头证券，同一证券只计一次。

原有日志格式、前端解析器、表单、任务和报告协议均不变。项目锁定 Scheme v1.3.4 后生效；已有历史任务不改写。

### 1.3.3 恢复原有日志格式

撤销 1.3.2 引入的自定义日志头，直接使用 Arena 的 `log_format`。标准库日志保留原始文件、函数和行号，回测模拟时间仅放在正文中。日志点与级别保留，前端组件不改动。新增消费端契约测试，直接将生成的日志传给现有前端解析器，校验普通记录、拒单警告、模拟时间和负收益正文，以及 DolphinScheduler 包装记录。

1.3.2 的自定义格式会使现有组件将多数记录当成纯文本，还可能把负收益的负号误当成分隔符。该格式已废弃；已保存的历史任务日志不重写。新建项目选择 Scheme v1.3.3；已有 1.3.x 项目更新 Scheme 的固定提交并同步环境后生效。

### 1.3.2 研究与回测日志

- 新增研究日志点与级别设置。Arena 每次查询的会话与建表步骤改为只在 DEBUG 级别显示；日线回测按批次反复查询，这些步骤此前占满日志，研究进度和结果反而没有输出。自定义日志头存在消费端兼容问题，已由 1.3.3 撤销。
- 新增任务、环境校验、组件加载、回测参数、股票池、按月进度、回测汇总、拒单汇总、因子分析和报告保存等日志点，回测中的日志附带模拟时间。
- 正式任务失败时以 ERROR 输出错误和 traceback；`scheme parameters` 的单行 stderr 错误协议不变。
- backtest 改为输出标准库日志并固定到 `811346a`；新增直接依赖 `loguru`，与 Arena 使用的版本相同，锁定环境不新增包。

研究接口、表单、任务 JSON、报告文件和 Runtime 协议均不变。请使用已恢复原格式并修复日志审查问题的 Scheme v1.3.4；已有 1.3.x 项目可将 `pyproject.toml` 中 Scheme 来源的 `rev` 改为 v1.3.4 对应提交后重新 `uv lock` 和 `uv sync`。Algo 模板 1.3.0 不需要更新。

### 1.3.1 公共表单字段归属与别名检查修复

- 策略类项目的参数不能再声明 `model/optimize/control/execution`。此前同名参数会悄悄替换上下游选择字段，保存版本时把参数值当成上游入口。表单不提供的内部字段（如 `symbols`）被重新声明为必填时，绑定表单即明确报错，不再在每次 `build()` 时失败。
- `AliasChoices("days", AliasPath("config", "window"))` 这类备用嵌套别名对分析参数合法，不再被误判为与 `config` 字段冲突。
- 项目重新声明的平台字段只修改默认值、标题或约束时，由表单校验并规范化；传给下游环节的公共值不再保留原始字符串，也不再产生序列化警告。

新建项目选择 Scheme v1.3.1 后生效，Algo 模板 1.3.0 不需要更新；已有项目继续使用各自锁定的版本。

### 1.3.0 公共研究表单与 URL 报告展示

五类 ReportForm 改为可绑定项目 AnalysisParams 的公共泛型表单。`start/end/pool/lookback`，以及策略环节的行情、基准、资金、费率和上下游选择，统一由 Scheme 声明并转换；项目算法字段只在 Params 声明一次，表单自动带出其默认值、约束和 Schema。`build()` 返回项目的具体 AnalysisParams，并执行其字段与跨字段校验。保存版本、策略组装的任务 JSON、报告文件和 Runtime 协议不变；旧式 `ReportForm[具体AnalysisParams]` 加手写 `build()`、直接使用 Scheme 公共 Form 仍然可用。

新模板在导入时就依赖公共泛型表单，旧的 1.2.x Scheme 无法加载，因此升级次版本，而不是发布 1.2.x 补丁。按主次版本契约，1.3.x 项目不能与 1.2.x 项目互为上游或组装到同一策略；已有 1.2.x 项目、版本、成果和报告继续使用各自锁定的 Scheme，不自动迁移。

`report.show()` 改由 Solo Jupyter 启动桥接，以 URL iframe 复用前端报告页面，不再在 Notebook 输出中内嵌 `srcdoc`；独立导出仍使用 `report.to_html()`。

### 1.2.3 日线回测停牌与未上市证券修复

日线合成快照只为当日有日线记录的证券生成行情。股票池成员并集中尚未上市、已退市或停牌的证券当日没有开盘价、收盘价和昨收，跳过其当日快照，当日不能成交。存在日线但缺少昨收、涨跌停价或价格非正的记录仍明确报错，不推测价格。此前沪深 300 等动态股票池的多日回测会因成员并集包含这类证券而在首批行情失败。新建项目选择 Scheme `v1.2.3` 后生效，已有项目和任务继续使用各自锁定的版本。

### 1.2.2 因子市值加权校验修复

市值加权先剔除因子值为空的记录，再要求参与分组的有效因子记录具备正数 `circ_mv`。停牌等无因子值样本缺少市值时不再导致任务失败；有效样本缺少市值仍明确报错，不会改为等权或静默跳过。新建项目选择 Scheme `v1.2.2` 后生效，已有项目继续使用各自锁定的版本。

### 1.2.1 表单默认值修复

五类研究 ReportForm 默认日期统一为 `2020-01-01` 至 `2027-01-01`（不含结束日），股票池默认沪深 300。因子表单默认收益周期 `[1, 5, 20]`、5 组、极端股票数 10、市值加权、日历 `000300.XSHG`；具体因子列仍由项目声明。运行 Params/AnalysisParams 的日期仍须显式传入，Form 仅通过 `build()` 生成完整参数。本补丁同时包含嵌套模型 validation alias 与默认值序列化修复，保持 `1.2.x` 兼容范围和报告协议不变。

新建项目选择 Algo / Scheme `v1.2.1` 后使用上述默认值。已有项目及任务继续使用各自冻结源码和锁文件，不自动改写；历史 `v1.2.0` 标签保持原样。

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
任务输入由 Scheme 参数模型定义，进程协议见根工作区 runtime/README.md。

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
    pool=StockPool.CSI300, max_lag=2,
)
params = form.build()
report = params.run(Factor)  # 传入项目的 Factor 类，自动构造 FactorParams。
report.show()
report.save(output_path)

# 也可直接构造分析参数，无需经过表单。
params = FactorAnalysisParams(
    start="2025-01-01", end="2026-01-01", columns=["momentum"],
    universe=Universe(pool=StockPool.CSI500),
)
report = params.run(Factor)

report = run_backtest(algos, ctx, StrategyAnalysisParams(
    start="2025-01-01", end="2026-01-01", symbols=["000001.XSHE"],
))
report.show()
```

`report.show()` 需要通过 Solo 启动 Jupyter Server，由启动桥接在 IPython shell 上安装
`solo_report_display` provider。provider 通过 Jupyter 文件后端提供报告数据，以 URL iframe
复用 Solo 前端的 `FactorAnalysisReport` 和 `BacktestReport`；不再内嵌页面或回退到 `srcdoc`。
因子分析结果自动保留分析参数，不需要重复传入。可设置 `height=1000`、`theme="dark"`；
回测支持 `annual_trading_days=252` 和 `risk_free_rate=0.0`，参数校验和显示由 provider 负责。
未安装 IPython、没有 shell 或可调用 provider 时会明确报错，可改用 `report.to_html()` 独立导出。
`report.preview(name)` 仍用于直接查看 DataFrame。

`Path("report.html").write_text(report.to_html(), encoding="utf-8")` 的独立导出接口及现有实现保持不变，
不依赖上述 provider。页面、DuckDB WASM、图表组件及数据仍按原实现内嵌，不连接前端服务或 CDN；
`show()` 的 URL 展示不改变独立 HTML 的离线 Parquet 加载方式。

维护独立 HTML 报告资源时，在工作区 `frontend` 执行 `npm ci`、`npm run build:scheme`，
把同一套前端组件编译到 `scheme/report/assets`，然后构建 Scheme wheel。
安装 Scheme 不需要 Node.js 或前端源码；`show()` 展示依赖上述 Solo Jupyter 服务与桥接。

所有日期采用 [start, end)，回测入口转换为插件的包含结束日配置。日线合成每日开盘、收盘两份快照，
要求真实涨跌停价，不使用当日高低价推算开盘可见范围；合成盘口近似无限流动性，不用于衡量真实冲击成本。
当日没有日线的停牌、未上市或已退市证券不生成快照。
snapshot 模式接收 Arena StockSnapshot 的 Market、SecurityID、TradeDate、TradeTime、Bid/Ask 五档字段，
转换为插件消息，并关联 CoreData 的昨收与涨跌停价。默认回放 09:30–11:30、13:00–15:00 的有效成交价快照。
可通过 `DOLPHIN_SNAPSHOT_DATABASE`、`DOLPHIN_SNAPSHOT_TABLE` 指定同结构表。
其他原始格式需继承 ResearchBacktest 实现 load_messages。

因子检验通过 `max_lag` 输入最大滞后阶数（严格正整数，默认 1），生成 `2N` 个按时间顺序排列的隔夜与日内收益字段。所有字段与 t 日因子配对；最大滞后 N 涉及 t 至 t+N，共 N+1 个交易日的行情，不是累计持有收益。N=2 时：

| 字段 | 时段 | 简单收益公式 |
| --- | --- | --- |
| `r0o` | t 收盘 → t+1 开盘 | `O(t+1) / C(t) - 1` |
| `r1` | t+1 开盘 → t+1 收盘 | `C(t+1) / O(t+1) - 1` |
| `r1o` | t+1 收盘 → t+2 开盘 | `O(t+2) / C(t+1) - 1` |
| `r2` | t+2 开盘 → t+2 收盘 | `C(t+2) / O(t+2) - 1` |

O、C 分别为 `open * adj_factor`、`close * adj_factor`；收益通过 Arena query DSL 在 DolphinDB 中按完整查询交易日轴计算，不能在已筛选或稀疏的因子面板上位移，也不能跳过缺行情寻找下一条非空价格。`o` 表示隔夜段。
Factor 面板上传后，在同一 DolphinDB 会话内拼接收益率、过滤因子空值、生成分组并调用 Arena factor 模块计算报告；不重复处理 Factor 的预处理。默认等权，market_value 使用 circ_mv；只生成分组标签，不修改因子值。
行情仍只读取 `[start,end)`，最大滞后不会扩展查询结束日；不足未来 k 个交易日时，`r{k-1}o` 与 `r{k}` 为 NULL，不填零。收盘缺失可能只影响日内段，不能因为某个标签为空就删除其它可评价标签；这些行仍保留因子、分组及换手。
报告为每段收益记录 `kind=simple, periods=1`，另标明 lag 和 overnight/intraday；不能将字段中的阶数当成持有期。换手独立比较因子分组在 1..N 个交易日间隔下的变化，不等同实际成交换手。
旧 `return_periods` 不会被自动解释成新滞后参数，旧保存报告由原有收益规格解释，不改写历史文件。评价交易日轴由 calendar_symbol（默认沪深300）行情确定，缺少基准数据报错。
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
`FactorAnalysisParams`、`StrategyAnalysisParams` 分别直接继承它们，不通过额外的日期或 Settings 混入类间接拼接。
日期取外层 `[start, end)`。

### 股票池（1.4.0 起）

`universe` 只选择 Arena 网页的股票池：`StockPool.ALL`（全市场）、`SSE50`（上证 50）、`CSI300`（沪深 300）、
`CSI500`（中证 500）、`CSI1000`（中证 1000），默认全市场；表单默认沪深 300。股票池没有回溯周期。

- **成员节点。** 节点名是类属性 `Universe.member`，值为 Arena 的 `stock_pool_member`。各股票池的成员条件直接写在
  `MEMBERS` 映射中（股票池 → DSL JSON），须逐日返回 BOOL。当前四个指数池为 `binary.gt(指数权重, 0)`，权重字段分别为
  `weight_000016SH`、`weight_000300SH`、`weight_000905SH`、`weight_000852SH`；全市场为恒真的 `nullary.true`。
  需要复合条件时，把对应股票池的 JSON 改为组合表达式即可。
- **指数动态池分两阶段。** 第一阶段 `codes_query` 用成员节点过滤，回溯为 `P0D`，取整个研究区间结果代码的并集；
  它只确定第二阶段要加载的候选代码，不代表这些代码每天都是成分。第二阶段只查询候选并集，并注入同一个成员节点逐日判定成分。
- **全市场没有第一阶段。** 第二阶段加载查询轴上的全部代码，成员节点恒为 true，不缩小股票池。
- **第二阶段不按成员过滤。** 结果包含候选代码在每个交易日的行，成员节点作为逐日 BOOL 列 `Universe.member` 输出，
  如何使用由下游决定：只保留当日成分时把 `Universe.member` 加入 `filters`；只限制截面计算范围时作为 CS 算符的 `on`，
  例如 `"on": Universe.member`；也可以读取该列自行处理。DSL 直接引用 `Universe.member`，不需要手写节点名，但不能重新定义它。
- **成分数据。** 指数权重来自 Tushare `index_weight`，每个自然日写入不晚于该日的最新快照。查询时，某日观察到该指数权重后，
  其他代码缺失的权重先填 0，再按代码前向填充。权重 NULL 的日期按非成分处理。

```python
from datetime import timedelta

from scheme import StockPool, StrategyParams, Universe

universe = Universe(pool=StockPool.CSI300)
# 只在当日成分内做截面排名；非成分行的结果为 NULL。
derivatives = {
    "close_rank": {
        "type": "CS", "op": "unary.rank_pct", "fields": {"col": "close"},
        "on": Universe.member,
    },
}
universe.codes_query(start, end)        # 第一阶段请求；全市场为 None
universe.dataset_query(start, end, derivatives=derivatives)  # 第二阶段请求：已注入成员列，不过滤
with universe.query(
    start, end, lookback=timedelta(days=60), derivatives=derivatives,
    filters=[Universe.member],          # 下游自行决定：只保留当日成分
) as result:
    frame = result.data                 # 两阶段在同一会话执行，在 with 内读取

params = StrategyParams(start="2026-06-01", end="2026-07-01", universe=universe)
panel = params.universe.evaluate(params.start, params.end)
```

面板索引名为 `date`，为查询区间的交易日；列为第一阶段的候选并集（全市场为查询轴代码，`.XSHG/.XSHE`），
值为当日的成员节点 `Universe.member`。退池后为 `False`，某天没有成员时保留全 `False` 行；指数池在整个区间没有成员时，
第一阶段按 Arena 规则报错。
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
