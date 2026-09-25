"""将原前端组件、DuckDB 和 Parquet 组合为无需服务或 CDN 的报告。"""

import base64
import html
import json
import math
from functools import lru_cache
from importlib.metadata import version
from importlib.resources import files
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scheme.base import FactorAnalysisParams
    from scheme.report.base import Report


@lru_cache(maxsize=1)
def _assets() -> dict[str, str]:
    root = files("scheme.report").joinpath("assets")
    names = {
        "script": "report.js",
        "style": "report.css",
        "worker": "duckdb-browser-mvp.worker.js",
        "wasm": "duckdb-mvp.wasm",
    }
    try:
        return {
            key: base64.b64encode(root.joinpath(name + ".gz").read_bytes()).decode()
            for key, name in names.items()
        }
    except FileNotFoundError as error:
        raise RuntimeError(
            "Scheme 缺少内置报告资源；请先在 frontend 执行 npm run build:scheme 再构建 wheel"
        ) from error


def report_html(
    report: "Report",
    *,
    theme: Literal["light", "dark"] = "light",
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> str:
    """导出完整交互报告；所有图表计算直接复用前端组件。"""
    if theme not in {"light", "dark"}:
        raise ValueError("theme 必须为 light 或 dark")
    if (
        isinstance(annual_trading_days, bool)
        or not isinstance(annual_trading_days, int)
        or annual_trading_days <= 0
    ):
        raise ValueError("annual_trading_days 必须为正数")
    if not math.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate 必须为有限数值")
    data = {
        "schemeVersion": version("scheme"),
        "kind": report.report_kind,
        "theme": theme,
        "annualTradingDays": annual_trading_days,
        "riskFreeRate": risk_free_rate,
        "files": {
            name: "data:application/octet-stream;base64,"
            + base64.b64encode(getattr(report, name).to_parquet(index=False)).decode()
            for name in report.filenames
        },
    }
    if report.report_kind == "factor":
        parameters: FactorAnalysisParams | None = getattr(report, "parameters", None)
        if parameters is None:
            raise ValueError(
                "因子报告缺少分析参数；请使用 analyze_factors 返回的结果，或设置 report.parameters"
            )
        data["parameters"] = {
            "factor_columns": parameters.columns,
            "return_columns": [f"return_{period}" for period in parameters.return_periods],
            "return_specs": {
                f"return_{period}": {"kind": "simple", "periods": period}
                for period in parameters.return_periods
            },
            "n_groups": parameters.groups,
            "n_select": parameters.n_select,
        }
    elif report.report_kind != "backtest":
        raise ValueError(f"不支持的报告类型：{report.report_kind}")
    payload = json.dumps({"report": data, "assets": _assets()}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    return (
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Scheme 研究报告</title></head>
<body><div id="root">正在加载报告…</div>
<script id="scheme-payload" type="application/json">"""
        + payload
        + """</script>
<script>
(async () => {
  const payload = JSON.parse(document.getElementById('scheme-payload').textContent);
  async function unpack(encoded, type) {
    const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return URL.createObjectURL(new Blob([await new Response(stream).arrayBuffer()], {type}));
  }
  const [script, style, worker, wasm] = await Promise.all([
    unpack(payload.assets.script, 'text/javascript'), unpack(payload.assets.style, 'text/css'),
    unpack(payload.assets.worker, 'text/javascript'), unpack(payload.assets.wasm, 'application/wasm')
  ]);
  window.__SCHEME_REPORT__ = payload.report;
  window.__SCHEME_DUCKDB__ = {mvp: {mainModule: wasm, mainWorker: worker}};
  const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = style; document.head.append(css);
  const js = document.createElement('script'); js.src = script; document.body.append(js);
  window.addEventListener('pagehide', () => [script, style, worker, wasm].forEach(URL.revokeObjectURL), {once: true});
})().catch(error => { document.getElementById('root').textContent = '报告加载失败：' + error.message; });
</script></body></html>"""
    )


def notebook_html(
    report: "Report",
    *,
    height: int = 1000,
    theme: Literal["light", "dark"] = "light",
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> str:
    """用 iframe 隔离样式与 JS，不修改 Notebook 页面。"""
    if isinstance(height, bool) or not isinstance(height, int) or height < 200:
        raise ValueError("height 不能小于 200")
    document = html.escape(
        report_html(
            report,
            theme=theme,
            annual_trading_days=annual_trading_days,
            risk_free_rate=risk_free_rate,
        ),
        quote=True,
    )
    return (
        f'<iframe title="Scheme 研究报告" style="width:100%;height:{int(height)}px;border:0" '
        f'sandbox="allow-scripts allow-same-origin allow-downloads" srcdoc="{document}"></iframe>'
    )
