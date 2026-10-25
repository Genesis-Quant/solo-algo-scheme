"""在 DolphinDB 内连接因子、所属行业和流通市值并执行中性化。"""

import re
from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import pandas as pd
from backtest import DosVar
from runtime.apps.query import api as query_api
from runtime.utils import logger as log

from scheme.config import DolphinSettings
from scheme.data.dolphindb.api import QUERY_SCRIPT
from scheme.data.dolphindb.database.session import create_session, download, release, upload
from scheme.data.dolphindb.symbols import engine_symbol
from scheme.errors import error_text

RESERVED = {"time", "code", "date", "symbol", "stock_pool_member"}

PREPROCESS_SCRIPT = """
use factor

def preprocessFactor(inputTable, settings) {
    required = ["time", "code"] join settings.columns
    metadata = schema(inputTable).colDefs
    if (any(!(required in metadata.name))) throw "因子表缺少日期、证券或声明的因子列"
    if (size(distinct(metadata.name)) != metadata.rows()) throw "因子表列名重复"
    factorTypes = exec typeString from metadata where name in settings.columns
    if (any(startsWith(factorTypes, "BOOL") || startsWith(factorTypes, "COMPLEX")))
        throw "因子值必须是实数"

    input = sql(
        sqlCol(`time, timestamp, `time) join sqlCol(`code, string, `code) join
            sqlCol(symbol(settings.columns), double, symbol(settings.columns)),
        inputTable
    ).eval()
    if (input.rows() == 0) throw "因子表不能为空"
    if (any(isNull(input.time) || timestamp(date(input.time)) != input.time))
        throw "因子日期必须是非空的日历日期"
    if (any(isNull(input.code) || strlen(trim(input.code)) == 0))
        throw "因子证券代码不能为空"
    replaceColumn!(
        input,
        `code,
        strReplace(strReplace(input.code, ".XSHG", ".SH"), ".XSHE", ".SZ")
    )
    duplicates = select count(*) as copies from input group by time, code
    if (any(duplicates.copies > 1)) throw "规范化证券代码后存在重复日期和证券"
    if (!any(each(any, each(isValid, input[symbol(settings.columns)])))) throw "因子值全部为空"

    print("[DOS 预处理] 读取每日所属行业和流通市值，因子记录数=" + string(input.rows()))
    request = dict(STRING, ANY)
    request["database"] = settings.database
    request["table"] = settings.table
    request["start"] = min(input.time)
    request["outputStart"] = request["start"]
    request["end"] = temporalAdd(max(input.time), 1, "d")
    request["codes"] = distinct(input.code)
    request["factors"] = settings.factors
    request["industry"] = settings.industry
    request["financial"] = settings.financial
    request["weights"] = settings.weights
    request["st"] = settings.st
    request["definitions"] = "{}"
    request["filters"] = array(STRING, 0)
    request["columns"] = ["time", "code"] join settings.factors
    reference = buildQuery(request).data
    replaceColumn!(reference, `code, string(reference.code))
    print("[DOS 预处理] 按日期和证券连接因子、所属行业及流通市值")
    working = lj(input, reference, `time`code)
    print("[DOS 中性化] 开始 factor::factorPreprocess，因子=" + concat(settings.columns, ", "))
    processed = factor::factorPreprocess(
        working,
        settings.columns,
        settings.groups,
        "time",
        "code",
        settings.market,
        settings.industryColumn
    )
    print("[DOS 中性化] 完成，校验日期和证券键")
    if (
        !eqObj(
            (exec time from input order by time, code),
            (exec time from processed order by time, code)
        ) || !eqObj(
            (exec code from input order by time, code),
            (exec code from processed order by time, code)
        )
    ) throw "中性化改变了输入日期或证券"
    return sql(sqlCol(symbol(["time", "code"] join settings.columns)), processed).eval()
}
"""


def preprocess(
    data: pd.DataFrame | DosVar,
    *,
    columns: Sequence[str],
    groups: int,
    session=None,
    market_value_column: str = "circ_mv",
    industry_column: str = "industry_l0",
    settings: DolphinSettings | None = None,
) -> pd.DataFrame:
    if (
        isinstance(groups, bool)
        or not isinstance(groups, int)
        or not 2 <= groups <= np.iinfo(np.int32).max
    ):
        raise ValueError("分组数量必须是至少为 2 的 INT 整数")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence) or not columns:
        raise ValueError("因子列必须是非空名称列表")
    selected = list(columns)
    for name in [*selected, market_value_column, industry_column]:
        if not isinstance(name, str) or not name.strip() or "\0" in name:
            raise ValueError("列名必须是非空字符串且不能包含 NUL")
    if len(set(selected)) != len(selected):
        raise ValueError("因子列不能重复")
    if (
        market_value_column == industry_column
        or {market_value_column, industry_column} & RESERVED
        or set(selected) & (RESERVED | {market_value_column, industry_column})
    ):
        raise ValueError("因子、行业、市值和日期证券列不能重名")
    factors = [market_value_column, industry_column]
    if unknown := set(factors) - set(query_api.available_factors()):
        raise ValueError(f"行业或市值字段未声明：{sorted(unknown)}")
    is_variable = isinstance(data, str)
    if is_variable:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", data) is None:
            raise ValueError("DosVar 必须是单个 DolphinDB 变量名")
        if session is None:
            raise ValueError("DosVar 必须提供所属 session")
        if data in {"preprocessInput", "preprocessSettings", "preprocessResult"}:
            raise ValueError("输入变量名与中性化内部变量重名")
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("预处理输入必须是 DataFrame 或 DosVar")
    input_ref = data if is_variable else "preprocessInput"
    if not is_variable:
        frame = data
        if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == ["date", "symbol"]:
            if set(frame.columns) & {"time", "code", "date", "symbol"}:
                raise ValueError("面板索引和日期证券列重名")
            frame = frame.reset_index().rename(columns={"date": "time", "symbol": "code"})
        if not frame.columns.is_unique or not {"time", "code", *selected} <= set(frame.columns):
            raise ValueError("输入缺列或列名重复")
        frame = frame[["time", "code", *selected]].copy()
        if frame.empty:
            raise ValueError("因子表不能为空")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["time"], errors="raise"))
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("time 必须是非空、无时区的日历日期")
        if not frame["code"].map(lambda code: isinstance(code, str) and bool(code.strip())).all():
            raise ValueError("code 必须是非空字符串")
        frame["time"] = dates.to_numpy(dtype="datetime64[ms]")
        frame["code"] = (
            frame["code"]
            .str.replace(".XSHG", ".SH", regex=False)
            .str.replace(".XSHE", ".SZ", regex=False)
        )
        if frame.duplicated(["time", "code"]).any():
            raise ValueError("规范化证券代码后存在重复日期和证券")
        for column in selected:
            if (
                pd.api.types.is_bool_dtype(frame[column])
                or pd.api.types.is_complex_dtype(frame[column])
                or (
                    not pd.api.types.is_numeric_dtype(frame[column])
                    and frame[column]
                    .map(
                        lambda value: isinstance(
                            value, (bool, np.bool_, complex, np.complexfloating)
                        )
                    )
                    .any()
                )
            ):
                raise ValueError("因子值必须是实数")
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not frame[selected].notna().any().any():
            raise ValueError("因子值全部为空")
    settings = settings if settings is not None else DolphinSettings.from_env()
    owns_session = session is None
    with create_session(settings) if owns_session else nullcontext(session) as session:
        try:
            query_api.redirect_session_output(session)
            upload(
                session,
                {
                    "preprocessSettings": {
                        "database": settings.database,
                        "table": settings.table,
                        "columns": np.asarray(selected, dtype=str),
                        "groups": np.int32(groups),
                        "market": market_value_column,
                        "industryColumn": industry_column,
                        "factors": np.asarray(factors, dtype=str),
                        "industry": np.asarray(
                            [f for f in factors if f in query_api.INDUSTRY_FACTORS], dtype=str
                        ),
                        "financial": np.asarray(
                            [f for f in factors if f in query_api.FINANCIAL_FACTORS], dtype=str
                        ),
                        "weights": np.asarray(
                            [f for f in factors if f.startswith(query_api.WEIGHT_PREFIX)], dtype=str
                        ),
                        "st": query_api.IS_ST_FACTOR,
                    }
                },
                "中性化列设置",
            )
            if not is_variable:
                upload(session, {input_ref: frame}, "本地因子表")
            session.run(QUERY_SCRIPT)
            session.run(PREPROCESS_SCRIPT)
            session.run(
                f"preprocessResult = preprocessFactor({input_ref}, preprocessSettings); NULL"
            )
            processed = download(session, "preprocessResult", "最终中性化因子")
            panel = processed[selected].astype(float)
            panel.index = pd.MultiIndex.from_arrays(
                [
                    pd.to_datetime(processed["time"]).to_numpy(dtype="datetime64[ns]"),
                    processed["code"].map(engine_symbol),
                ],
                names=["date", "symbol"],
            )
            panel = panel.sort_index()
            log.info(f"[预处理] 完成：{len(panel)} 行；各因子空值数 {panel.isna().sum().to_dict()}")
            return panel
        except BaseException as error:
            log.error(f"[预处理] 失败：{error_text(error, settings.password.get_secret_value())}")
            raise
        finally:
            if not owns_session:
                release(
                    session,
                    [
                        "preprocessSettings",
                        "preprocessResult",
                        *([] if is_variable else ["preprocessInput"]),
                    ],
                    ["preprocessFactor", "buildQuery"],
                )
