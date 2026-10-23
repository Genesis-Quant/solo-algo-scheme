"""通过原 factor 模块完成因子预处理，不在 Python 中维护 DOS 算法副本。"""

import json
import re
from collections.abc import Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from backtest import DosVar
from runtime.utils import logger as log

from scheme._errors import error_text
from scheme.config import DolphinSettings
from scheme.data.dolphindb.api import query
from scheme.data.dolphindb.database.session import open_session
from scheme.data.dolphindb.symbols import engine_symbol, source_symbol

_KEYS = ["time", "code"]
_RESERVED = {*_KEYS, "date", "symbol", "stock_pool_member"}


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{label} must be a nonempty column name")
    return value


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not set(_KEYS) <= set(frame.columns):
        raise ValueError("preprocess requires time and code columns")
    if frame.empty:
        raise ValueError("preprocess input must not be empty")
    keys = frame[_KEYS].copy().reset_index(drop=True)
    keys["time"] = pd.to_datetime(keys["time"], errors="raise")
    days = pd.DatetimeIndex(keys["time"])
    if days.hasnans or days.tz is not None or not days.equals(days.normalize()):
        raise ValueError("time must contain non-NULL, timezone-naive calendar dates")
    keys["time"] = days.as_unit("ns")
    if not keys["code"].map(lambda code: isinstance(code, str) and bool(code.strip())).all():
        raise ValueError("code must contain nonempty strings")
    normalized = keys.assign(code=keys["code"].map(engine_symbol))
    if normalized.duplicated(_KEYS).any():
        raise ValueError("duplicate time/code keys after symbol normalization")
    return keys


def _columns(available, columns, market_value_column, industry_column) -> list[str]:
    if len(set(available)) != len(available):
        raise ValueError("input column names must be unique")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise ValueError("columns must be a sequence of factor names")
    selected = list(columns)
    if not selected:
        raise ValueError("columns must not be empty")
    for column in selected:
        _name(column, "factor column")
    if len(set(selected)) != len(selected):
        raise ValueError("factor columns must be unique")
    if set(selected) & (_RESERVED | {market_value_column, industry_column}):
        raise ValueError("factor columns collide with keys or exposure columns")
    if missing := set(selected) - set(available):
        raise ValueError(f"missing factor columns: {sorted(missing)}")
    return selected


def preprocess(
    data: pd.DataFrame | DosVar,
    *,
    columns: Sequence[str],
    session=None,
    market_value_column: str = "circ_mv",
    industry_column: str = "industry_l0",
    settings: DolphinSettings | None = None,
) -> pd.DataFrame:
    """保留输入键和 NULL，调用原 factor::factorPreprocess 后只返回因子列。"""
    for value, label in ((market_value_column, "market_value_column"),
                         (industry_column, "industry_column")):
        _name(value, label)
    if (market_value_column == industry_column
            or {market_value_column, industry_column} & _RESERVED):
        raise ValueError("exposure column names collide with keys or each other")
    is_variable = isinstance(data, str)
    if is_variable:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", data) is None:
            raise ValueError("DosVar must be a single DolphinDB identifier")
        if session is None:
            raise ValueError("DosVar requires an explicit session")
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("preprocess data must be a DataFrame or DosVar")

    frame = None
    if not is_variable:
        frame = data
        if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == ["date", "symbol"]:
            if set(frame.columns) & {"time", "code", "date", "symbol"}:
                raise ValueError("canonical index collides with input columns")
            frame = frame.reset_index().rename(columns={"date": "time", "symbol": "code"})
        selected = _columns(frame.columns, columns, market_value_column, industry_column)
        frame = frame[_KEYS + selected].copy(deep=True)
        expected = _keys(frame)
        frame["time"] = expected["time"].to_numpy(dtype="datetime64[ms]")
        frame["code"] = expected["code"].map(source_symbol).to_numpy()
        expected = _keys(frame)
        for column in selected:
            if pd.api.types.is_bool_dtype(frame[column]) or pd.api.types.is_complex_dtype(frame[column]):
                raise ValueError("factor columns must be real numeric values")
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if frame[selected].isna().all().all():
            raise ValueError("preprocess factors are all NULL")

    settings = settings if settings is not None else DolphinSettings.from_env()
    owns_session = session is None
    if owns_session:
        session = open_session(settings)
    prefix = "soloFp" + uuid4().hex
    names = {key: prefix + key for key in ("Input", "Exposure", "Working", "Processed", "Columns", "Market", "Industry")}
    try:
        if is_variable:
            selected = _columns(
                list(session.run(f"columnNames({data})")), columns,
                market_value_column, industry_column,
            )
            metadata = session.run(f"schema({data}).colDefs")
            factor_types = metadata.set_index("name").loc[selected, "typeString"]
            if factor_types.str.upper().str.startswith(("BOOL", "COMPLEX")).any():
                raise ValueError("factor columns must be real numeric values")
            expected = _keys(session.run(f"select time, code from {data}"))
            session.run(f"""
                {names['Input']} = sql(
                    sqlCol(`time, timestamp, `time) join sqlCol(`code, string, `code) join
                    sqlCol(symbol({json.dumps(selected, ensure_ascii=False)}), double,
                        symbol({json.dumps(selected, ensure_ascii=False)})), {data}).eval()
            """)
        else:
            session.upload({names["Input"]: frame})

        request = {
            "start_date": expected["time"].min().date().isoformat(),
            "end_date": expected["time"].max().date().isoformat(),
            "codes": sorted({source_symbol(code) for code in expected["code"]}),
            "factors": [market_value_column, industry_column],
        }
        log.info("因子预处理：{}，{} 行、{} 只股票；使用 factor 模块",
                 selected, len(expected), len(request["codes"]))
        with query(request, settings=settings) as exposure_result:
            exposure = exposure_result.data
        if not exposure.empty:
            _keys(exposure)
        exposure["time"] = pd.to_datetime(exposure["time"]).astype("datetime64[ns]")
        exposure["code"] = exposure["code"].map(source_symbol)
        aliases = expected[["code"]].drop_duplicates().assign(
            source_code=lambda value: value["code"].map(source_symbol),
        ).rename(columns={"code": "input_code"})
        exposure = exposure.merge(
            aliases, left_on="code", right_on="source_code", how="inner", validate="many_to_one",
        )
        exposure["code"] = exposure.pop("input_code")
        exposure = exposure[_KEYS + request["factors"]]
        exposure["time"] = exposure["time"].to_numpy(dtype="datetime64[ms]")
        session.upload({
            names["Exposure"]: exposure,
            names["Columns"]: np.asarray(selected, dtype=str),
            names["Market"]: market_value_column,
            names["Industry"]: industry_column,
        })
        del exposure
        session.run(f"""
            use factor
            {names['Working']} = lj({names['Input']}, {names['Exposure']}, `time`code)
        """)
        has_values = session.run(
            f"any(each(any, each(isValid, {names['Input']}[symbol({names['Columns']})])))"
        )
        if not has_values:
            raise ValueError("preprocess factors are all NULL")
        session.run(f"""
            {names['Processed']} = factor::factorPreprocess(
                {names['Working']}, {names['Columns']}, 5, "time", "code",
                {names['Market']}, {names['Industry']})
        """)
        processed = session.run(
            f"sql(sqlCol(symbol([\"time\", \"code\"] join {names['Columns']})), "
            f"{names['Processed']}).eval()"
        )
        keys = _keys(processed)
        if len(processed) != len(expected) or not set(selected) <= set(processed.columns):
            raise ValueError("DolphinDB preprocess changed the input shape")
        expected_index = pd.MultiIndex.from_arrays(
            [expected["time"], expected["code"].map(engine_symbol)], names=["date", "symbol"],
        )
        panel = processed[selected].astype(float).copy()
        panel.index = pd.MultiIndex.from_arrays(
            [keys["time"], keys["code"].map(engine_symbol)], names=expected_index.names,
        )
        if not panel.index.sort_values().equals(expected_index.sort_values()):
            raise ValueError("DolphinDB preprocess changed the input time/code keys")
        panel = panel.sort_index()
        log.info("因子预处理完成：{} 行，因子={}，各因子 NULL 数={}",
                 len(panel), selected, panel.isna().sum().to_dict())
        return panel
    except Exception as error:
        log.error("因子预处理失败：{}", error_text(error, settings.password.get_secret_value()))
        raise
    finally:
        try:
            session.run(f"undef({json.dumps(list(names.values()))}, VAR)")
        except Exception:
            pass
        if owns_session:
            session.close()
