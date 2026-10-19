"""因子预处理；暴露独立查询，DOS 直接使用 Arena 的变量和计算步骤。"""

import re
from collections.abc import Sequence

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
    """逐因子执行 3MAD、标准化、行业及 log 市值回归，保留输入键和无效值。"""
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
        normalized = expected.assign(code=expected["code"].map(source_symbol))
        request = {
            "start_date": expected["time"].min().date().isoformat(),
            "end_date": expected["time"].max().date().isoformat(),
            "codes": sorted(normalized["code"].unique().tolist()),
            "factors": [market_value_column, industry_column],
        }
        log.info("因子预处理：{}，{} 行、{} 只股票；独立查询 {} 和 {}",
                 selected, len(expected), len(request["codes"]), market_value_column, industry_column)
        with query(request, settings=settings) as result:
            exposure = result.data
        if (not isinstance(exposure, pd.DataFrame)
                or not set(_KEYS + request["factors"]) <= set(exposure.columns)):
            raise ValueError("exposure query did not return the requested columns")
        if not exposure.empty:
            _keys(exposure)
        exposure["time"] = pd.to_datetime(exposure["time"]).astype("datetime64[ns]")
        exposure["code"] = exposure["code"].map(source_symbol)
        exposure = normalized.merge(
            exposure[_KEYS + request["factors"]], how="left", on=_KEYS,
            sort=False, validate="one_to_one",
        )
        exposure["code"] = expected["code"].to_numpy()
        missing_industry = int(exposure[industry_column].isna().sum())
        missing_market_value = int(exposure[market_value_column].isna().sum())
        if missing_industry or missing_market_value:
            log.warning("暴露缺失：行业 {}/{} 行，市值 {}/{} 行；对应输出保留 NULL",
                        missing_industry, len(exposure), missing_market_value, len(exposure))
        exposure["time"] = exposure["time"].to_numpy(dtype="datetime64[ms]")
        session.upload({
            "factorCols": np.asarray(selected, dtype=str),
            "mktmvCol": market_value_column,
            "industryCol": industry_column,
            "exposure": exposure,
        })
        if frame is not None:
            session.upload({"rawFactorTable": frame})
        source = data if is_variable else "rawFactorTable"
        log.debug("session.run: 拼接因子与暴露，不修改输入表")
        session.run(f"""
            workingTable = sql(sqlCol(symbol(["time", "code"] join factorCols)), {source}).eval()
            replaceColumn!(workingTable, `time, timestamp(workingTable.time))
            replaceColumn!(workingTable, `code, string(workingTable.code))
            workingTable = lj(workingTable, exposure, `time`code)
            workingTable.sortBy!(`time`code)
            addColumn(workingTable, `factor_preprocess_row_id, LONG)
            update workingTable set factor_preprocess_row_id = long(0 .. (workingTable.rows() - 1))
            processedTable = select time, code from workingTable
            hasValues = false
            for (factorCol in factorCols) {{
                if (any(isValid(double(workingTable[factorCol])))) hasValues = true
                processedTable[factorCol] = take(double(NULL), workingTable.rows())
            }}
            if (!hasValues) throw "preprocess factors are all NULL"
        """)
        log.info("执行 MAD 去极值、标准化和行业、市值中性化：{} 个交易日",
                 expected["time"].nunique())
        result = session.run(r"""
            def factorZScore(values) {
                valueStd = stdp(values)
                if (!isValid(valueStd) || valueStd == 0) return take(double(NULL), size(values))
                return (values - avg(values)) \ valueStd
            }
            def factorClipMad(values) {
                valueMedian = median(values)
                mad = median(abs(values - valueMedian))
                if (!isValid(mad) || mad == 0) return values
                return values.clip(valueMedian - 3 * mad, valueMedian + 3 * mad)
            }
            dates = sort(distinct(workingTable.time))
            dateIndex = 0
            for (currentDate in dates) {
                crossSection = select * from workingTable where time = currentDate
                crossSection.sortBy!(`code)
                for (factorCol in factorCols) {
                    factorColSym = symbol([string(factorCol)])[0]
                    rawFactor = double(crossSection[factorColSym])
                    rawMarketValue = double(crossSection[mktmvCol])
                    rawIndustry = string(crossSection[industryCol])
                    validMask = isValid(rawFactor) && isValid(rawMarketValue) && isValid(rawIndustry)
                    if (sum(validMask) == 0) continue
                    rowIds = crossSection.factor_preprocess_row_id[validMask]
                    marketValue = rawMarketValue[validMask]
                    marketValue = iif(marketValue < 1, 1.0, marketValue)
                    mv = factorZScore(log(marketValue))
                    industryValues = rawIndustry[validMask]
                    y = factorZScore(factorClipMad(rawFactor[validMask]))
                    regressionMask = isValid(y) && isValid(mv)
                    if (sum(regressionMask) == 0) continue
                    rowIds = rowIds[regressionMask]
                    y = y[regressionMask]
                    mv = mv[regressionMask]
                    industryValues = industryValues[regressionMask]
                    industries = sort(distinct(industryValues))
                    x = matrix(mv)
                    if (size(industries) > 1) {
                        for (i in 1 .. (size(industries) - 1)) {
                            x = x join matrix(double(industryValues == industries[i]))
                        }
                    }
                    design = matrix(take(1.0, size(y))) join x
                    singularValues = svd(design, false, false)
                    rankTolerance = max(singularValues) * max([rows(design), cols(design)]) * 2.220446049250313e-16
                    effectiveRank = sum(singularValues > rankTolerance)
                    if (size(y) <= effectiveRank) continue
                    beta = ols(y, x, true, 0, "svd")
                    fitted = beta[0] + beta[1] * mv
                    if (size(industries) > 1) {
                        for (i in 1 .. (size(industries) - 1)) {
                            fitted += beta[i + 1] * flatten(x[, i])
                        }
                    }
                    rawResidual = y - fitted
                    residualTolerance = max(abs(y)) * max([rows(design), cols(design)]) * 2.220446049250313e-16
                    if (!isValid(stdp(rawResidual)) || stdp(rawResidual) <= residualTolerance) continue
                    residual = factorZScore(rawResidual)
                    processedValues = processedTable[factorColSym]
                    processedValues[rowIds] = residual
                    processedTable[factorColSym] = processedValues
                }
                dateIndex += 1
                if (dateIndex == 1 || dateIndex % 100 == 0 || dateIndex == size(dates)) {
                    print("因子中性化：" + string(dateIndex) + "/" + string(size(dates)) + "，日期 " + string(date(currentDate)))
                }
            }
            processedTable
        """)
        keys = _keys(result)
        if not set(selected) <= set(result.columns):
            raise ValueError("DolphinDB preprocess result is missing factor columns")
        panel = result[selected].copy().reset_index(drop=True)
        for column in selected:
            panel[column] = pd.to_numeric(panel[column], errors="raise").astype(float)
        panel.index = pd.MultiIndex.from_arrays(
            [keys["time"], keys["code"].map(engine_symbol)], names=["date", "symbol"],
        )
        expected_index = pd.MultiIndex.from_arrays(
            [expected["time"], expected["code"].map(engine_symbol)], names=panel.index.names,
        )
        if len(panel) != len(expected) or not panel.index.sort_values().equals(expected_index.sort_values()):
            raise ValueError("DolphinDB preprocess changed the input time/code keys")
        panel = panel.sort_index()
        log.info("因子预处理完成：{} 行，因子={}，各因子 NULL 数={}",
                 len(panel), selected, panel.isna().sum().to_dict())
        return panel
    except Exception as error:
        log.error("因子预处理失败：{}", error_text(error, settings.password.get_secret_value()))
        raise
    finally:
        if owns_session:
            session.close()
