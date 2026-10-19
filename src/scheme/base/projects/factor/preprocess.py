"""按完整交易日分批预处理，暴露仍按原完整区间独立查询。"""

import json
import re
from collections.abc import Sequence
from datetime import timedelta

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
    """逐日中性化；每批保留完整截面，不切割证券域或重置暴露填充。"""
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
            # pandas 的转换可能依赖整列类型推断，不能拆批后再转换。
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if frame[selected].isna().all().all():
            raise ValueError("preprocess factors are all NULL")
        input_dates = pd.DatetimeIndex(frame["time"])
        dates = input_dates.unique().sort_values()
        codes = frame["code"].unique().tolist()
        row_count = len(frame)
        del expected

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
            row_count, times, codes = session.run(
                f"[{data}.rows(), sort(distinct({data}.time)), distinct({data}.code)]"
            )
            dates = pd.DatetimeIndex(times).unique().sort_values()
        if not row_count:
            raise ValueError("preprocess input must not be empty")
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("time must contain non-NULL, timezone-naive calendar dates")
        if not all(isinstance(code, str) and bool(code.strip()) for code in codes):
            raise ValueError("code must contain nonempty strings")
        request = {
            "start_date": dates[0].date().isoformat(),
            "end_date": dates[-1].date().isoformat(),
            "codes": sorted({source_symbol(code) for code in codes}),
            "factors": [market_value_column, industry_column],
        }
        log.info("因子预处理：{}，{} 行、{} 只股票、{} 个交易日；每批最多 64 个完整交易日",
                 selected, row_count, len(request["codes"]), len(dates))
        panels = []
        has_values = False
        missing_industry = missing_market_value = 0
        with query(request, settings=settings) as exposure_result:
            # 函数参数与工作变量只在当前批次内存活，不占用或覆盖调用方的表变量。
            session.run(r"""
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
                def factorPreprocess(mutable workingTable, factorCols, mktmvCol, industryCol) {
                    replaceColumn!(workingTable, `time, timestamp(workingTable.time))
                    replaceColumn!(workingTable, `code, string(workingTable.code))
                    workingTable.sortBy!(`time`code)
                    addColumn(workingTable, `factor_preprocess_row_id, LONG)
                    update workingTable set factor_preprocess_row_id = long(0 .. (workingTable.rows() - 1))
                    processedTable = select time, code from workingTable
                    hasValues = false
                    for (factorCol in factorCols) {
                        if (any(isValid(double(workingTable[factorCol])))) hasValues = true
                        processedTable[factorCol] = take(double(NULL), workingTable.rows())
                    }
                    dates = sort(distinct(workingTable.time))
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
                    }
                    return [processedTable, hasValues]
                }
            """)
            for offset in range(0, len(dates), 64):
                begin = dates[offset]
                end = dates[offset + 64] if offset + 64 < len(dates) else dates[-1] + timedelta(days=1)
                if is_variable:
                    # DosVar 沿用 DolphinDB 的 double 转换，不经 pandas 重新解析。
                    batch = session.run(f"""
                        sql(sqlCol(`time, timestamp, `time) join sqlCol(`code) join sqlCol(
                                symbol({json.dumps(selected, ensure_ascii=False)}), double,
                                symbol({json.dumps(selected, ensure_ascii=False)})), {data},
                            [<timestamp(time) >= {begin:%Y.%m.%d}>,
                             <timestamp(time) < {end:%Y.%m.%d}>]).eval()
                    """)
                else:
                    mask = (input_dates >= begin) & (input_dates < end)
                    batch = frame.loc[mask].copy()
                expected = _keys(batch)
                exposure = exposure_result.session.run(f"""
                    select * from {exposure_result.data_ref}
                    where time >= {begin:%Y.%m.%d}, time < {end:%Y.%m.%d}
                """)
                if not exposure.empty:
                    _keys(exposure)
                exposure["time"] = pd.to_datetime(exposure["time"]).astype("datetime64[ns]")
                exposure["code"] = exposure["code"].map(source_symbol)
                aligned = expected.assign(code=expected["code"].map(source_symbol)).merge(
                    exposure, how="left", on=_KEYS, sort=False, validate="one_to_one",
                )
                batch["time"] = expected["time"].to_numpy(dtype="datetime64[ms]")
                batch["code"] = expected["code"].to_numpy()
                for column in request["factors"]:
                    batch[column] = aligned[column].to_numpy()
                missing_industry += int(batch[industry_column].isna().sum())
                missing_market_value += int(batch[market_value_column].isna().sum())
                processed, valid = session.run(
                    "factorPreprocess", batch, np.asarray(selected, dtype=str),
                    market_value_column, industry_column,
                )
                has_values = has_values or bool(valid)
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
                panels.append(panel)
                log.info("因子中性化：{}/{} 个交易日，当前批次 {} 行，因子={}",
                         min(offset + 64, len(dates)), len(dates), len(panel), selected)
        if not has_values:
            raise ValueError("preprocess factors are all NULL")
        panel = pd.concat(panels).sort_index()
        if len(panel) != row_count:
            raise ValueError("DolphinDB preprocess changed the input row count")
        if missing_industry or missing_market_value:
            log.warning("暴露缺失：行业 {}/{} 行，市值 {}/{} 行；对应输出保留 NULL",
                        missing_industry, row_count, missing_market_value, row_count)
        log.info("因子预处理完成：{} 行，因子={}，各因子 NULL 数={}",
                 len(panel), selected, panel.isna().sum().to_dict())
        return panel
    except Exception as error:
        log.error("因子预处理失败：{}", error_text(error, settings.password.get_secret_value()))
        raise
    finally:
        if owns_session:
            session.close()
