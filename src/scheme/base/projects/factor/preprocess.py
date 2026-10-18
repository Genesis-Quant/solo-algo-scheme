"""Session-safe Python transport for DolphinDB factor neutralization."""

import re
from collections.abc import Sequence
from contextlib import nullcontext
from uuid import uuid4

import numpy as np
import pandas as pd
from backtest import DosVar

from scheme.config import DolphinSettings
from scheme.data.dolphindb.api import query
from scheme.data.dolphindb.database.session import create_session
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
    excluded = _RESERVED | {market_value_column, industry_column}
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise ValueError("columns must be a sequence of factor names")
    selected = list(columns)
    if not selected:
        raise ValueError("columns must not be empty")
    for column in selected:
        _name(column, "factor column")
    if len(set(selected)) != len(selected):
        raise ValueError("factor columns must be unique")
    if set(selected) & excluded:
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
    """Clip, standardize and neutralize selected factors in DolphinDB only.

    Empty inputs and inputs whose selected factors are all NULL are errors.
    Invalid exposures and degenerate cross sections retain their keys with NULL
    outputs. Only declared factors are read from data; exposures are always queried
    independently in a separate owned session and left-joined on time/code.
    """
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

    context = (nullcontext(session) if session is not None
               else create_session(settings or DolphinSettings.from_env()))
    with context as current:
        prefix = "soloFp" + uuid4().hex
        names = {suffix: prefix + suffix for suffix in (
            "Input", "Factors", "Exposure", "Mv", "Industry", "Working", "Output",
            "HasValues", "Selected", "Raw", "Day", "DayRows", "CrossSection",
            "RawMarketValue", "RawIndustry", "ValidMask", "MarketValue", "MvValues",
            "MvStd", "Y", "Median", "Mad", "Lower", "Upper", "YStd", "RegressionMask",
            "IndustryValues", "Industries", "X", "Index", "Design", "SingularValues",
            "RankTolerance", "EffectiveRank", "Beta", "Fitted", "RawResidual",
            "ResidualTolerance", "ResidualStd", "Residual", "ProcessedValues", "RowIds",
        )}
        failed = False
        try:
            if is_variable:
                available = list(current.run(f"columnNames({data})"))
                selected = _columns(available, columns, market_value_column, industry_column)
                metadata = current.run(f"schema({data}).colDefs")
                factor_types = metadata.set_index("name").loc[selected, "typeString"]
                if factor_types.str.upper().str.startswith(("BOOL", "COMPLEX")).any():
                    raise ValueError("factor columns must be real numeric values")
                # Download keys only, never the caller's factors or query cache.
                expected = _keys(current.run(f"select time, code from {data}"))
            exposure_columns = [market_value_column, industry_column]
            normalized = expected.assign(code=expected["code"].map(source_symbol))
            request = {
                "start_date": expected["time"].min().date().isoformat(),
                "end_date": expected["time"].max().date().isoformat(),
                "codes": sorted(normalized["code"].unique().tolist()),
                "factors": exposure_columns,
            }
            # QueryResult.close() also closes its session; never borrow current here.
            with create_session(settings or DolphinSettings.from_env()) as query_session:
                query_result = query(request, session=query_session)
                exposure = query_result.data.copy()
            if (not isinstance(exposure, pd.DataFrame)
                    or not set(_KEYS + exposure_columns) <= set(exposure.columns)):
                raise ValueError("exposure query did not return the requested columns")
            if not exposure.empty:
                _keys(exposure)
            exposure["time"] = pd.to_datetime(exposure["time"]).astype("datetime64[ns]")
            exposure["code"] = exposure["code"].map(source_symbol)
            exposure = normalized.merge(
                exposure[_KEYS + exposure_columns], how="left", on=_KEYS,
                sort=False, validate="one_to_one",
            )
            exposure["code"] = expected["code"].to_numpy()
            payload = {
                names["Factors"]: np.asarray(selected, dtype=str),
                names["Mv"]: market_value_column,
                names["Industry"]: industry_column,
            }
            if frame is not None:
                payload[names["Input"]] = frame
            exposure["time"] = exposure["time"].to_numpy(dtype="datetime64[ms]")
            payload[names["Exposure"]] = exposure
            current.upload(payload)
            source = data if is_variable else names["Input"]
            result = current.run(f'''
{prefix}Working = sql(sqlCol(symbol(["time", "code"] join {prefix}Factors)), {source}).eval()
replaceColumn!({prefix}Working, `time, timestamp({prefix}Working[`time]))
replaceColumn!({prefix}Working, `code, string({prefix}Working[`code]))
{prefix}Working = lj({prefix}Working, {prefix}Exposure, `time`code)
{prefix}Working.sortBy!(`time`code)
{prefix}Output = select time, code from {prefix}Working
{prefix}HasValues = false
for ({prefix}Selected in {prefix}Factors) {{
    {prefix}Raw = double({prefix}Working[{prefix}Selected])
    if (any(isValid({prefix}Raw))) {prefix}HasValues = true
    {prefix}Output[{prefix}Selected] = take(double(NULL), {prefix}Working.rows())
}}
if (!{prefix}HasValues) throw "preprocess factors are all NULL"
for ({prefix}Day in sort(distinct({prefix}Working[`time]))) {{
    {prefix}DayRows = at({prefix}Working[`time] == {prefix}Day)
    {prefix}CrossSection = {prefix}Working[{prefix}Working[`time] == {prefix}Day]
    for ({prefix}Selected in {prefix}Factors) {{
        {prefix}Raw = double({prefix}CrossSection[{prefix}Selected])
        {prefix}RawMarketValue = double({prefix}CrossSection[{prefix}Mv])
        {prefix}RawIndustry = string({prefix}CrossSection[{prefix}Industry])
        {prefix}ValidMask = isValid({prefix}Raw) && isValid({prefix}RawMarketValue) && isValid({prefix}RawIndustry)
        if (sum({prefix}ValidMask) == 0) continue
        {prefix}MarketValue = {prefix}RawMarketValue[{prefix}ValidMask]
        {prefix}MarketValue = iif({prefix}MarketValue < 1, 1.0, {prefix}MarketValue)
        {prefix}MvValues = log({prefix}MarketValue)
        {prefix}MvStd = stdp({prefix}MvValues)
        if (isNull({prefix}MvStd) || {prefix}MvStd <= 0) continue
        {prefix}MvValues = ({prefix}MvValues - avg({prefix}MvValues)) \\ {prefix}MvStd
        {prefix}Y = {prefix}Raw[{prefix}ValidMask]
        {prefix}Median = median({prefix}Y)
        {prefix}Mad = median(abs({prefix}Y - {prefix}Median))
        if (isValid({prefix}Mad) && {prefix}Mad > 0) {{
            {prefix}Lower = {prefix}Median - 3.0 * {prefix}Mad
            {prefix}Upper = {prefix}Median + 3.0 * {prefix}Mad
            {prefix}Y = iif({prefix}Y < {prefix}Lower, {prefix}Lower, iif({prefix}Y > {prefix}Upper, {prefix}Upper, {prefix}Y))
        }}
        {prefix}YStd = stdp({prefix}Y)
        if (isNull({prefix}YStd) || {prefix}YStd <= 0) continue
        {prefix}Y = ({prefix}Y - avg({prefix}Y)) \\ {prefix}YStd
        {prefix}RegressionMask = isValid({prefix}Y) && isValid({prefix}MvValues)
        {prefix}Y = {prefix}Y[{prefix}RegressionMask]
        {prefix}MvValues = {prefix}MvValues[{prefix}RegressionMask]
        if (size({prefix}Y) == 0) continue
        {prefix}IndustryValues = {prefix}RawIndustry[{prefix}ValidMask][{prefix}RegressionMask]
        {prefix}RowIds = {prefix}DayRows[{prefix}ValidMask][{prefix}RegressionMask]
        {prefix}Industries = sort(distinct({prefix}IndustryValues))
        {prefix}X = matrix({prefix}MvValues)
        if (size({prefix}Industries) > 1) {{
            for ({prefix}Index in 1 .. (size({prefix}Industries) - 1)) {{
                {prefix}X = {prefix}X join matrix(double({prefix}IndustryValues == {prefix}Industries[{prefix}Index]))
            }}
        }}
        {prefix}Design = matrix(take(1.0, size({prefix}Y))) join {prefix}X
        {prefix}SingularValues = svd({prefix}Design, false, false)
        {prefix}RankTolerance = max([rows({prefix}Design), cols({prefix}Design)]) * max({prefix}SingularValues) * 2.220446049250313e-16
        {prefix}EffectiveRank = sum({prefix}SingularValues > {prefix}RankTolerance)
        if (size({prefix}Y) <= {prefix}EffectiveRank) continue
        {prefix}Beta = ols({prefix}Y, {prefix}X, true, 0, "svd")
        {prefix}Fitted = {prefix}Beta[0] + {prefix}Beta[1] * {prefix}MvValues
        if (size({prefix}Industries) > 1) {{
            for ({prefix}Index in 1 .. (size({prefix}Industries) - 1)) {{
                {prefix}Fitted = {prefix}Fitted + {prefix}Beta[{prefix}Index + 1] * flatten({prefix}X[, {prefix}Index])
            }}
        }}
        {prefix}RawResidual = {prefix}Y - {prefix}Fitted
        {prefix}ResidualTolerance = max(abs({prefix}Y)) * max([rows({prefix}Design), cols({prefix}Design)]) * 2.220446049250313e-16
        {prefix}ResidualStd = stdp({prefix}RawResidual)
        if (isNull({prefix}ResidualStd) || {prefix}ResidualStd <= {prefix}ResidualTolerance) continue
        {prefix}Residual = ({prefix}RawResidual - avg({prefix}RawResidual)) \\ {prefix}ResidualStd
        {prefix}ProcessedValues = {prefix}Output[{prefix}Selected]
        {prefix}ProcessedValues[{prefix}RowIds] = {prefix}Residual
        {prefix}Output[{prefix}Selected] = {prefix}ProcessedValues
    }}
}}
{prefix}Output
''')
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
            if (len(panel) != len(expected)
                    or not panel.index.sort_values().equals(expected_index.sort_values())):
                raise ValueError("DolphinDB preprocess changed the input time/code keys")
            return panel.sort_index()
        except BaseException:
            failed = True
            raise
        finally:
            cleanup = "; ".join(
                f'if ("{name}" in objs().name) {{ undef("{name}", VAR) }}' for name in names.values()
            )
            try:
                current.run(cleanup)
            except Exception:
                if not failed:
                    raise
