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
from scheme.data.dolphindb.scripts import factor_script
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
    if columns is None:
        selected = [column for column in available if column not in excluded]
    else:
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


def _exposures(keys: pd.DataFrame, missing: list[str], settings) -> pd.DataFrame:
    normalized = keys.assign(code=keys["code"].map(source_symbol))
    request = {
        "start_date": keys["time"].min().date().isoformat(),
        "end_date": keys["time"].max().date().isoformat(),
        "codes": sorted(normalized["code"].unique().tolist()),
        "factors": missing,
    }
    # A new session has no coreQuerySourceData cache. Never give QueryResult the
    # borrowed preprocessing session: its close() closes that session as well.
    with create_session(settings or DolphinSettings.from_env()) as session:
        result = query(request, session=session)
        exposure = result.data.copy()
    if not isinstance(exposure, pd.DataFrame) or not set(_KEYS + missing) <= set(exposure.columns):
        raise ValueError("exposure query did not return the requested columns")
    if not exposure.empty:
        _keys(exposure)
    exposure["time"] = pd.to_datetime(exposure["time"]).astype("datetime64[ns]")
    exposure["code"] = exposure["code"].map(source_symbol)
    merged = normalized.merge(exposure[_KEYS + missing], how="left", on=_KEYS,
                              sort=False, validate="one_to_one")
    merged["code"] = keys["code"].to_numpy()
    return merged


def _canonical(result: pd.DataFrame, columns: list[str], expected: pd.DataFrame) -> pd.DataFrame:
    keys = _keys(result)
    if not set(columns) <= set(result.columns):
        raise ValueError("DolphinDB preprocess result is missing factor columns")
    panel = result[columns].copy().reset_index(drop=True)
    for column in columns:
        panel[column] = pd.to_numeric(panel[column], errors="raise").astype(float)
    panel.index = pd.MultiIndex.from_arrays(
        [keys["time"], keys["code"].map(engine_symbol)], names=["date", "symbol"],
    )
    expected_index = pd.MultiIndex.from_arrays(
        [expected["time"], expected["code"].map(engine_symbol)], names=panel.index.names,
    )
    if len(panel) != len(expected) or not panel.index.sort_values().equals(expected_index.sort_values()):
        raise ValueError("DolphinDB preprocess changed the input time/code keys")
    return panel.sort_index()


def preprocess(
    data: pd.DataFrame | DosVar,
    *,
    session=None,
    columns: Sequence[str] | None = None,
    market_value_column: str = "circ_mv",
    industry_column: str = "industry_l0",
    settings: DolphinSettings | None = None,
) -> pd.DataFrame:
    """Clip, standardize and neutralize selected factors in DolphinDB only.

    Empty inputs and inputs whose selected factors are all NULL are errors.
    Invalid exposures and degenerate cross sections retain their keys with NULL
    outputs. Existing exposure columns are used unchanged (including their NULLs).
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
        frame = data.copy(deep=True)
        if isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == ["date", "symbol"]:
            if set(frame.columns) & {"time", "code", "date", "symbol"}:
                raise ValueError("canonical index collides with input columns")
            frame = frame.reset_index().rename(columns={"date": "time", "symbol": "code"})
        selected = _columns(frame.columns, columns, market_value_column, industry_column)
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
        available = list(frame.columns)

    context = (nullcontext(session) if session is not None
               else create_session(settings or DolphinSettings.from_env()))
    with context as current:
        prefix = "soloFp" + uuid4().hex
        names = {suffix: prefix + suffix for suffix in ("Input", "Factors", "Exposure", "Mv", "Industry", "Result")}
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
            missing = [name for name in (market_value_column, industry_column) if name not in available]
            exposure = _exposures(expected, missing, settings) if missing else None
            current.run(factor_script("preprocess"))
            payload = {
                names["Factors"]: np.asarray(selected, dtype=str),
                names["Mv"]: market_value_column,
                names["Industry"]: industry_column,
            }
            if frame is not None:
                # Extra columns cannot affect factor processing or upload types.
                payload[names["Input"]] = frame[_KEYS + selected + [
                    name for name in (market_value_column, industry_column) if name in available
                ]].copy()
            if exposure is not None:
                exposure["time"] = exposure["time"].to_numpy(dtype="datetime64[ms]")
                payload[names["Exposure"]] = exposure
            current.upload(payload)
            source = names["Input"]
            if is_variable:
                current.run(f"{source} = {data}.copy(); {source}[`time] = timestamp({source}[`time]); "
                            f"{source}[`code] = string({source}[`code])")
            if exposure is not None:
                current.run(f"{source} = lj({source}, {names['Exposure']}, `time`code)")
            current.run(
                f"{names['Result']} = soloFactorPreprocess({source}, {names['Factors']}, "
                f'"time", "code", {names["Mv"]}, {names["Industry"]})'
            )
            return _canonical(current.run(names["Result"]), selected, expected)
        except BaseException:
            failed = True
            raise
        finally:
            cleanup = "; ".join(
                f'if (isDef("{name}")) undef("{name}", VAR)' for name in names.values()
            )
            try:
                current.run(cleanup)
            except Exception:
                if not failed:
                    raise
