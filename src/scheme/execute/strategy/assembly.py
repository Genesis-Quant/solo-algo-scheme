"""策略组装契约、默认算法和环节顺序。"""

from collections.abc import Sequence
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin, overload

import pandas as pd
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict

from scheme.base import (
    Algo,
    ControlAlgo,
    ExecutionAlgo,
    OptimizeAlgo,
    ResearchContext,
    StrategyParams,
)

if TYPE_CHECKING:
    from scheme.base.projects.factor.algo import Factor
    from scheme.base.projects.factor.params import FactorParams

__all__ = ["AlgoComponents", "Strategy"]


def _generic_model(
    cls: type, index: int, bindings: dict[Any, Any] | None = None
) -> type[BaseModel]:
    """从 Factor[P] / Algo[P, C] 的继承关系解析参数或上下文模型。"""
    bindings = bindings or {}
    for base in cls.__dict__.get("__orig_bases__", cls.__bases__):
        origin = get_origin(base) or base
        args = tuple(bindings.get(arg, arg) for arg in get_args(base))
        if origin.__name__ in {"Algo", "Factor"} and origin.__module__.split(".")[0] in {
            "backtest",
            "scheme",
        }:
            if len(args) > index:
                model = args[index]
                model = getattr(model, "__bound__", None) or model
                if isinstance(model, type) and issubclass(model, BaseModel):
                    return model
        mapping = dict(zip(getattr(origin, "__type_params__", ()), args))
        if origin is not object:
            try:
                return _generic_model(origin, index, mapping)
            except TypeError:
                pass
    raise TypeError(f"{cls.__name__} 必须明确指定 BaseModel 参数泛型")


@overload
def parameter_type[P: "FactorParams"](cls: "type[Factor[P]]") -> type[P]: ...


@overload
def parameter_type[P: BaseModel, C: ResearchContext[Any]](cls: type[Algo[P, C]]) -> type[P]: ...


def parameter_type(cls: type) -> type[BaseModel]:
    return _generic_model(cls, 0)


def _has_alias_path(value: Any, path: Sequence[str | int]) -> bool:
    """Match attribute-mode lookup without treating an invalid value as missing."""
    for key in path:
        if isinstance(value, str):
            return False
        try:
            if isinstance(value, dict) or isinstance(key, int):
                value = value[key]
            else:
                value = getattr(value, key)
        except (AttributeError, KeyError, IndexError, TypeError):
            return False
    return True


def _with_alias_path(value: Any, path: Sequence[str | int], field_value: Any) -> Any:
    """Copy only alias-path containers; never mutate the source's nested values."""
    if not path:
        return field_value
    key, *rest = path
    if isinstance(value, (list, tuple)) and isinstance(key, int) and -len(value) <= key < len(value):
        items = list(value)
        items[key] = _with_alias_path(items[key], rest, field_value)
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, BaseModel) and isinstance(key, str):
        return value.model_copy(update={
            key: _with_alias_path(getattr(value, key, None), rest, field_value),
        })
    items = dict(value) if isinstance(value, dict) else {}
    items[key] = _with_alias_path(items.get(key), rest, field_value)
    return items


def project_parameters[P: BaseModel](
    model: type[P], source: BaseModel | dict[str, Any], *, overrides: dict[str, Any] | None = None,
) -> P:
    # dict(model) retains raw nested models and fields excluded from serialization.
    values = dict(source)
    overrides = overrides or {}
    canonical = isinstance(source, model)
    for name, field in model.model_fields.items():
        if name in overrides:
            values[name] = overrides[name]
        elif canonical and hasattr(source, name):
            values[name] = getattr(source, name)
        if name not in values or model.model_config.get("validate_by_alias") is False:
            continue
        alias = field.validation_alias
        if alias is None:
            continue
        choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
        paths = [choice.path if isinstance(choice, AliasPath) else [choice] for choice in choices]
        if canonical or name in overrides or not any(_has_alias_path(values, path) for path in paths):
            values = _with_alias_path(values, paths[0], values[name])
    # Attribute validation rebuilds the concrete model and ignores undeclared extras.
    return model.model_validate(SimpleNamespace(**values), from_attributes=True)


def context_type[P: BaseModel, C: ResearchContext[Any]](cls: type[Algo[P, C]]) -> type[C]:
    context = _generic_model(cls, 1)
    if not issubclass(context, ResearchContext):
        raise TypeError("Model 上下文必须继承 ResearchContext")
    return cast(type[C], context)


def validate_context(algo: Any, ctx: BaseModel) -> None:
    expected = _generic_model(type(algo), 1)
    generic = getattr(expected, "__pydantic_generic_metadata__", {})
    if generic.get("args") == (Any,):
        expected = generic["origin"]
    if not isinstance(ctx, expected):
        raise TypeError(
            f"{type(algo).__name__} 需要 {expected.__name__}，收到 {type(ctx).__name__}"
        )


class AlgoComponents[T](BaseModel):
    """T 可以是待安装的包描述；字段与顺序由 scheme 维护。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: T
    optimize: T | None = None
    control: T | None = None
    execution: T | None = None


class Strategy[P: StrategyParams, C: ResearchContext[Any]]:
    """从统一参数实例化 Algo 类，省略的下游环节使用默认算法。"""

    def __init__(
        self,
        ctx: C,
        *,
        params: P,
        model: type[Algo[Any, C]],
        optimize: type[OptimizeAlgo[Any, C]] | None = None,
        control: type[ControlAlgo[Any, C]] | None = None,
        execution: type[ExecutionAlgo[Any, C]] | None = None,
    ) -> None:
        from scheme.base.projects.control.default import NoControl
        from scheme.base.projects.execution.default import DirectExecution
        from scheme.base.projects.optimize.default import RiskParity

        self.ctx = ctx
        self.params = params
        components = (
            (model, Algo),
            (optimize if optimize is not None else RiskParity, OptimizeAlgo),
            (control if control is not None else NoControl, ControlAlgo),
            (execution if execution is not None else DirectExecution, ExecutionAlgo),
        )
        for cls, base in components:
            if not isinstance(cls, type) or not issubclass(cls, base):
                raise TypeError(f"{base.__name__} 环节需要传入对应的 Algo 类")
        self.algos: tuple[Algo[Any, C], ...] = tuple(
            cls(project_parameters(parameter_type(cls), params))
            for cls, _ in components
        )
        for algo in self.algos:
            validate_context(algo, ctx)

    @cached_property
    def universe(self) -> pd.DataFrame:
        """参数区间的股票池 bool 面板，首次访问时查询。"""
        return self.params.universe.evaluate(self.params.start, self.params.end)
