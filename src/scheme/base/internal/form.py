"""UI 表单只负责把简化输入转换成具体分析参数。"""

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import date, timedelta
from functools import cache
from typing import Any, ClassVar, Literal, TypeVar, cast

from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    SkipValidation,
    WrapValidator,
    create_model,
)

from scheme.data.dolphindb.universe import StockPool, Universe

__all__ = ["ReportForm"]


class ReportForm[P: BaseModel](BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @abstractmethod
    def build(self) -> P:
        """构造并校验分析参数；不执行研究。"""
        ...


def _input_key(model: type[BaseModel], name: str) -> str:
    alias = model.model_fields[name].validation_alias
    if model.model_config.get("validate_by_alias") is False:
        return name
    if isinstance(alias, AliasChoices):
        alias = next((choice for choice in alias.choices if isinstance(choice, str)), None)
        if alias is None:
            raise TypeError(f"Form 字段 {name!r} 的 AliasPath 不支持扁平表单 JSON")
    if isinstance(alias, AliasPath):
        raise TypeError(f"Form 字段 {name!r} 的 AliasPath 不支持扁平表单 JSON")
    return alias if isinstance(alias, str) else name


def _field_owner(model: type[BaseModel], name: str) -> type | None:
    return next(
        (base for base in model.__mro__ if name in base.__dict__.get("__annotations__", {})),
        None,
    )


def _check_aliases(model: type[BaseModel]) -> None:
    owners: dict[str, str] = {}
    for name, field in model.model_fields.items():
        _input_key(model, name)
        aliases = field.validation_alias
        choices = aliases.choices if isinstance(aliases, AliasChoices) else [aliases]
        keys = set()
        if model.model_config.get("validate_by_alias") is not False:
            keys.update(
                alias.path[0] if isinstance(alias, AliasPath) else alias
                for alias in choices if alias is not None
            )
        if not keys or model.model_config.get("validate_by_name"):
            keys.add(name)
        for key in keys:
            if key in owners and owners[key] != name:
                raise TypeError(
                    f"Form 字段的 validation alias 必须唯一：{name!r} 与 {owners[key]!r}"
                    f" 共用 {key!r}"
                )
            owners[key] = name


def _input_field(model: type[BaseModel], name: str):
    copied = deepcopy(model.model_fields[name])
    validators = {
        "before": BeforeValidator, "after": AfterValidator,
        "plain": PlainValidator, "wrap": WrapValidator,
    }
    for decorator in model.__pydantic_decorators__.field_validators.values():
        info = decorator.info
        if name not in info.fields and "*" not in info.fields:
            continue
        options = {} if info.mode == "after" else {
            "json_schema_input_type": info.json_schema_input_type,
        }
        copied.metadata.append(validators[info.mode](decorator.func, **options))
    # 保留约束与输入 schema；业务转换只交给最终 Analysis，不能在 Form 先转换一次。
    copied.metadata.append(SkipValidation())
    return copied


class ResearchForm[P: BaseModel](ReportForm[P]):
    """公共研究输入；具体类型绑定后展开项目声明的分析字段。"""

    _analysis_base: ClassVar[type[BaseModel] | None] = None
    _analysis_model: ClassVar[type[BaseModel] | None] = None
    _internal_fields: ClassVar[frozenset[str]] = frozenset({"universe", "config", "symbols"})

    start: date = Field(default=date(2020, 1, 1), title="开始日期")
    end: date = Field(default=date(2027, 1, 1), title="结束日期（不含）")
    pool: Literal[
        StockPool.ALL, StockPool.SSE50, StockPool.CSI300, StockPool.CSI500, StockPool.CSI1000
    ] = Field(default=StockPool.CSI300, title="股票池")
    lookback: timedelta = Field(default=timedelta(0), title="回溯周期")

    @classmethod
    def __class_getitem__(cls, target: Any) -> type:
        if isinstance(target, tuple) and len(target) == 1:
            target = target[0]
        return cls._specialize(target)

    @classmethod
    @cache
    def _specialize(cls, target: Any) -> type:
        if cls._analysis_base is None or isinstance(target, TypeVar):
            parent = super().__class_getitem__(target)
            if parent is not cls:
                parent._analysis_model = None
            return parent
        if not isinstance(target, type) or not issubclass(target, cls._analysis_base):
            raise TypeError(f"{cls.__name__} 需要 {cls._analysis_base.__name__} 或其分析子类")
        if target.__pydantic_generic_metadata__.get("parameters"):
            raise TypeError(f"{cls.__name__} 必须绑定具体分析参数类型，不能保留未绑定泛型")

        _check_aliases(target)
        fields = {}
        for name, field in target.model_fields.items():
            if field.default_factory_takes_validated_data:
                raise TypeError(
                    f"自动 Form 字段 {name!r} 不支持依赖其他字段的 default_factory；"
                    "请使用无参默认工厂，或在分析参数的 model_validator 中推导"
                )
            if name not in cls._analysis_base.model_fields:
                if name in cls.model_fields or name in cls._internal_fields or hasattr(cls, name):
                    raise TypeError(f"分析字段 {name!r} 与 Scheme 表单保留字段冲突")
            if name in cls._internal_fields:
                continue
            inherited = _field_owner(target, name) is _field_owner(cls._analysis_base, name)
            if name not in cls.model_fields or not inherited:
                fields[name] = (field.annotation, _input_field(target, name))

        parent = super().__class_getitem__(target)
        generated = create_model(
            parent.__name__, __base__=parent, __module__=cls.__module__, **fields,
        )
        generated._analysis_model = target
        # Analysis 的 extra/validators 不属于 UI；只沿用输入别名策略。
        generated.model_config = ConfigDict(**{
            **generated.model_config,
            **{key: target.model_config[key] for key in (
                "populate_by_name", "validate_by_alias", "validate_by_name",
            ) if key in target.model_config},
        })
        generated.model_rebuild(force=True)
        _check_aliases(generated)
        return generated

    @classmethod
    def analysis_model(cls) -> type[P]:
        if cls._analysis_model is None or (
            cls.__dict__.get("__type_params__") and "_analysis_model" not in cls.__dict__
        ):
            raise TypeError(f"{cls.__name__} 必须绑定具体分析参数类型")
        return cast(type[P], cls._analysis_model)

    def _analysis_values(self) -> dict[str, Any]:
        # 保留嵌套类型及 exclude=True 字段，隔离 validator 对原表单的修改。
        values = deepcopy(dict(self))
        values["universe"] = Universe.model_validate({
            "pool": values.pop("pool"), "lookback": values.pop("lookback"),
        })
        return values

    def build(self) -> P:
        model = self.analysis_model()
        values = {}
        for name, value in self._analysis_values().items():
            key = _input_key(model, name) if name in model.model_fields else name
            if key in values:
                raise ValueError(f"分析参数输入别名冲突：{key!r}")
            values[key] = value
        return model.model_validate(values)


class BacktestForm[P: BaseModel](ResearchForm[P]):
    """四类策略环节共享的行情、资金和费率输入。"""

    market_data: Literal["stock_daily", "stock_snapshot"] = Field(
        default="stock_daily", title="行情类型",
        json_schema_extra={"x-enum-labels": ["日频合成快照", "Tick 快照"]},
    )
    benchmark: str | None = Field(default="000300.SH", title="基准代码")
    batch_days: int = Field(default=1, ge=1, title="行情加载批次天数")
    cash: float = Field(default=1_000_000.0, gt=0, title="初始资金")
    commission: float = Field(default=0.0003, ge=0, title="佣金费率")
    tax: float = Field(default=0.0005, ge=0, title="印花税率")

    def _analysis_values(self) -> dict[str, Any]:
        values = super()._analysis_values()
        field = self.analysis_model().model_fields["config"]
        config = {} if field.is_required() else deepcopy(field.get_default(call_default_factory=True))
        values["config"] = {
            **config, **{name: values.pop(name) for name in ("cash", "commission", "tax")},
        }
        return values
