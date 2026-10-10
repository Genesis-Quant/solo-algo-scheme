"""UI 表单只负责把简化输入转换成具体分析参数。"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from copy import deepcopy
from datetime import date, timedelta
from functools import cache
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypeAliasType,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

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


def _check_aliases(model: type[BaseModel], passed: Collection[str] | None = None) -> None:
    """平铺输入键必须唯一；只有表单实际传入的字段需要平铺键。"""
    owners: dict[str, str] = {}
    for name, field in model.model_fields.items():
        if passed is None or name in passed:
            _input_key(model, name)
        aliases = field.validation_alias
        choices = aliases.choices if isinstance(aliases, AliasChoices) else [aliases]
        keys = set()
        if model.model_config.get("validate_by_alias") is not False:
            # AliasPath 读取另一个输入的嵌套值，沿用分析参数自身语义，不占用平铺键。
            keys.update(alias for alias in choices if isinstance(alias, str))
        if not keys or model.model_config.get("validate_by_name"):
            keys.add(name)
        for key in keys:
            if key in owners and owners[key] != name:
                raise TypeError(
                    f"Form 字段的 validation alias 必须唯一：{name!r} 与 {owners[key]!r}"
                    f" 共用 {key!r}"
                )
            owners[key] = name


_VALIDATORS = {
    "before": BeforeValidator, "after": AfterValidator,
    "plain": PlainValidator, "wrap": WrapValidator,
}


def _converts(annotation: Any, metadata: Sequence[Any] = (), seen: frozenset[int] = frozenset()) -> bool:
    """Annotated 转换器和嵌套模型都会在最终 Analysis 中再执行一次。"""
    if any(isinstance(item, tuple(_VALIDATORS.values())) for item in metadata):
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    if id(annotation) in seen:
        return False
    seen |= {id(annotation)}
    if isinstance(annotation, TypeAliasType):
        return _converts(annotation.__value__, (), seen)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _converts(args[0], args[1:], seen)
    if isinstance(origin, TypeAliasType) and _converts(origin, (), seen):
        return True
    return any(_converts(arg, (), seen) for arg in args)


def _input_field(model: type[BaseModel], name: str, *, platform: bool):
    field = model.model_fields[name]
    copied = deepcopy(field)
    decorators = [
        decorator for decorator in model.__pydantic_decorators__.field_validators.values()
        if name in decorator.info.fields or "*" in decorator.info.fields
    ]
    if platform and not decorators and not _converts(field.annotation, field.metadata):
        # 只改默认值、标题或约束的平台字段仍由表单校验，交给下游的公共值保持规范。
        return copied
    for decorator in decorators:
        info = decorator.info
        options = {} if info.mode == "after" else {
            "json_schema_input_type": info.json_schema_input_type,
        }
        copied.metadata.append(_VALIDATORS[info.mode](decorator.func, **options))
    # 保留约束与输入 schema；业务转换只交给最终 Analysis，不能在 Form 先转换一次。
    copied.metadata.append(SkipValidation())
    return copied


class ResearchForm[P: BaseModel](ReportForm[P]):
    """公共研究输入；具体类型绑定后展开项目声明的分析字段。"""

    _analysis_base: ClassVar[type[BaseModel] | None] = None
    _analysis_model: ClassVar[type[BaseModel] | None] = None
    _internal_fields: ClassVar[frozenset[str]] = frozenset({"universe", "config", "symbols"})
    _supplied_fields: ClassVar[frozenset[str]] = frozenset({"universe"})
    _stage_fields: ClassVar[frozenset[str]] = frozenset()

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

        base = cls._analysis_base
        _check_aliases(target, target.model_fields.keys() - (cls._internal_fields - cls._supplied_fields))
        fields = {}
        for name, field in target.model_fields.items():
            if field.default_factory_takes_validated_data:
                raise TypeError(
                    f"自动 Form 字段 {name!r} 不支持依赖其他字段的 default_factory；"
                    "请使用无参默认工厂，或在分析参数的 model_validator 中推导"
                )
            inherited = _field_owner(target, name) is _field_owner(base, name)
            if name in cls._stage_fields and not (name in base.model_fields and inherited):
                raise TypeError(f"{name!r} 是 Scheme 管理的上下游选择字段，项目参数不能声明同名字段")
            if name not in base.model_fields and (
                name in cls.model_fields or name in cls._internal_fields or hasattr(cls, name)
            ):
                raise TypeError(f"分析字段 {name!r} 与 Scheme 表单保留字段冲突")
            if name in cls._internal_fields:
                if name not in cls._supplied_fields and field.is_required():
                    raise TypeError(f"分析字段 {name!r} 由 Scheme 管理且不在表单中填写，必须提供默认值")
                continue
            if name not in cls.model_fields or not inherited:
                fields[name] = (
                    field.annotation, _input_field(target, name, platform=name in cls.model_fields),
                )

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

    _supplied_fields: ClassVar[frozenset[str]] = frozenset({"universe", "config"})
    _stage_fields: ClassVar[frozenset[str]] = frozenset({"model", "optimize", "control", "execution"})

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
