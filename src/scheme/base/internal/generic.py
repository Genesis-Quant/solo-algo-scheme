"""Resolve bound arguments across typing and Pydantic generic inheritance."""

from functools import reduce
from operator import or_
from types import UnionType
from typing import Annotated, Any, TypeVar, get_args, get_origin


def generic_parameters(cls: type) -> tuple[Any, ...]:
    metadata = getattr(cls, "__pydantic_generic_metadata__", {})
    return (
        metadata.get("parameters", ())
        or getattr(cls, "__type_params__", ())
        or getattr(cls, "__parameters__", ())
    )


def _parts(value: Any) -> tuple[Any, tuple[Any, ...]]:
    metadata = getattr(value, "__pydantic_generic_metadata__", {})
    if metadata.get("origin") is not None:
        return metadata["origin"], metadata["args"]
    return get_origin(value) or value, get_args(value)


def _substitute(value: Any, bindings: dict[Any, Any], seen: frozenset[Any] = frozenset()) -> Any:
    if isinstance(value, TypeVar):
        replacement = bindings.get(value, value)
        if replacement is value or value in seen:
            return value
        return _substitute(replacement, bindings, seen | {value})

    metadata = getattr(value, "__pydantic_generic_metadata__", {})
    if metadata.get("origin") is not None or metadata.get("parameters"):
        origin = metadata.get("origin") or value
        old = metadata["args"] if metadata.get("origin") is not None else metadata["parameters"]
        args = tuple(_substitute(arg, bindings, seen) for arg in old)
        return value if args == old else origin[args[0] if len(args) == 1 else args]

    origin, old = get_origin(value), get_args(value)
    if not old:
        return value
    if origin is Annotated:
        annotation = _substitute(old[0], bindings, seen)
        return value if annotation == old[0] else Annotated[annotation, *old[1:]]
    args = tuple(_substitute(arg, bindings, seen) for arg in old)
    if args == old:
        return value
    if origin is UnionType:
        return reduce(or_, args)
    if hasattr(value, "copy_with"):
        return value.copy_with(args)
    return origin[args[0] if len(args) == 1 else args]


def generic_arguments(cls: type, target: type | tuple[type, ...]) -> tuple[Any, ...]:
    targets = target if isinstance(target, tuple) else (target,)
    origin, args = _parts(cls)
    if origin in targets and args:
        return args

    def walk(current: type, bindings: dict[Any, Any]) -> tuple[Any, ...] | None:
        for base in current.__dict__.get("__orig_bases__", current.__bases__):
            origin, original_args = _parts(base)
            args = tuple(_substitute(arg, bindings) for arg in original_args)
            if origin in targets:
                if args:
                    return args
                continue
            mapping = {**bindings, **dict(zip(generic_parameters(origin), args))}
            if origin is not object:
                found = walk(origin, mapping)
                if found is not None:
                    return found
        return None

    found = walk(cls, {})
    if found is None:
        raise TypeError(f"{cls.__name__} 未继承指定的泛型基类")
    return found
