"""Tiny CLI override parser for dataclass configs, plus JSON dumping.

Overrides look like `a.b.c=value` (list items by index: `a.items.1.field=value`). Values are coerced against the dataclass field's
type hint, so `sim_dt=0.001`, `render=true`, `scene.tcp_offset=[0,0,0.11]` and
`scene.joint_damping=none` all type-check. Unknown keys raise an error that lists
the valid field names.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any, Union

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_NONE = {"none", "null"}


def parse_cli(argv: list[str]) -> dict[str, str]:
    """Split `key=value` tokens into a dict.

    Args:
        argv: CLI tokens, e.g. `sys.argv[1:]`.

    Returns:
        Ordered dict mapping dotted key -> raw string value.
    """
    out: dict[str, str] = {}
    for tok in argv:
        if "=" not in tok:
            raise ValueError(f"Override '{tok}' is not of the form key=value")
        k, v = tok.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _literal(raw: str) -> Any:
    """Parse a Python/JSON-ish literal, falling back to the raw string.

    Args:
        raw: Raw value string.

    Returns:
        Parsed value (number, list, tuple, dict, ...) or `raw` itself.
    """
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if low in _NONE:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def coerce(raw: Any, tp: Any) -> Any:
    """Coerce a raw value (string or parsed literal) to type hint `tp`.

    Supports bool, int, float, str, None/Optional, Union, Literal, tuple[...], list[...],
    dict[...] and Any. Raises `TypeError`/`ValueError` when coercion is impossible.

    Args:
        raw: Raw string from the CLI, or an already-parsed value (for nested containers).
        tp: Target type hint.

    Returns:
        The coerced value.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return _literal(raw) if isinstance(raw, str) else raw
    if origin in (Union, types.UnionType):
        if isinstance(raw, str) and raw.lower() in _NONE and type(None) in args:
            return None
        errors = []
        # Try non-None members; prefer scalar types before containers for strings like "1.0".
        for a in sorted((a for a in args if a is not type(None)), key=lambda a: typing.get_origin(a) is not None):
            try:
                return coerce(raw, a)
            except (TypeError, ValueError) as e:
                errors.append(f"{a}: {e}")
        raise ValueError(f"cannot coerce {raw!r} to {tp}: {errors}")
    if tp is type(None):
        if raw is None or (isinstance(raw, str) and raw.lower() in _NONE):
            return None
        raise ValueError(f"expected None, got {raw!r}")
    if origin is typing.Literal:
        val = _literal(raw) if isinstance(raw, str) else raw
        if val in args or raw in args:
            return val if val in args else raw
        raise ValueError(f"{raw!r} not in {args}")
    if tp is bool:
        if isinstance(raw, bool):
            return raw
        s = str(raw).lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"expected bool, got {raw!r}")
    if tp is int:
        if isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer()):
            raise ValueError(f"expected int, got {raw!r}")
        return int(raw)
    if tp is float:
        if isinstance(raw, bool):
            raise ValueError(f"expected float, got {raw!r}")
        return float(raw)
    if tp is str:
        return str(raw)
    if origin in (tuple, list):
        val = _literal(raw) if isinstance(raw, str) else raw
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"expected a sequence, got {raw!r}")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(val):
                raise ValueError(f"expected {len(args)} items, got {len(val)}")
            return tuple(coerce(v, a) for v, a in zip(val, args))
        elem = args[0] if args else Any
        items = [coerce(v, elem) for v in val]
        return tuple(items) if origin is tuple else items
    if origin is dict or tp is dict:
        val = _literal(raw) if isinstance(raw, str) else raw
        if not isinstance(val, dict):
            raise ValueError(f"expected a dict, got {raw!r}")
        return val
    if tp in (list, tuple):
        val = _literal(raw) if isinstance(raw, str) else raw
        return tp(val)
    raise TypeError(f"unsupported override type {tp}")


def _field_types(obj: Any) -> dict[str, Any]:
    """Resolve (string) annotations of a dataclass instance.

    Args:
        obj: Dataclass instance.

    Returns:
        Field name -> resolved type hint.
    """
    return typing.get_type_hints(type(obj))


def apply_overrides(cfg: Any, overrides: dict[str, str]) -> Any:
    """Apply dotted overrides in place onto a (nested) dataclass config.

    Intermediate path elements may be dataclass fields, dict keys or list indices (e.g.
    `task.scene.assets.1.inner_radius`). Leaf values are type-checked against the dataclass
    field hint; dict and list leaves are parsed as literals.

    Args:
        cfg: Root dataclass instance (mutated).
        overrides: Mapping from `parse_cli`.

    Returns:
        `cfg` (for chaining).
    """
    for key, raw in overrides.items():
        parts = key.split(".")
        node = cfg
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            path = ".".join(parts[: i + 1])
            if dataclasses.is_dataclass(node):
                names = [f.name for f in dataclasses.fields(node)]
                if p not in names:
                    raise KeyError(f"Unknown config key '{path}'. Valid keys here: {names}")
                if last:
                    try:
                        setattr(node, p, coerce(raw, _field_types(node)[p]))
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"Bad value for '{key}': {e}") from None
                else:
                    node = getattr(node, p)
            elif isinstance(node, dict):
                if last:
                    node[p] = _literal(raw)
                else:
                    if p not in node:
                        raise KeyError(f"Unknown config key '{path}'. Valid keys here: {list(node)}")
                    node = node[p]
            elif isinstance(node, list):
                if not (p.lstrip("-").isdigit() and -len(node) <= int(p) < len(node)):
                    raise KeyError(f"Bad list index '{path}': list has {len(node)} items")
                if last:
                    node[int(p)] = _literal(raw)
                else:
                    node = node[int(p)]
            else:
                raise KeyError(f"Cannot descend into '{path}' (type {type(node).__name__})")
    return cfg


def to_jsonable(obj: Any) -> Any:
    """Recursively convert configs into JSON-serializable structures.

    Dataclasses become dicts tagged with `_type`; numpy arrays become lists; callables
    become their qualified names; anything else unknown becomes `repr`.

    Args:
        obj: Any config object.

    Returns:
        JSON-serializable structure.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {"_type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            d[f.name] = to_jsonable(getattr(obj, f.name))
        return d
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):  # numpy arrays / scalars
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if callable(obj):
        return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"
    return repr(obj)


def dump_json(cfg: Any, path: str | Path) -> None:
    """Write a resolved config to a JSON file.

    Args:
        cfg: Config object.
        path: Output file path (parent directories are created).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(cfg), indent=2))
