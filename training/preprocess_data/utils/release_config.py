"""Resolve online job templates without importing data or model runtimes."""

from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any

import yaml

_VARIABLE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z_0-9]*)\}")


def merge_mapping(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_mapping(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_includes(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Cyclic template include: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Template must be a mapping: {path}")
    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged = {}
    for include in includes:
        merged = merge_mapping(merged, _read_includes(path.parent / include, (*stack, path)))
    return merge_mapping(merged, raw)


def expand_template(path: str | Path) -> dict:
    """Resolve includes and typed variables for one ingestion/action/value job."""
    raw = _read_includes(Path(path))
    if raw.get("template_version") != 1:
        raise ValueError("Expected template_version: 1")
    variables = raw.get("variables", {})
    resolved = {}

    def variable(name, stack):
        if name in stack:
            raise ValueError(f"Cyclic variable: {' -> '.join((*stack, name))}")
        if name not in variables:
            raise ValueError(f"Undefined template variable: {name}")
        if name not in resolved:
            resolved[name] = expand(variables[name], (*stack, name))
        return resolved[name]

    def expand(value: Any, stack=()):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                expanded_key = expand(key, stack)
                if expanded_key in result:
                    raise ValueError(f"Duplicate key after variable expansion: {expanded_key}")
                result[expanded_key] = expand(item, stack)
            return result
        if isinstance(value, list):
            return [expand(item, stack) for item in value]
        if not isinstance(value, str):
            return value
        match = _VARIABLE.fullmatch(value)
        if match:
            return copy.deepcopy(variable(match[1], stack))
        return _VARIABLE.sub(lambda match: str(variable(match[1], stack)), value)

    document = expand({key: value for key, value in raw.items() if key != "variables"})
    if not isinstance(document.get("job"), dict):
        raise ValueError("Expected a job mapping")
    return document
