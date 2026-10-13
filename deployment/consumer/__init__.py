"""Device consumers, imported on demand so recording does not load models."""

from importlib import import_module

_MODULES = {
    "BaseConsumer": "base",
    "PolicyConnector": "policy_connector",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
