from importlib import import_module


def resolve_export(name, exports, namespace):
    try:
        module, attribute = exports[name]
    except KeyError:
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), attribute)
    namespace[name] = value
    return value
