"""Dynamic component factory.

Usage:
    from small_agent.registry import build

    tool = build("tools.web_fetch.WebFetchTool", {"max_chars": 5000})
    backend = build("backends.lmstudio.LMStudioBackend", {"base_url": "...", "model": "..."})
"""

from __future__ import annotations

import importlib
from typing import Any


_PACKAGE = "small_agent"


def build(type_path: str, config: dict[str, Any]) -> Any:
    """Instantiate a component by dotted class path relative to the ``small_agent`` package.

    Args:
        type_path: Dotted path such as ``"tools.web_fetch.WebFetchTool"``.
                   A fully-qualified path (containing the package name) is also accepted.
        config:    Keyword arguments forwarded to the class constructor.

    Returns:
        An instantiated component.
    """
    if not type_path.startswith(_PACKAGE + "."):
        type_path = f"{_PACKAGE}.{type_path}"

    module_path, class_name = type_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not import module '{module_path}'. "
            f"Check that the type path '{type_path}' is correct."
        ) from exc

    if not hasattr(module, class_name):
        available = [n for n in dir(module) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in '{module_path}'. "
            f"Available names: {available}"
        )

    cls = getattr(module, class_name)
    return cls(**config)
