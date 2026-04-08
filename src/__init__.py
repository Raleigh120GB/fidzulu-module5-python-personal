"""
Compatibility package to allow imports using the legacy `src.*` namespace
Tests and some modules import `src.fidzulu.*`. Creating this empty package
makes `src` a valid top-level package that exposes the `fidzulu` subpackage
already present under `src/fidzulu`.
"""

__all__ = []
