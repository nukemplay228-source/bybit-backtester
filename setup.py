"""Compatibility shim for older pip versions.

Editable installs from ``pyproject.toml`` (PEP 660) require pip 21.3 or
newer. This minimal ``setup.py`` lets ``pip install -e .`` work on older
pip releases shipped with Python 3.10/3.11 on Windows by giving them a
legacy entry point. All real metadata still lives in ``pyproject.toml``.
"""

from setuptools import setup

setup()
