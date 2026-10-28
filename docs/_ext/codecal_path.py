"""Sphinx extension: make `codecal` importable for autodoc when the book is
built from the repository root without installing the package."""
import os
import sys


def setup(app):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return {'version': '1.0', 'parallel_read_safe': True}
