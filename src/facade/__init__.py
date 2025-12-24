"""
Clean public API (Facade Pattern).

Simple, high-level functions for common visualization tasks.
"""

from src.facade.bracket_api import render_bracket
from src.facade.qb_api import render_qb_comparison

__all__ = [
    'render_bracket',
    'render_qb_comparison',
]

