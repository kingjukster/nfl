"""
Deterministic layout computation.

No drawing here - just pure layout geometry computation.
"""

from src.layout.layout_model import LayoutModel, Box, Connector, Label, BoxId
from src.layout.bracket_layout import compute_bracket_layout
from src.layout.grid_calculator import GridCalculator

__all__ = [
    'LayoutModel',
    'Box',
    'Connector',
    'Label',
    'BoxId',
    'compute_bracket_layout',
    'GridCalculator',
]

