"""
Rendering module (Builder Pattern).

Step-by-step figure construction with no inference during render.
"""

from src.render.figure_builder import FigureBuilder
from src.render.bracket_renderer import BracketRenderer
from src.render.connectors import draw_connector, draw_elbow_arrow

__all__ = [
    'FigureBuilder',
    'BracketRenderer',
    'draw_connector',
    'draw_elbow_arrow',
]

