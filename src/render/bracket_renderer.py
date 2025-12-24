"""
Bracket renderer using LayoutModel.

No inference during render - just draws what's in the LayoutModel.
"""

from pathlib import Path
from typing import Optional
from src.layout.layout_model import LayoutModel
from src.render.figure_builder import FigureBuilder


class BracketRenderer:
    """
    Renders bracket from LayoutModel.
    
    Uses FigureBuilder for step-by-step construction.
    No inference, no winner determination - just rendering.
    """
    
    def __init__(self, logo_dir: Optional[Path] = None):
        """
        Initialize renderer.
        
        Args:
            logo_dir: Directory containing team logos
        """
        self.logo_dir = logo_dir
    
    def render(
        self,
        layout_model: LayoutModel,
        out_path: Path,
        season: int,
        figsize: tuple = (14, 10),
        dpi: int = 300
    ) -> None:
        """
        Render bracket from layout model.
        
        Args:
            layout_model: LayoutModel with all geometry
            out_path: Output file path
            season: Season year (for title)
            figsize: Figure size
            dpi: Output DPI
        """
        builder = FigureBuilder()
        
        builder.create_canvas(figsize=figsize, logo_dir=self.logo_dir)
        builder.draw_background()
        builder.draw_boxes(layout_model)
        builder.draw_connectors(layout_model)
        builder.annotate_winners(layout_model)
        
        # Add title
        if builder.ax:
            builder.ax.text(
                0.5, 0.98,
                f"NFL Playoff Bracket - Season {season}",
                ha="center", va="top",
                fontsize=16, fontweight="bold",
                transform=builder.ax.transAxes,
                zorder=10
            )
        
        builder.export(out_path, dpi=dpi)

