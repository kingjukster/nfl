"""
Base chart strategy interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import matplotlib.figure
from src.layout.layout_model import LayoutModel
from src.models.bracket import FrozenBracket


class ChartStrategy(ABC):
    """
    Strategy interface for chart creation.
    
    Different strategies answer different questions:
    - "Who is best per metric?" → GroupedBarStrategy
    - "What is each QB's profile?" → RadarStrategy
    - "Matchup comparison?" → RadarStrategy (2-3 QBs)
    """
    
    @abstractmethod
    def make_figure(
        self,
        data: Any,
        theme: str = 'default',
        options: Dict[str, Any] = None
    ) -> matplotlib.figure.Figure:
        """
        Create a figure from data.
        
        Args:
            data: Input data (format depends on strategy)
            theme: Theme name for styling
            options: Strategy-specific options
        
        Returns:
            matplotlib Figure ready for rendering
        """
        pass


class LayoutStrategy(ABC):
    """
    Strategy interface for bracket layout computation.
    
    Different strategies can produce different bracket layouts:
    - Standard layout (default)
    - Compact layout
    - Top-finals layout
    """
    
    @abstractmethod
    def compute_layout(
        self,
        frozen_bracket: FrozenBracket,
        options: Dict[str, Any] = None
    ) -> LayoutModel:
        """
        Compute layout model from frozen bracket.
        
        Args:
            frozen_bracket: FrozenBracket to layout
            options: Layout-specific options
        
        Returns:
            LayoutModel with computed geometry
        """
        pass

