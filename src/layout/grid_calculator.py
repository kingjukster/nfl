"""
Deterministic grid position calculator.

Computes fixed grid positions for bracket layout.
"""

from typing import Tuple, List
from dataclasses import dataclass


@dataclass(frozen=True)
class GridCalculator:
    """
    Calculates deterministic grid positions for bracket layout.
    
    All positions are in normalized 0-1 coordinate space.
    """
    figure_width: float = 14.0  # inches
    figure_height: float = 10.0  # inches
    
    # Fixed column positions (normalized 0-1)
    AFC_WC_X: float = 0.15
    AFC_DIV_X: float = 0.30
    AFC_CONF_X: float = 0.45
    
    NFC_WC_X: float = 0.85
    NFC_DIV_X: float = 0.70
    NFC_CONF_X: float = 0.55
    
    # Super Bowl positions
    SB_AFC_X: float = 0.45
    SB_NFC_X: float = 0.55
    SB_Y: float = 0.10
    
    # Box dimensions (normalized)
    BOX_WIDTH: float = 0.18
    BOX_HEIGHT: float = 0.045
    
    # Wild Card row positions (normalized 0-1, top to bottom)
    WC_ROWS: List[float] = None
    
    # Divisional row positions
    DIV_ROWS: List[float] = None
    
    # Conference row positions
    CONF_ROWS: List[float] = None
    
    def __post_init__(self):
        """Initialize default row positions if not provided."""
        if self.WC_ROWS is None:
            object.__setattr__(self, 'WC_ROWS', [0.90, 0.80, 0.65, 0.55, 0.40, 0.30])
        if self.DIV_ROWS is None:
            object.__setattr__(self, 'DIV_ROWS', [0.75, 0.65, 0.50, 0.40])
        if self.CONF_ROWS is None:
            object.__setattr__(self, 'CONF_ROWS', [0.60, 0.50])
    
    def get_wc_position(self, conference: str, slot: int, position: str) -> Tuple[float, float]:
        """
        Get position for Wild Card round.
        
        Args:
            conference: 'AFC' or 'NFC'
            slot: Slot index (0-2 for 3 matchups)
            position: 'top' or 'bottom' for the two teams in a matchup
        
        Returns:
            (x_left, y_center) tuple in normalized coordinates
        """
        if conference == 'AFC':
            x = self.AFC_WC_X
        else:
            x = self.NFC_WC_X
        
        # Each slot has 2 positions (top and bottom)
        row_idx = slot * 2 + (0 if position == 'top' else 1)
        y = self.WC_ROWS[row_idx]
        
        return (x, y)
    
    def get_div_position(self, conference: str, slot: int, position: str) -> Tuple[float, float]:
        """Get position for Divisional round."""
        if conference == 'AFC':
            x = self.AFC_DIV_X
        else:
            x = self.NFC_DIV_X
        
        row_idx = slot * 2 + (0 if position == 'top' else 1)
        y = self.DIV_ROWS[row_idx]
        
        return (x, y)
    
    def get_conf_position(self, conference: str, position: str) -> Tuple[float, float]:
        """Get position for Conference Championship."""
        if conference == 'AFC':
            x = self.AFC_CONF_X
        else:
            x = self.NFC_CONF_X
        
        y = self.CONF_ROWS[0] if position == 'top' else self.CONF_ROWS[1]
        
        return (x, y)
    
    def get_sb_position(self, conference: str) -> Tuple[float, float]:
        """Get position for Super Bowl."""
        x = self.SB_AFC_X if conference == 'AFC' else self.SB_NFC_X
        return (x, self.SB_Y)

