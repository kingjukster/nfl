"""
Layout model data structures.

Stores all computed geometry for rendering.
No matplotlib, no drawing - just data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from typing_extensions import NewType

# Type aliases for clarity
BoxId = NewType('BoxId', str)


@dataclass(frozen=True)
class Box:
    """
    Represents a team box in the layout.
    
    All coordinates are in normalized 0-1 space.
    """
    id: BoxId
    x_left: float  # Left edge x position (0-1)
    y_center: float  # Center y position (0-1)
    width: float
    height: float
    team_name: str
    seed: int
    conference: str
    is_winner: bool = False
    
    @property
    def x_right(self) -> float:
        """Right edge x position."""
        return self.x_left + self.width
    
    @property
    def y_top(self) -> float:
        """Top edge y position."""
        return self.y_center + self.height / 2
    
    @property
    def y_bottom(self) -> float:
        """Bottom edge y position."""
        return self.y_center - self.height / 2
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (x, y)."""
        return (self.x_left + self.width / 2, self.y_center)


@dataclass(frozen=True)
class Connector:
    """
    Represents a connector between two boxes.
    
    Uses explicit endpoints (not inferred during render).
    """
    src_id: BoxId
    dst_id: BoxId
    style: str  # 'elbow_h', 'elbow_v', 'straight'
    color: str = '#666666'
    linewidth: float = 1.5
    zorder: int = 1


@dataclass(frozen=True)
class Label:
    """
    Represents a text label/annotation in the layout.
    """
    x: float
    y: float
    text: str
    fontsize: float = 10.0
    fontweight: str = 'normal'
    color: str = '#000000'
    ha: str = 'center'  # horizontal alignment
    va: str = 'center'  # vertical alignment
    zorder: int = 5


@dataclass
class LayoutModel:
    """
    Complete layout model storing all computed geometry.
    
    This is the "ready-to-plot" structure that renderers consume.
    """
    boxes: Dict[BoxId, Box]
    connectors: List[Connector]
    annotations: List[Label]
    
    def __post_init__(self):
        """Validate layout model."""
        # Check all connector endpoints exist
        box_ids = set(self.boxes.keys())
        for connector in self.connectors:
            if connector.src_id not in box_ids:
                raise ValueError(f"Connector source {connector.src_id} not found in boxes")
            if connector.dst_id not in box_ids:
                raise ValueError(f"Connector destination {connector.dst_id} not found in boxes")
    
    def get_box(self, box_id: BoxId) -> Optional[Box]:
        """Get a box by ID."""
        return self.boxes.get(box_id)
    
    def get_connectors_from(self, box_id: BoxId) -> List[Connector]:
        """Get all connectors originating from a box."""
        return [c for c in self.connectors if c.src_id == box_id]
    
    def get_connectors_to(self, box_id: BoxId) -> List[Connector]:
        """Get all connectors ending at a box."""
        return [c for c in self.connectors if c.dst_id == box_id]

