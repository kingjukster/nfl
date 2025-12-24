"""
Canonical QB metrics data models.

Represents normalized QB comparison data for visualization.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd


@dataclass
class QBComparisonDataset:
    """
    Normalized QB metrics dataset ready for visualization.
    
    All metrics should be normalized to 0-1 scale for fair comparison.
    """
    data: pd.DataFrame
    metrics: List[str]  # List of normalized metric column names
    season: int
    
    def __post_init__(self):
        """Validate QB dataset."""
        if self.data.empty:
            raise ValueError("QB dataset cannot be empty")
        
        required_cols = ['player_name', 'team']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check all metrics exist
        for metric in self.metrics:
            if metric not in self.data.columns:
                raise ValueError(f"Metric {metric} not found in dataset")
        
        # Check metrics are normalized (0-1 range)
        for metric in self.metrics:
            if self.data[metric].min() < 0 or self.data[metric].max() > 1:
                raise ValueError(f"Metric {metric} not normalized (should be 0-1 range)")
    
    @property
    def n_qbs(self) -> int:
        """Number of QBs in dataset."""
        return len(self.data)
    
    def get_team_color(self, team: str) -> str:
        """Get team color for a QB (delegates to constants)."""
        from src.constants import get_team_color
        return get_team_color(team)

