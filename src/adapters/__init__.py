"""
Input format adapters.

Protect renderer from simulation changes by converting various input formats
to canonical models.
"""

from src.adapters.simulation_adapter import SimulationOutputAdapter
from src.adapters.stats_adapter import StatsAdapter

__all__ = [
    'SimulationOutputAdapter',
    'StatsAdapter',
]

