"""
Canonical data models representing "data truth".

These models contain no rendering logic, no matplotlib, no x/y coordinates.
Pure data structures that represent the state of the playoff bracket.
"""

from src.models.bracket import Team, Matchup, FrozenBracket
from src.models.seeding import PlayoffSeed, ConferenceSeeding
from src.models.qb_metrics import QBComparisonDataset

__all__ = [
    'Team',
    'Matchup',
    'FrozenBracket',
    'PlayoffSeed',
    'ConferenceSeeding',
    'QBComparisonDataset',
]

