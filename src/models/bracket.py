"""
Canonical bracket data models.

These are immutable dataclasses representing the playoff bracket structure.
No rendering logic, no coordinates - pure data truth.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Team:
    """Represents a team in the playoff bracket."""
    name: str
    seed: int
    conf: str  # "AFC" or "NFC"
    
    def __post_init__(self):
        """Validate team data."""
        if self.conf not in ["AFC", "NFC"]:
            raise ValueError(f"Invalid conference: {self.conf}")
        if not (1 <= self.seed <= 7):
            raise ValueError(f"Invalid seed: {self.seed} (must be 1-7)")


@dataclass(frozen=True)
class Matchup:
    """Represents a playoff matchup between two teams."""
    a: Team
    b: Team
    winner: Team
    
    def __post_init__(self):
        """Validate matchup data."""
        if self.a.conf != self.b.conf:
            raise ValueError(f"Teams must be in same conference: {self.a.conf} vs {self.b.conf}")
        if self.winner.name not in [self.a.name, self.b.name]:
            raise ValueError(f"Winner {self.winner.name} must be either {self.a.name} or {self.b.name}")


@dataclass(frozen=True)
class FrozenBracket:
    """
    Frozen bracket structure for a single conference.
    
    This represents the complete bracket state after all matchups are determined.
    No rendering, no coordinates - just the data structure.
    """
    wc: List[Matchup]   # Wild Card round: 3 matchups
    div: List[Matchup]  # Divisional round: 2 matchups
    conf: Matchup       # Conference championship: 1 matchup
    champ: Team         # Conference champion (same as conf.winner)
    
    def __post_init__(self):
        """Validate bracket structure."""
        if len(self.wc) != 3:
            raise ValueError(f"Expected 3 Wild Card matchups, got {len(self.wc)}")
        if len(self.div) != 2:
            raise ValueError(f"Expected 2 Divisional matchups, got {len(self.div)}")
        if self.champ.name != self.conf.winner.name:
            raise ValueError(f"Champion {self.champ.name} must match conference winner {self.conf.winner.name}")

