"""
Canonical seeding data models.

Represents playoff seeding information for teams.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PlayoffSeed:
    """Represents a playoff seed for a team."""
    team: str
    conference: str
    seed: int
    division: str
    wins: float
    losses: float
    win_pct: float
    points_for: float = 0.0
    points_against: float = 0.0
    
    def __post_init__(self):
        """Validate seed data."""
        if self.conference not in ["AFC", "NFC"]:
            raise ValueError(f"Invalid conference: {self.conference}")
        if not (1 <= self.seed <= 7):
            raise ValueError(f"Invalid seed: {self.seed} (must be 1-7)")


@dataclass(frozen=True)
class ConferenceSeeding:
    """Represents seeding for an entire conference."""
    conference: str
    seeds: List[PlayoffSeed]
    
    def __post_init__(self):
        """Validate conference seeding."""
        if self.conference not in ["AFC", "NFC"]:
            raise ValueError(f"Invalid conference: {self.conference}")
        if len(self.seeds) != 7:
            raise ValueError(f"Expected 7 seeds, got {len(self.seeds)}")
        
        # Check for unique seeds
        seed_numbers = [s.seed for s in self.seeds]
        if len(set(seed_numbers)) != 7:
            raise ValueError(f"Duplicate seeds found: {seed_numbers}")
        
        # Check for unique teams
        team_names = [s.team for s in self.seeds]
        if len(set(team_names)) != 7:
            raise ValueError(f"Duplicate teams found: {team_names}")
        
        # Check all teams are in same conference
        for seed in self.seeds:
            if seed.conference != self.conference:
                raise ValueError(f"Team {seed.team} in wrong conference: {seed.conference} != {self.conference}")

