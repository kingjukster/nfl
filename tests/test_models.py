"""
Model tests.

Tests for canonical data models.
"""

import pytest
from src.models.bracket import Team, Matchup, FrozenBracket
from src.models.seeding import PlayoffSeed, ConferenceSeeding


class TestTeam:
    """Tests for Team model."""
    
    def test_team_creation(self):
        """Test creating a valid team."""
        team = Team('KC', 1, 'AFC')
        assert team.name == 'KC'
        assert team.seed == 1
        assert team.conf == 'AFC'
    
    def test_team_invalid_conference(self):
        """Test team creation with invalid conference."""
        with pytest.raises(ValueError, match="Invalid conference"):
            Team('KC', 1, 'INVALID')
    
    def test_team_invalid_seed(self):
        """Test team creation with invalid seed."""
        with pytest.raises(ValueError, match="Invalid seed"):
            Team('KC', 8, 'AFC')


class TestMatchup:
    """Tests for Matchup model."""
    
    def test_matchup_creation(self):
        """Test creating a valid matchup."""
        team1 = Team('KC', 1, 'AFC')
        team2 = Team('BUF', 2, 'AFC')
        matchup = Matchup(team1, team2, team1)
        
        assert matchup.a == team1
        assert matchup.b == team2
        assert matchup.winner == team1
    
    def test_matchup_invalid_winner(self):
        """Test matchup with winner not in teams."""
        team1 = Team('KC', 1, 'AFC')
        team2 = Team('BUF', 2, 'AFC')
        invalid_winner = Team('SF', 1, 'NFC')
        
        with pytest.raises(ValueError, match="Winner.*must be either"):
            Matchup(team1, team2, invalid_winner)


class TestFrozenBracket:
    """Tests for FrozenBracket model."""
    
    def test_frozen_bracket_creation(self):
        """Test creating a valid frozen bracket."""
        team1 = Team('KC', 1, 'AFC')
        team2 = Team('BUF', 2, 'AFC')
        matchup = Matchup(team1, team2, team1)
        
        frozen = FrozenBracket(
            wc=[matchup, matchup, matchup],
            div=[matchup, matchup],
            conf=matchup,
            champ=team1
        )
        
        assert len(frozen.wc) == 3
        assert len(frozen.div) == 2
        assert frozen.champ == team1
    
    def test_frozen_bracket_invalid_wc_count(self):
        """Test frozen bracket with wrong WC count."""
        team1 = Team('KC', 1, 'AFC')
        team2 = Team('BUF', 2, 'AFC')
        matchup = Matchup(team1, team2, team1)
        
        with pytest.raises(ValueError, match="Expected 3 Wild Card matchups"):
            FrozenBracket(
                wc=[matchup, matchup],  # Only 2
                div=[matchup, matchup],
                conf=matchup,
                champ=team1
            )

