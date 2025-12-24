"""
Validation tests.

Tests for all validators to ensure they catch errors early.
"""

import pytest
import pandas as pd
from src.models.bracket import Team, Matchup, FrozenBracket
from src.validation import (
    validate_bracket_dict,
    validate_frozen_bracket,
    validate_seeding,
    validate_qb_metrics
)


class TestBracketValidator:
    """Tests for bracket validation."""
    
    def test_validate_bracket_dict_valid(self):
        """Test validation of valid bracket dict."""
        bracket_dict = {
            'AFC': {
                'WC': [
                    {'a_name': 'KC', 'a_seed': 2, 'b_name': 'BUF', 'b_seed': 7, 'winner_name': 'KC', 'winner_seed': 2},
                    {'a_name': 'BAL', 'a_seed': 3, 'b_name': 'HOU', 'b_seed': 6, 'winner_name': 'BAL', 'winner_seed': 3},
                    {'a_name': 'MIA', 'a_seed': 4, 'b_name': 'PIT', 'b_seed': 5, 'winner_name': 'MIA', 'winner_seed': 4},
                ],
                'DIV': [
                    {'a_name': 'DET', 'a_seed': 1, 'b_name': 'MIA', 'b_seed': 4, 'winner_name': 'DET', 'winner_seed': 1},
                    {'a_name': 'KC', 'a_seed': 2, 'b_name': 'BAL', 'b_seed': 3, 'winner_name': 'KC', 'winner_seed': 2},
                ],
                'CONF': {
                    'a_name': 'DET', 'a_seed': 1, 'b_name': 'KC', 'b_seed': 2,
                    'winner_name': 'KC', 'winner_seed': 2
                }
            },
            'NFC': {
                'WC': [
                    {'a_name': 'SF', 'a_seed': 2, 'b_name': 'GB', 'b_seed': 7, 'winner_name': 'SF', 'winner_seed': 2},
                    {'a_name': 'DAL', 'a_seed': 3, 'b_name': 'LAR', 'b_seed': 6, 'winner_name': 'DAL', 'winner_seed': 3},
                    {'a_name': 'TB', 'a_seed': 4, 'b_name': 'PHI', 'b_seed': 5, 'winner_name': 'TB', 'winner_seed': 4},
                ],
                'DIV': [
                    {'a_name': 'DET', 'a_seed': 1, 'b_name': 'TB', 'b_seed': 4, 'winner_name': 'DET', 'winner_seed': 1},
                    {'a_name': 'SF', 'a_seed': 2, 'b_name': 'DAL', 'b_seed': 3, 'winner_name': 'SF', 'winner_seed': 2},
                ],
                'CONF': {
                    'a_name': 'DET', 'a_seed': 1, 'b_name': 'SF', 'b_seed': 2,
                    'winner_name': 'SF', 'winner_seed': 2
                }
            }
        }
        
        # Should not raise
        validate_bracket_dict(bracket_dict)
    
    def test_validate_bracket_dict_invalid_winner(self):
        """Test validation catches invalid winner."""
        bracket_dict = {
            'AFC': {
                'WC': [
                    {'a_name': 'KC', 'a_seed': 2, 'b_name': 'BUF', 'b_seed': 7, 'winner_name': 'INVALID', 'winner_seed': 2},
                ] * 3,
                'DIV': [{'a_name': 'KC', 'a_seed': 1, 'b_name': 'BUF', 'b_seed': 2, 'winner_name': 'KC', 'winner_seed': 1}] * 2,
                'CONF': {'a_name': 'KC', 'a_seed': 1, 'b_name': 'BUF', 'b_seed': 2, 'winner_name': 'KC', 'winner_seed': 1}
            },
            'NFC': {
                'WC': [{'a_name': 'SF', 'a_seed': 2, 'b_name': 'GB', 'b_seed': 7, 'winner_name': 'SF', 'winner_seed': 2}] * 3,
                'DIV': [{'a_name': 'SF', 'a_seed': 1, 'b_name': 'GB', 'b_seed': 2, 'winner_name': 'SF', 'winner_seed': 1}] * 2,
                'CONF': {'a_name': 'SF', 'a_seed': 1, 'b_name': 'GB', 'b_seed': 2, 'winner_name': 'SF', 'winner_seed': 1}
            }
        }
        
        with pytest.raises(ValueError, match="Winner.*must be either"):
            validate_bracket_dict(bracket_dict)
    
    def test_validate_frozen_bracket_valid(self):
        """Test validation of valid frozen bracket."""
        team1 = Team('KC', 1, 'AFC')
        team2 = Team('BUF', 2, 'AFC')
        matchup = Matchup(team1, team2, team1)
        
        frozen = FrozenBracket(
            wc=[matchup, matchup, matchup],
            div=[matchup, matchup],
            conf=matchup,
            champ=team1
        )
        
        # Should not raise
        validate_frozen_bracket(frozen)


class TestSeedingValidator:
    """Tests for seeding validation."""
    
    def test_validate_seeding_valid(self):
        """Test validation of valid seeding."""
        seeding = {
            'AFC': [
                {'team': 'KC', 'seed': 1, 'win_pct': 0.8},
                {'team': 'BUF', 'seed': 2, 'win_pct': 0.75},
                {'team': 'BAL', 'seed': 3, 'win_pct': 0.7},
                {'team': 'HOU', 'seed': 4, 'win_pct': 0.65},
                {'team': 'MIA', 'seed': 5, 'win_pct': 0.6},
                {'team': 'PIT', 'seed': 6, 'win_pct': 0.55},
                {'team': 'CLE', 'seed': 7, 'win_pct': 0.5},
            ],
            'NFC': [
                {'team': 'SF', 'seed': 1, 'win_pct': 0.8},
                {'team': 'DAL', 'seed': 2, 'win_pct': 0.75},
                {'team': 'DET', 'seed': 3, 'win_pct': 0.7},
                {'team': 'TB', 'seed': 4, 'win_pct': 0.65},
                {'team': 'PHI', 'seed': 5, 'win_pct': 0.6},
                {'team': 'LAR', 'seed': 6, 'win_pct': 0.55},
                {'team': 'GB', 'seed': 7, 'win_pct': 0.5},
            ]
        }
        
        # Should not raise
        validate_seeding(seeding)
    
    def test_validate_seeding_duplicate_seeds(self):
        """Test validation catches duplicate seeds."""
        seeding = {
            'AFC': [
                {'team': 'KC', 'seed': 1, 'win_pct': 0.8},
                {'team': 'BUF', 'seed': 1, 'win_pct': 0.75},  # Duplicate seed
            ] + [{'team': f'Team{i}', 'seed': i, 'win_pct': 0.5} for i in range(3, 8)],
            'NFC': [{'team': f'Team{i}', 'seed': i, 'win_pct': 0.5} for i in range(1, 8)]
        }
        
        with pytest.raises(ValueError, match="Duplicate seeds"):
            validate_seeding(seeding)


class TestQBValidator:
    """Tests for QB metrics validation."""
    
    def test_validate_qb_metrics_valid(self):
        """Test validation of valid QB metrics."""
        df = pd.DataFrame({
            'player_name': ['QB1', 'QB2'],
            'team': ['KC', 'BUF'],
            'completion_pct_norm': [0.8, 0.75],
            'epa_per_play_norm': [0.9, 0.85],
            'win_rate_norm': [1.0, 0.8]
        })
        
        # Should not raise
        validate_qb_metrics(df)
    
    def test_validate_qb_metrics_out_of_range(self):
        """Test validation catches out-of-range metrics."""
        df = pd.DataFrame({
            'player_name': ['QB1'],
            'team': ['KC'],
            'completion_pct_norm': [1.5],  # Out of range
        })
        
        with pytest.raises(ValueError, match="out of range"):
            validate_qb_metrics(df, required_metrics=['completion_pct_norm'])

