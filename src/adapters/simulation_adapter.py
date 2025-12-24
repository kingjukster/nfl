"""
Adapter to convert simulation output to canonical bracket models.
"""

from typing import Dict, Any, List
from src.models.bracket import FrozenBracket, Matchup, Team
from src.validation import validate_bracket_dict


class SimulationOutputAdapter:
    """
    Converts playoff_predictor.simulate_full_playoffs() output to FrozenBracket.
    
    Protects renderer from simulation format changes.
    """
    
    @staticmethod
    def to_frozen_bracket(simulation_output: Dict[str, Any], conference: str) -> FrozenBracket:
        """
        Convert simulation output to FrozenBracket for a conference.
        
        Args:
            simulation_output: Output from simulate_full_playoffs()
            conference: 'AFC' or 'NFC'
        
        Returns:
            FrozenBracket for the specified conference
        """
        if conference not in ['AFC', 'NFC']:
            raise ValueError(f"Invalid conference: {conference}")
        
        # Get bracket dict from simulation output
        bracket_dict = SimulationOutputAdapter._extract_bracket_dict(simulation_output)
        
        # Validate bracket dict
        validate_bracket_dict(bracket_dict)
        
        # Convert to FrozenBracket
        conf_data = bracket_dict[conference]
        
        # Convert WC matchups
        wc_matchups = []
        for wc_matchup in conf_data['WC']:
            team_a = Team(
                name=wc_matchup['a_name'],
                seed=wc_matchup['a_seed'],
                conf=conference
            )
            team_b = Team(
                name=wc_matchup['b_name'],
                seed=wc_matchup['b_seed'],
                conf=conference
            )
            winner = Team(
                name=wc_matchup['winner_name'],
                seed=wc_matchup['winner_seed'],
                conf=conference
            )
            wc_matchups.append(Matchup(a=team_a, b=team_b, winner=winner))
        
        # Convert DIV matchups
        div_matchups = []
        for div_matchup in conf_data['DIV']:
            team_a = Team(
                name=div_matchup['a_name'],
                seed=div_matchup['a_seed'],
                conf=conference
            )
            team_b = Team(
                name=div_matchup['b_name'],
                seed=div_matchup['b_seed'],
                conf=conference
            )
            winner = Team(
                name=div_matchup['winner_name'],
                seed=div_matchup['winner_seed'],
                conf=conference
            )
            div_matchups.append(Matchup(a=team_a, b=team_b, winner=winner))
        
        # Convert CONF matchup
        conf_matchup_dict = conf_data['CONF']
        team_a = Team(
            name=conf_matchup_dict['a_name'],
            seed=conf_matchup_dict['a_seed'],
            conf=conference
        )
        team_b = Team(
            name=conf_matchup_dict['b_name'],
            seed=conf_matchup_dict['b_seed'],
            conf=conference
        )
        winner = Team(
            name=conf_matchup_dict['winner_name'],
            seed=conf_matchup_dict['winner_seed'],
            conf=conference
        )
        conf_matchup = Matchup(a=team_a, b=team_b, winner=winner)
        
        # Create FrozenBracket
        return FrozenBracket(
            wc=wc_matchups,
            div=div_matchups,
            conf=conf_matchup,
            champ=conf_matchup.winner
        )
    
    @staticmethod
    def _extract_bracket_dict(simulation_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract bracket dictionary from simulation output.
        
        Handles different simulation output formats.
        """
        # If already in bracket dict format, return it
        if 'AFC' in simulation_output and 'NFC' in simulation_output:
            if isinstance(simulation_output['AFC'], dict) and 'WC' in simulation_output['AFC']:
                return simulation_output
        
        # Otherwise, try to construct from seeding and matchup_probs
        # This is a fallback for older simulation formats
        bracket_dict = {
            'AFC': {'WC': [], 'DIV': [], 'CONF': {}},
            'NFC': {'WC': [], 'DIV': [], 'CONF': {}}
        }
        
        # This would need to be implemented based on actual simulation output format
        # For now, raise error if format is not recognized
        raise ValueError(
            "Simulation output format not recognized. "
            "Expected bracket dict with 'AFC' and 'NFC' keys containing WC, DIV, CONF."
        )

