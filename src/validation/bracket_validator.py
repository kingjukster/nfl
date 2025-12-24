"""
Bracket validation functions.

Validates bracket dictionaries and frozen bracket structures.
"""

from typing import Dict, Any
from src.models.bracket import FrozenBracket, Matchup, Team


def validate_bracket_dict(bracket_dict: Dict[str, Any]) -> None:
    """
    Validate a bracket dictionary structure.
    
    Raises ValueError if invalid.
    
    Expected format:
    {
        'AFC': {
            'WC': [{'a_name': str, 'a_seed': int, 'b_name': str, 'b_seed': int, 'winner_name': str, 'winner_seed': int}, ...],
            'DIV': [...],
            'CONF': {'a_name': str, ...}
        },
        'NFC': {...}
    }
    """
    if not isinstance(bracket_dict, dict):
        raise ValueError("Bracket dict must be a dictionary")
    
    for conf in ['AFC', 'NFC']:
        if conf not in bracket_dict:
            raise ValueError(f"Missing conference: {conf}")
        
        conf_data = bracket_dict[conf]
        if not isinstance(conf_data, dict):
            raise ValueError(f"{conf} data must be a dictionary")
        
        # Validate Wild Card round
        if 'WC' not in conf_data:
            raise ValueError(f"{conf}: Missing Wild Card round")
        wc_matchups = conf_data['WC']
        if not isinstance(wc_matchups, list) or len(wc_matchups) != 3:
            raise ValueError(f"{conf}: Expected 3 Wild Card matchups, got {len(wc_matchups) if isinstance(wc_matchups, list) else 'non-list'}")
        
        for i, matchup in enumerate(wc_matchups):
            _validate_matchup_dict(matchup, f"{conf} WC[{i}]")
        
        # Validate Divisional round
        if 'DIV' not in conf_data:
            raise ValueError(f"{conf}: Missing Divisional round")
        div_matchups = conf_data['DIV']
        if not isinstance(div_matchups, list) or len(div_matchups) != 2:
            raise ValueError(f"{conf}: Expected 2 Divisional matchups, got {len(div_matchups) if isinstance(div_matchups, list) else 'non-list'}")
        
        for i, matchup in enumerate(div_matchups):
            _validate_matchup_dict(matchup, f"{conf} DIV[{i}]")
        
        # Validate Conference Championship
        if 'CONF' not in conf_data:
            raise ValueError(f"{conf}: Missing Conference Championship")
        conf_matchup = conf_data['CONF']
        if not isinstance(conf_matchup, dict):
            raise ValueError(f"{conf}: Conference Championship must be a dictionary")
        _validate_matchup_dict(conf_matchup, f"{conf} CONF")
        
        # Validate winner consistency
        for matchup in wc_matchups + div_matchups + [conf_matchup]:
            winner = matchup.get('winner_name')
            a_name = matchup.get('a_name')
            b_name = matchup.get('b_name')
            if winner not in [a_name, b_name]:
                raise ValueError(f"Winner {winner} must be either {a_name} or {b_name}")


def _validate_matchup_dict(matchup: Dict[str, Any], context: str) -> None:
    """Validate a single matchup dictionary."""
    required_fields = ['a_name', 'a_seed', 'b_name', 'b_seed', 'winner_name', 'winner_seed']
    for field in required_fields:
        if field not in matchup:
            raise ValueError(f"{context}: Missing required field: {field}")
    
    # Validate seeds
    a_seed = matchup['a_seed']
    b_seed = matchup['b_seed']
    winner_seed = matchup['winner_seed']
    
    if not (1 <= a_seed <= 7):
        raise ValueError(f"{context}: Invalid seed a_seed={a_seed} (must be 1-7)")
    if not (1 <= b_seed <= 7):
        raise ValueError(f"{context}: Invalid seed b_seed={b_seed} (must be 1-7)")
    if not (1 <= winner_seed <= 7):
        raise ValueError(f"{context}: Invalid seed winner_seed={winner_seed} (must be 1-7)")
    
    # Validate winner is one of the teams
    winner = matchup['winner_name']
    a_name = matchup['a_name']
    b_name = matchup['b_name']
    
    if winner not in [a_name, b_name]:
        raise ValueError(f"{context}: Winner {winner} must be either {a_name} or {b_name}")
    
    # Validate winner_seed matches winner
    if winner == a_name and winner_seed != a_seed:
        raise ValueError(f"{context}: Winner {winner} has seed {winner_seed} but should be {a_seed}")
    if winner == b_name and winner_seed != b_seed:
        raise ValueError(f"{context}: Winner {winner} has seed {winner_seed} but should be {b_seed}")


def validate_frozen_bracket(frozen: FrozenBracket) -> None:
    """
    Validate a FrozenBracket structure.
    
    Raises ValueError if invalid.
    """
    if not isinstance(frozen, FrozenBracket):
        raise ValueError("Must be a FrozenBracket instance")
    
    # Validate structure (already done in __post_init__, but double-check)
    if len(frozen.wc) != 3:
        raise ValueError(f"Expected 3 Wild Card matchups, got {len(frozen.wc)}")
    if len(frozen.div) != 2:
        raise ValueError(f"Expected 2 Divisional matchups, got {len(frozen.div)}")
    
    # Validate all matchups
    all_matchups = frozen.wc + frozen.div + [frozen.conf]
    for matchup in all_matchups:
        if not isinstance(matchup, Matchup):
            raise ValueError("All matchups must be Matchup instances")
        
        # Winner must be either a or b
        if matchup.winner.name not in [matchup.a.name, matchup.b.name]:
            raise ValueError(f"Winner {matchup.winner.name} must be either {matchup.a.name} or {matchup.b.name}")
        
        # Teams must be in same conference
        if matchup.a.conf != matchup.b.conf:
            raise ValueError(f"Teams must be in same conference: {matchup.a.conf} != {matchup.b.conf}")
        if matchup.winner.conf != matchup.a.conf:
            raise ValueError(f"Winner must be in same conference: {matchup.winner.conf} != {matchup.a.conf}")
    
    # Validate champion matches conference winner
    if frozen.champ.name != frozen.conf.winner.name:
        raise ValueError(f"Champion {frozen.champ.name} must match conference winner {frozen.conf.winner.name}")
    
    # Check for duplicate teams in same round (unless properly promoted)
    _check_duplicate_teams(frozen.wc, "Wild Card")
    _check_duplicate_teams(frozen.div, "Divisional")
    
    # Validate promotion: DIV teams must come from WC winners or seed 1/2
    wc_winners = {m.winner.name for m in frozen.wc}
    div_teams = set()
    for m in frozen.div:
        div_teams.add(m.a.name)
        div_teams.add(m.b.name)
    
    # Seed 1 and 2 are always in DIV, so they're allowed
    seed1_team = next((m.a.name for m in frozen.div if m.a.seed == 1), None)
    seed2_team = next((m.a.name for m in frozen.div if m.a.seed == 2), None)
    if seed1_team:
        div_teams.discard(seed1_team)
    if seed2_team:
        div_teams.discard(seed2_team)
    
    # All other DIV teams must be WC winners
    for team in div_teams:
        if team not in wc_winners:
            raise ValueError(f"Divisional team {team} must be a Wild Card winner or seed 1/2")


def _check_duplicate_teams(matchups: list, round_name: str) -> None:
    """Check for duplicate teams in a round."""
    all_teams = []
    for matchup in matchups:
        all_teams.append(matchup.a.name)
        all_teams.append(matchup.b.name)
    
    seen = set()
    duplicates = []
    for team in all_teams:
        if team in seen:
            duplicates.append(team)
        seen.add(team)
    
    if duplicates:
        raise ValueError(f"{round_name} round has duplicate teams: {duplicates}")

