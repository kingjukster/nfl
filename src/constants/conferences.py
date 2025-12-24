"""
NFL Conference and Division Constants

Conference structure, divisions, and related utilities.
"""
from typing import Optional

# NFL Conference and Division Structure
NFL_CONFERENCES = {
    'AFC': {
        'divisions': ['AFC_North', 'AFC_South', 'AFC_East', 'AFC_West'],
        'teams': {
            'AFC_North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_West': ['DEN', 'KC', 'LV', 'LAC']
        }
    },
    'NFC': {
        'divisions': ['NFC_North', 'NFC_South', 'NFC_East', 'NFC_West'],
        'teams': {
            'NFC_North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_West': ['ARI', 'LAR', 'SF', 'SEA']
        }
    }
}

# Stadium types for weather adjustments
DOME_TEAMS = ['ATL', 'DET', 'IND', 'NO', 'DAL', 'HOU', 'ARI', 'LAR', 'LV']
COLD_WEATHER_TEAMS = ['GB', 'CHI', 'MIN', 'BUF', 'NE', 'PIT', 'CLE', 'DEN']


def get_team_conference(team: str) -> Optional[str]:
    """
    Get the conference for a given team.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
        
    Returns:
    --------
    str or None
        Conference name ('AFC' or 'NFC') or None if not found
    """
    from .teams import normalize_team_name
    
    normalized = normalize_team_name(team)
    
    for conference, conf_data in NFL_CONFERENCES.items():
        for division_teams in conf_data['teams'].values():
            if normalized in division_teams:
                return conference
    
    return None


def get_team_division(team: str) -> Optional[str]:
    """
    Get the division for a given team.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
        
    Returns:
    --------
    str or None
        Division name (e.g., 'AFC_North') or None if not found
    """
    from .teams import normalize_team_name
    
    normalized = normalize_team_name(team)
    
    for conference, conf_data in NFL_CONFERENCES.items():
        for division, teams in conf_data['teams'].items():
            if normalized in teams:
                return division
    
    return None

