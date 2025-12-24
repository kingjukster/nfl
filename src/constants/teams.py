"""
NFL Team Constants

Team colors, logos, name mappings, and related utilities.
"""
from pathlib import Path
from typing import Optional
from src.config import data_config

# NFL Team Colors (primary colors)
TEAM_COLORS = {
    'KC': '#E31837', 'BUF': '#00338D', 'BAL': '#241773', 'HOU': '#03202F',
    'LAC': '#0080C6', 'DEN': '#FB4F14', 'PIT': '#FFB612', 'CIN': '#FB4F14',
    'CLE': '#311D00', 'IND': '#002C5F', 'JAX': '#006778', 'TEN': '#0C2340',
    'MIA': '#008E97', 'NE': '#002244', 'NYJ': '#125740', 'LV': '#000000',
    'DET': '#0076B6', 'PHI': '#004C54', 'TB': '#D50A0A', 'SEA': '#002244',
    'MIN': '#4F2683', 'WAS': '#773141', 'GB': '#203731', 'CHI': '#0B162A',
    'CAR': '#0085CA', 'NO': '#D3BC8D', 'ATL': '#A71930', 'DAL': '#003594',
    'NYG': '#0B2265', 'ARI': '#97233F', 'LAR': '#003594', 'SF': '#AA0000'
}

# Team logo filename aliases (for file system naming)
TEAM_LOGO_ALIASES = {
    'JAX': 'JAG',  # Jacksonville uses JAG.gif
    'LV': 'LAV',   # Las Vegas uses LAV.gif
}

# Team name mappings (handle variations and historical changes)
TEAM_NAME_MAPPING = {
    # Standard abbreviations
    'ARI': 'ARI', 'ARZ': 'ARI', 'Arizona': 'ARI', 'PHO': 'ARI', 'Phoenix': 'ARI',
    'ATL': 'ATL', 'Atlanta': 'ATL',
    'BAL': 'BAL', 'Baltimore': 'BAL',
    'BUF': 'BUF', 'Buffalo': 'BUF',
    'CAR': 'CAR', 'Carolina': 'CAR',
    'CHI': 'CHI', 'Chicago': 'CHI',
    'CIN': 'CIN', 'Cincinnati': 'CIN',
    'CLE': 'CLE', 'Cleveland': 'CLE',
    'DAL': 'DAL', 'Dallas': 'DAL',
    'DEN': 'DEN', 'Denver': 'DEN',
    'DET': 'DET', 'Detroit': 'DET',
    'GB': 'GB', 'GNB': 'GB', 'Green Bay': 'GB',
    'HOU': 'HOU', 'Houston': 'HOU',
    'IND': 'IND', 'Indianapolis': 'IND',
    'JAX': 'JAX', 'JAC': 'JAX', 'Jacksonville': 'JAX',
    'KC': 'KC', 'KAN': 'KC', 'Kansas City': 'KC',
    'LAC': 'LAC', 'SD': 'LAC', 'San Diego': 'LAC', 'Los Angeles Chargers': 'LAC',
    'LAR': 'LAR', 'LA': 'LAR', 'STL': 'LAR', 'STL Rams': 'LAR', 
    'Los Angeles Rams': 'LAR', 'LA Rams': 'LAR',
    'LV': 'LV', 'OAK': 'LV', 'Las Vegas': 'LV', 'Oakland': 'LV', 'LV Raiders': 'LV',
    'MIA': 'MIA', 'Miami': 'MIA',
    'MIN': 'MIN', 'Minnesota': 'MIN',
    'NE': 'NE', 'NWE': 'NE', 'New England': 'NE',
    'NO': 'NO', 'NOR': 'NO', 'New Orleans': 'NO',
    'NYG': 'NYG', 'New York Giants': 'NYG',
    'NYJ': 'NYJ', 'New York Jets': 'NYJ',
    'PHI': 'PHI', 'Philadelphia': 'PHI',
    'PIT': 'PIT', 'Pittsburgh': 'PIT',
    'SF': 'SF', 'SFO': 'SF', 'San Francisco': 'SF',
    'SEA': 'SEA', 'Seattle': 'SEA',
    'TB': 'TB', 'TAM': 'TB', 'Tampa Bay': 'TB',
    'TEN': 'TEN', 'Tennessee': 'TEN',
    'WAS': 'WAS', 'WSH': 'WAS', 'Washington': 'WAS', 
    'Washington Redskins': 'WAS', 'Washington Football Team': 'WAS', 
    'Washington Commanders': 'WAS',
    # Historical team changes
    'BAL Colts': 'IND',  # Colts moved from Baltimore to Indianapolis
    'HOU Oilers': 'TEN',  # Oilers became Titans
}

# Standard team abbreviations (NFL standard)
STANDARD_TEAM_ABBREVIATIONS = {
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
    'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAR', 'LAC', 'LV', 'MIA',
    'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB',
    'TEN', 'WAS'
}


def get_team_color(team: str) -> str:
    """
    Get team color, with fallback to gray if not found.
    
    Parameters:
    -----------
    team : str
        Team abbreviation (e.g., 'KC', 'BUF')
        
    Returns:
    --------
    str
        Hex color code
    """
    normalized = normalize_team_name(team)
    return TEAM_COLORS.get(normalized, '#808080')


def get_team_logo_path(team: str) -> Optional[Path]:
    """
    Get path to team logo file.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
        
    Returns:
    --------
    Path or None
        Path to logo file if exists, None otherwise
    """
    normalized = normalize_team_name(team)
    logo_name = TEAM_LOGO_ALIASES.get(normalized, normalized)
    logo_path = data_config.logo_dir / f"{logo_name}.gif"
    
    if logo_path.exists():
        return logo_path
    return None


def normalize_team_name(team: str) -> str:
    """
    Normalize team name to standard abbreviation.
    
    Parameters:
    -----------
    team : str
        Team name or abbreviation (various formats)
        
    Returns:
    --------
    str
        Standard team abbreviation (e.g., 'KC', 'BUF')
    """
    if not team:
        return 'UNK'
    
    # Try direct lookup first
    team_upper = team.upper().strip()
    if team_upper in STANDARD_TEAM_ABBREVIATIONS:
        return team_upper
    
    # Try mapping
    if team_upper in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[team_upper]
    
    # Try case-insensitive lookup in mapping
    for key, value in TEAM_NAME_MAPPING.items():
        if key.upper() == team_upper:
            return value
    
    # Return original if no match found
    return team_upper

