"""
NFL Constants Module

Centralized location for all NFL-related constants including:
- Team information (colors, logos, names)
- Conference and division structure
- Team name mappings
- Stadium information
"""

from .teams import (
    TEAM_COLORS,
    TEAM_LOGO_ALIASES,
    get_team_color,
    get_team_logo_path,
    normalize_team_name,
    TEAM_NAME_MAPPING,
    STANDARD_TEAM_ABBREVIATIONS
)

from .conferences import (
    NFL_CONFERENCES,
    DOME_TEAMS,
    COLD_WEATHER_TEAMS,
    get_team_conference,
    get_team_division
)

__all__ = [
    'TEAM_COLORS',
    'TEAM_LOGO_ALIASES',
    'get_team_color',
    'get_team_logo_path',
    'normalize_team_name',
    'TEAM_NAME_MAPPING',
    'STANDARD_TEAM_ABBREVIATIONS',
    'NFL_CONFERENCES',
    'DOME_TEAMS',
    'COLD_WEATHER_TEAMS',
    'get_team_conference',
    'get_team_division',
]

