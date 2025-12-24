"""
Normalize historical NFL data across different eras.

This module handles:
- Team name changes (STL → LA Rams, etc.)
- Stat column name differences across eras
- Season length changes (14 → 16 → 17 games)
- Missing data handling
- Schema unification
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import centralized team constants
from src.constants import (
    TEAM_NAME_MAPPING,
    STANDARD_TEAM_ABBREVIATIONS,
    normalize_team_name
)


def normalize_team_names(df: pd.DataFrame, team_column: str = 'team') -> pd.DataFrame:
    """
    Normalize team names/abbreviations across different eras.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with team names to normalize
    team_column : str
        Name of the column containing team names
        
    Returns:
    --------
    pd.DataFrame : DataFrame with normalized team names
    """
    if team_column not in df.columns:
        logger.warning(f"Team column '{team_column}' not found in DataFrame")
        return df
    
    df = df.copy()
    
    # Use centralized normalize_team_name function
    def map_team_name(team_name):
        if pd.isna(team_name):
            return team_name
        return normalize_team_name(team_name)
        
        # Try to extract standard abbreviation (first 2-3 chars)
        if len(team_str) >= 2:
            potential_abbr = team_str[:3] if len(team_str) >= 3 else team_str[:2]
            if potential_abbr in STANDARD_TEAM_ABBREVIATIONS:
                return potential_abbr
        
        # Return original if no mapping found
        logger.debug(f"No mapping found for team: {team_name}")
        return team_str
    
    df[team_column] = df[team_column].apply(map_team_name)
    
    return df


def normalize_stat_columns(df: pd.DataFrame, era: str = 'modern') -> pd.DataFrame:
    """
    Standardize column names across different eras.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with potentially inconsistent column names
    era : str
        Era identifier ('early', 'modern', 'recent')
        
    Returns:
    --------
    pd.DataFrame : DataFrame with normalized column names
    """
    df = df.copy()
    
    # Common column name mappings
    column_mappings = {
        # Points
        'PF': 'points_for', 'Points For': 'points_for', 'Pts': 'points_for',
        'PA': 'points_against', 'Points Against': 'points_against', 'Pts Allowed': 'points_against',
        
        # Yards
        'Yds': 'yards', 'Yards': 'yards', 'Total Yds': 'yards',
        'Off Yds': 'offensive_yards', 'Off Yards': 'offensive_yards',
        'Def Yds': 'defensive_yards', 'Def Yards': 'defensive_yards',
        
        # Passing
        'Pass Yds': 'passing_yards', 'Pass Yards': 'passing_yards', 'PYds': 'passing_yards',
        'Pass TD': 'passing_tds', 'Pass TDs': 'passing_tds', 'PTD': 'passing_tds',
        'Int': 'interceptions', 'INT': 'interceptions', 'Interceptions': 'interceptions',
        
        # Rushing
        'Rush Yds': 'rushing_yards', 'Rush Yards': 'rushing_yards', 'RYds': 'rushing_yards',
        'Rush TD': 'rushing_tds', 'Rush TDs': 'rushing_tds', 'RTD': 'rushing_tds',
        
        # Receiving
        'Rec Yds': 'receiving_yards', 'Rec Yards': 'receiving_yards', 'ReYds': 'receiving_yards',
        'Rec TD': 'receiving_tds', 'Rec TDs': 'receiving_tds', 'ReTD': 'receiving_tds',
        'Rec': 'receptions', 'Receptions': 'receptions',
        
        # Turnovers
        'TO': 'turnovers', 'Turnovers': 'turnovers', 'Giveaways': 'turnovers',
        'Takeaways': 'takeaways', 'TA': 'takeaways',
        
        # Games
        'G': 'games', 'Games': 'games', 'GP': 'games',
        'W': 'wins', 'Wins': 'wins', 'Win': 'wins',
        'L': 'losses', 'Losses': 'losses', 'Loss': 'losses',
        'T': 'ties', 'Ties': 'ties', 'Tie': 'ties',
    }
    
    # Rename columns
    df.rename(columns=column_mappings, inplace=True)
    
    # Normalize remaining column names (lowercase, replace spaces with underscores)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    return df


def handle_season_length_changes(df: pd.DataFrame, season_column: str = 'season') -> pd.DataFrame:
    """
    Account for different season lengths across eras.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with season data
    season_column : str
        Name of the season column
        
    Returns:
    --------
    pd.DataFrame : DataFrame with season length information added
    """
    if season_column not in df.columns:
        logger.warning(f"Season column '{season_column}' not found")
        return df
    
    df = df.copy()
    
    def get_season_length(year):
        """Get number of games in a season based on year."""
        if year < 1978:
            return 14
        elif year < 2021:
            return 16
        else:
            return 17
    
    df['season_length'] = df[season_column].apply(get_season_length)
    
    # If games_played exists, calculate per-game stats
    if 'games' in df.columns or 'games_played' in df.columns:
        games_col = 'games_played' if 'games_played' in df.columns else 'games'
        
        # Normalize stats to per-game if needed
        stat_cols_to_normalize = [
            'points_for', 'points_against', 'offensive_yards', 'defensive_yards',
            'passing_yards', 'rushing_yards', 'receiving_yards'
        ]
        
        for col in stat_cols_to_normalize:
            if col in df.columns and games_col in df.columns:
                # Only normalize if values seem like totals (not already per-game)
                if df[col].max() > 100:  # Heuristic: totals are usually > 100
                    df[f'{col}_pg'] = df[col] / df[games_col]
    
    return df


def merge_pfr_nflfastr_data(pfr_data: pd.DataFrame, pbp_data: pd.DataFrame,
                            merge_key: str = 'game_id') -> pd.DataFrame:
    """
    Combine PFR and nflfastR data intelligently.
    
    Parameters:
    -----------
    pfr_data : pd.DataFrame
        Data from Pro-Football-Reference
    pbp_data : pd.DataFrame
        Play-by-play data from nflfastR
    merge_key : str
        Key to merge on (game_id, season, etc.)
        
    Returns:
    --------
    pd.DataFrame : Combined dataset
    """
    # This is a placeholder - actual implementation would depend on
    # the specific structure of PFR and nflfastR data
    
    # For now, return concatenated data with source indicator
    pfr_data['data_source'] = 'pfr'
    pbp_data['data_source'] = 'nflfastr'
    
    # If merge_key exists in both, merge
    if merge_key in pfr_data.columns and merge_key in pbp_data.columns:
        merged = pd.merge(
            pfr_data,
            pbp_data,
            on=merge_key,
            how='outer',
            suffixes=('_pfr', '_pbp')
        )
        return merged
    else:
        # Otherwise, just concatenate
        return pd.concat([pfr_data, pbp_data], ignore_index=True)


def create_unified_schema() -> Dict[str, List[str]]:
    """
    Define consistent schema for all years.
    
    Returns:
    --------
    Dict[str, List[str]] : Dictionary mapping data types to required columns
    """
    schemas = {
        'team_season': [
            'team', 'season', 'wins', 'losses', 'ties',
            'points_for', 'points_against',
            'offensive_yards', 'defensive_yards',
            'turnovers', 'takeaways',
            'season_length'
        ],
        'game': [
            'game_id', 'season', 'week',
            'home_team', 'away_team',
            'home_score', 'away_score',
            'game_date'
        ],
        'pbp': [
            'game_id', 'play_id', 'season', 'week',
            'posteam', 'defteam',
            'down', 'ydstogo', 'yards_gained',
            'epa', 'wp', 'play_type'
        ],
        'player_season': [
            'player_name', 'team', 'position', 'season',
            'games_played', 'games_started',
            'passing_yards', 'passing_tds', 'interceptions',
            'rushing_yards', 'rushing_tds',
            'receiving_yards', 'receiving_tds', 'receptions'
        ]
    }
    
    return schemas


def normalize_dataframe(df: pd.DataFrame, data_type: str, 
                       team_column: str = 'team',
                       season_column: str = 'season') -> pd.DataFrame:
    """
    Comprehensive normalization function that applies all normalizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to normalize
    data_type : str
        Type of data (team_season, game, pbp, player_season)
    team_column : str
        Name of team column
    season_column : str
        Name of season column
        
    Returns:
    --------
    pd.DataFrame : Normalized DataFrame
    """
    logger.info(f"Normalizing {data_type} data...")
    
    df = df.copy()
    
    # Apply all normalizations
    if team_column in df.columns:
        df = normalize_team_names(df, team_column)
    
    df = normalize_stat_columns(df)
    
    if season_column in df.columns:
        df = handle_season_length_changes(df, season_column)
    
    # Get schema for this data type
    schemas = create_unified_schema()
    if data_type in schemas:
        required_cols = schemas[data_type]
        # Add missing columns with NaN
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
                logger.debug(f"Added missing column: {col}")
    
    logger.info(f"Normalization complete. Shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test normalization functions
    test_data = pd.DataFrame({
        'team': ['STL', 'OAK', 'SD', 'BAL', 'KC'],
        'season': [1999, 2000, 2001, 2002, 2003],
        'W': [10, 12, 8, 9, 11],
        'L': [6, 4, 8, 7, 5],
        'PF': [300, 350, 280, 320, 340]
    })
    
    print("Original data:")
    print(test_data)
    
    normalized = normalize_dataframe(test_data, 'team_season')
    print("\nNormalized data:")
    print(normalized)

