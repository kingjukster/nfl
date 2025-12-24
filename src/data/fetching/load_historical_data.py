"""
Utility functions to load and use historical NFL data in the project.

This module provides easy access to the fetched historical data for use in:
- Model training (extended historical context)
- Playoff predictions (better team stats)
- Player predictions (more training data)
- Analysis and validation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import data_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pbp_data(start_year: int = 1999, end_year: Optional[int] = None,
                 seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load play-by-play data from historical dataset.
    
    Parameters:
    -----------
    start_year : int
        Starting year (default: 1999)
    end_year : int, optional
        Ending year (default: current year)
    seasons : List[int], optional
        Specific seasons to load
        
    Returns:
    --------
    pd.DataFrame : Play-by-play data
    """
    # Try aggregated file first
    aggregated_file = data_config.historical_data_dir / "aggregated" / "pbp_data_1999_present.csv"
    
    if aggregated_file.exists():
        try:
            logger.info(f"Loading aggregated play-by-play data from {aggregated_file}")
            df = pd.read_csv(aggregated_file, low_memory=False)
            
            # Filter by season if specified
            if 'season' in df.columns:
                if seasons:
                    df = df[df['season'].isin(seasons)]
                elif end_year:
                    df = df[(df['season'] >= start_year) & (df['season'] <= end_year)]
                else:
                    df = df[df['season'] >= start_year]
            
            logger.info(f"Loaded {len(df)} play-by-play records")
            return df
        except Exception as e:
            logger.warning(f"Error loading aggregated file: {e}")
    
    # Fallback to individual year files
    logger.info("Loading from individual year files...")
    dataframes = []
    pbp_dir = data_config.historical_data_dir / "nflfastr" / "pbp"
    
    if seasons is None:
        if end_year is None:
            from datetime import datetime
            end_year = datetime.now().year
        seasons = list(range(start_year, end_year + 1))
    
    for year in seasons:
        year_file = pbp_dir / f"pbp_{year}.csv"
        if year_file.exists():
            try:
                df = pd.read_csv(year_file, low_memory=False)
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Error loading {year_file}: {e}")
    
    if dataframes:
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(combined)} play-by-play records from {len(dataframes)} files")
        return combined
    
    logger.warning("No play-by-play data found")
    return pd.DataFrame()


def derive_team_stats_from_pbp(pbp_df: pd.DataFrame, season: Optional[int] = None) -> pd.DataFrame:
    """
    Derive team season statistics from play-by-play data.
    
    This creates team-level aggregates similar to what you'd get from PFR team stats.
    
    Parameters:
    -----------
    pbp_df : pd.DataFrame
        Play-by-play data
    season : int, optional
        Filter to specific season
        
    Returns:
    --------
    pd.DataFrame : Team season statistics
    """
    if pbp_df.empty:
        return pd.DataFrame()
    
    df = pbp_df.copy()
    
    if season:
        if 'season' in df.columns:
            df = df[df['season'] == season]
        else:
            logger.warning("No season column in pbp data")
    
    # CRITICAL: Filter to regular season only (exclude playoffs)
    # Playoff games should not be included in regular season stats for seeding
    if 'week' in df.columns:
        # Regular season is typically weeks 1-18 (or 1-17 for older seasons)
        max_reg_week = 18 if (season and season >= 2021) else 17
        if season is None and 'season' in df.columns:
            # Filter by season if available
            for s in df['season'].unique():
                max_week = 18 if s >= 2021 else 17
                df.loc[df['season'] == s, 'is_reg_season'] = df.loc[df['season'] == s, 'week'] <= max_week
            df = df[df['is_reg_season'] == True].copy()
            df = df.drop(columns=['is_reg_season'])
        else:
            df = df[df['week'] <= max_reg_week].copy()
        logger.debug(f"Filtered PBP data to regular season (week <= {max_reg_week})")
    elif 'game_type' in df.columns:
        df = df[df['game_type'] == 'REG'].copy()
        logger.debug("Filtered PBP data to regular season using game_type")
    elif 'season_type' in df.columns:
        df = df[df['season_type'] == 'REG'].copy()
        logger.debug("Filtered PBP data to regular season using season_type")
    else:
        logger.warning("No week/game_type/season_type column in PBP data. Cannot filter playoffs!")
    
    # Group by team and season
    group_cols = ['posteam', 'season'] if 'season' in df.columns else ['posteam']
    
    team_stats = []
    
    for (team, season_val), group in df.groupby(group_cols):
        if pd.isna(team):
            continue
        
        stats = {
            'team': team,
            'season': season_val if 'season' in df.columns else None,
        }
        
        # Offensive stats (when team has possession)
        stats['offensive_yards'] = group['yards_gained'].fillna(0).sum()
        stats['passing_yards'] = group[group['play_type'] == 'pass']['yards_gained'].fillna(0).sum()
        stats['rushing_yards'] = group[group['play_type'] == 'run']['yards_gained'].fillna(0).sum()
        stats['passing_tds'] = (group['touchdown'] == 1) & (group['play_type'] == 'pass').sum()
        stats['rushing_tds'] = (group['touchdown'] == 1) & (group['play_type'] == 'run').sum()
        stats['total_tds'] = (group['touchdown'] == 1).sum()
        stats['interceptions'] = (group['interception'] == 1).sum()
        stats['fumbles'] = (group['fumble'] == 1).sum()
        stats['turnovers'] = stats['interceptions'] + stats['fumbles']
        
        # Points (if available)
        if 'posteam_score' in group.columns:
            # Get final score for each game
            game_scores = group.groupby('game_id')['posteam_score'].max()
            stats['points_for'] = game_scores.sum()
        
        team_stats.append(stats)
    
    result_df = pd.DataFrame(team_stats)
    
    # Also calculate defensive stats (when team is on defense)
    def_stats = []
    for (team, season_val), group in df.groupby(['defteam', 'season'] if 'season' in df.columns else ['defteam']):
        if pd.isna(team):
            continue
        
        stats = {
            'team': team,
            'season': season_val if 'season' in df.columns else None,
        }
        
        stats['defensive_yards_allowed'] = group['yards_gained'].fillna(0).sum()
        stats['defensive_tds_allowed'] = (group['touchdown'] == 1).sum()
        
        if 'defteam_score' in group.columns:
            game_scores = group.groupby('game_id')['defteam_score'].max()
            stats['points_against'] = game_scores.sum()
        
        def_stats.append(stats)
    
    def_df = pd.DataFrame(def_stats)
    
    # Merge offensive and defensive stats
    if not result_df.empty and not def_df.empty:
        merge_cols = ['team', 'season'] if 'season' in result_df.columns else ['team']
        result_df = result_df.merge(def_df, on=merge_cols, how='outer', suffixes=('', '_def'))
    
    return result_df


def load_rosters(start_year: int = 1999, end_year: Optional[int] = None) -> pd.DataFrame:
    """Load roster data."""
    rosters_file = data_config.historical_data_dir / "nflfastr" / "rosters" / "rosters_1999_2025.csv"
    
    if rosters_file.exists():
        df = pd.read_csv(rosters_file)
        if 'season' in df.columns:
            if end_year is None:
                from datetime import datetime
                end_year = datetime.now().year
            df = df[(df['season'] >= start_year) & (df['season'] <= end_year)]
        return df
    
    return pd.DataFrame()


def load_schedules(start_year: int = 1999, end_year: Optional[int] = None) -> pd.DataFrame:
    """Load schedule data."""
    schedules_file = data_config.historical_data_dir / "nflfastr" / "schedules" / "schedules_1999_2025.csv"
    
    if schedules_file.exists():
        df = pd.read_csv(schedules_file)
        if 'season' in df.columns:
            if end_year is None:
                from datetime import datetime
                end_year = datetime.now().year
            df = df[(df['season'] >= start_year) & (df['season'] <= end_year)]
        return df
    
    return pd.DataFrame()


def get_team_season_stats_from_pbp(team: str, season: int) -> Optional[Dict]:
    """
    Get comprehensive team statistics for a specific team and season from PBP data.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
    season : int
        Season year
        
    Returns:
    --------
    Dict : Team statistics, or None if not found
    """
    pbp = load_pbp_data(seasons=[season])
    if pbp.empty:
        return None
    
    team_pbp = pbp[(pbp['posteam'] == team) | (pbp['defteam'] == team)]
    if team_pbp.empty:
        return None
    
    # Calculate various stats
    stats = {
        'team': team,
        'season': season,
    }
    
    # Offensive stats
    off_plays = team_pbp[team_pbp['posteam'] == team]
    stats['offensive_plays'] = len(off_plays)
    stats['offensive_yards'] = off_plays['yards_gained'].fillna(0).sum()
    stats['passing_yards'] = off_plays[off_plays['play_type'] == 'pass']['yards_gained'].fillna(0).sum()
    stats['rushing_yards'] = off_plays[off_plays['play_type'] == 'run']['yards_gained'].fillna(0).sum()
    
    # Defensive stats
    def_plays = team_pbp[team_pbp['defteam'] == team]
    stats['defensive_yards_allowed'] = def_plays['yards_gained'].fillna(0).sum()
    
    # Game results (regular season only for seeding)
    schedules = load_schedules(seasons=[season])
    if not schedules.empty:
        # Filter to regular season only (exclude playoffs)
        if 'game_type' in schedules.columns:
            schedules = schedules[schedules['game_type'] == 'REG'].copy()
        elif 'season_type' in schedules.columns:
            schedules = schedules[schedules['season_type'] == 'REG'].copy()
        elif 'week' in schedules.columns:
            max_reg_week = 18 if season >= 2021 else 17
            schedules = schedules[schedules['week'] <= max_reg_week].copy()
        
        if team in schedules['home_team'].values or team in schedules['away_team'].values:
            team_games = schedules[(schedules['home_team'] == team) | (schedules['away_team'] == team)]
            # Calculate wins/losses if scores available
            if 'home_score' in team_games.columns and 'away_score' in team_games.columns:
                wins = 0
                losses = 0
                for _, game in team_games.iterrows():
                    if game['home_team'] == team:
                        if game['home_score'] > game['away_score']:
                            wins += 1
                        elif game['home_score'] < game['away_score']:
                            losses += 1
                    else:  # away team
                        if game['away_score'] > game['home_score']:
                            wins += 1
                        elif game['away_score'] < game['home_score']:
                            losses += 1
                
                stats['wins'] = wins
                stats['losses'] = losses
                stats['win_pct'] = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    return stats


def enhance_team_stats_with_pbp(team_stats_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Enhance existing team stats DataFrame with data derived from play-by-play.
    
    Parameters:
    -----------
    team_stats_df : pd.DataFrame
        Existing team statistics
    season : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Enhanced team statistics
    """
    pbp = load_pbp_data(seasons=[season])
    if pbp.empty:
        logger.warning(f"No PBP data for {season}, returning original stats")
        return team_stats_df
    
    pbp_stats = derive_team_stats_from_pbp(pbp, season)
    if pbp_stats.empty:
        return team_stats_df
    
    # Merge with existing stats
    merge_cols = ['team', 'season'] if 'season' in team_stats_df.columns else ['team']
    
    enhanced = team_stats_df.merge(
        pbp_stats,
        on=merge_cols,
        how='left',
        suffixes=('', '_pbp')
    )
    
    # Fill missing values from PBP where original is missing
    for col in pbp_stats.columns:
        if col not in merge_cols:
            if col in enhanced.columns and f"{col}_pbp" in enhanced.columns:
                enhanced[col] = enhanced[col].fillna(enhanced[f"{col}_pbp"])
                enhanced = enhanced.drop(columns=[f"{col}_pbp"])
    
    return enhanced


if __name__ == "__main__":
    # Example usage
    print("Loading play-by-play data...")
    pbp = load_pbp_data(start_year=2020, end_year=2023)
    print(f"Loaded {len(pbp)} records")
    
    print("\nDeriving team stats from PBP...")
    team_stats = derive_team_stats_from_pbp(pbp, season=2023)
    print(f"Derived stats for {len(team_stats)} teams")
    print(team_stats.head())

