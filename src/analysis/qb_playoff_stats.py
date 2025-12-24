"""
Calculate quarterback playoff statistics from play-by-play data.

This module provides functions to:
- Filter play-by-play data to playoff games only
- Calculate QB playoff performance metrics
- Map teams to their primary quarterbacks
- Normalize metrics for visualization
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.data.fetching.load_historical_data import load_pbp_data, load_rosters
from src.config import data_config

logger = logging.getLogger(__name__)


def filter_playoff_games(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter play-by-play data to playoff games only.
    
    Parameters:
    -----------
    pbp_df : pd.DataFrame
        Play-by-play data
        
    Returns:
    --------
    pd.DataFrame : Filtered to playoff games only
    """
    if pbp_df.empty:
        return pbp_df
    
    df = pbp_df.copy()
    
    # Method 1: Use game_type column (most reliable)
    if 'game_type' in df.columns:
        playoff_df = df[df['game_type'] == 'POST'].copy()
        if not playoff_df.empty:
            logger.debug(f"Filtered to {len(playoff_df)} playoff plays using game_type")
            return playoff_df
    
    # Method 2: Use season_type column
    if 'season_type' in df.columns:
        playoff_df = df[df['season_type'] == 'POST'].copy()
        if not playoff_df.empty:
            logger.debug(f"Filtered to {len(playoff_df)} playoff plays using season_type")
            return playoff_df
    
    # Method 3: Use week column (playoffs are week > 18 for 18-game seasons, > 17 for 17-game)
    if 'week' in df.columns and 'season' in df.columns:
        playoff_df = df.copy()
        # Determine max regular season week by season
        playoff_df['max_reg_week'] = playoff_df['season'].apply(
            lambda s: 18 if s >= 2021 else 17
        )
        playoff_df = playoff_df[playoff_df['week'] > playoff_df['max_reg_week']].copy()
        playoff_df = playoff_df.drop(columns=['max_reg_week'])
        if not playoff_df.empty:
            logger.debug(f"Filtered to {len(playoff_df)} playoff plays using week")
            return playoff_df
    
    logger.warning("Could not identify playoff games - no game_type, season_type, or week column")
    return pd.DataFrame()


def get_qb_for_team(team: str, season: int, rosters_df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """
    Get the primary quarterback for a team in a given season.
    
    Parameters:
    -----------
    team : str
        Team abbreviation
    season : int
        Season year
    rosters_df : pd.DataFrame
        Roster data with player information
        
    Returns:
    --------
    Dict with 'player_id' and 'player_name', or None if not found
    """
    if rosters_df.empty:
        return None
    
    # Filter to team, season, and QB position
    team_qbs = rosters_df[
        (rosters_df['team'] == team) &
        (rosters_df['season'] == season) &
        (rosters_df['position'] == 'QB')
    ].copy()
    
    if team_qbs.empty:
        logger.debug(f"No QB found for {team} in {season}")
        return None
    
    # If multiple QBs, prefer the one with most games or highest status
    # Sort by games (if available) or just take first
    if 'games' in team_qbs.columns:
        team_qbs = team_qbs.sort_values('games', ascending=False)
    elif 'gs' in team_qbs.columns:  # games started
        team_qbs = team_qbs.sort_values('gs', ascending=False)
    
    primary_qb = team_qbs.iloc[0]
    
    return {
        'player_id': primary_qb.get('player_id', primary_qb.get('gsis_id', None)),
        'player_name': primary_qb.get('player_name', primary_qb.get('name', 'Unknown'))
    }


def calculate_qb_playoff_stats(pbp_df: pd.DataFrame, playoff_teams: List[str], 
                                min_playoff_games: int = 3) -> pd.DataFrame:
    """
    Calculate playoff statistics for quarterbacks.
    
    Parameters:
    -----------
    pbp_df : pd.DataFrame
        Play-by-play data (should be filtered to playoff games)
    playoff_teams : List[str]
        List of team abbreviations in playoffs
    min_playoff_games : int
        Minimum number of playoff games required (default: 3)
        
    Returns:
    --------
    pd.DataFrame : QB playoff statistics with columns:
        - player_id, player_name, team
        - playoff_games, playoff_wins
        - epa_per_play, completion_pct, td_rate, int_rate, sack_rate, win_rate
    """
    if pbp_df.empty:
        logger.warning("No play-by-play data provided")
        return pd.DataFrame()
    
    # Filter to passing plays only
    passing_plays = pbp_df[
        (pbp_df['play_type'] == 'pass') &
        (pbp_df['passer_id'].notna()) &
        (pbp_df['posteam'].isin(playoff_teams))
    ].copy()
    
    if passing_plays.empty:
        logger.warning("No passing plays found for playoff teams")
        return pd.DataFrame()
    
    # Group by passer_id and calculate stats
    qb_stats = []
    
    for passer_id, group in passing_plays.groupby('passer_id'):
        # Get unique games for this QB
        games = group['game_id'].unique()
        n_games = len(games)
        
        if n_games < min_playoff_games:
            continue
        
        # Get QB name (from first occurrence)
        qb_name = group['passer_player_name'].iloc[0] if 'passer_player_name' in group.columns else 'Unknown'
        if pd.isna(qb_name):
            qb_name = 'Unknown'
        
        # Get team (most common team for this QB)
        teams = group['posteam'].value_counts()
        primary_team = teams.index[0] if not teams.empty else None
        
        # Calculate passing stats
        pass_attempts = (group['pass_attempt'] == 1).sum()
        completions = (group['complete_pass'] == 1).sum()
        touchdowns = ((group['touchdown'] == 1) & (group['pass_attempt'] == 1)).sum()
        interceptions = (group['interception'] == 1).sum()
        
        # EPA per play
        epa_values = group['epa'].fillna(0)
        epa_per_play = epa_values.mean() if len(epa_values) > 0 else 0.0
        
        # Completion percentage
        completion_pct = (completions / pass_attempts * 100) if pass_attempts > 0 else 0.0
        
        # TD rate (per attempt)
        td_rate = (touchdowns / pass_attempts * 100) if pass_attempts > 0 else 0.0
        
        # INT rate (per attempt) - will be inverted later
        int_rate = (interceptions / pass_attempts * 100) if pass_attempts > 0 else 0.0
        
        # Sack rate
        dropbacks = group['dropback'].sum() if 'dropback' in group.columns else pass_attempts
        sacks = (group['sack'] == 1).sum()
        sack_rate = (sacks / dropbacks * 100) if dropbacks > 0 else 0.0
        
        # Calculate win rate from game results
        wins = 0
        for game_id in games:
            game_plays = group[group['game_id'] == game_id]
            if not game_plays.empty:
                # Get final scores for this game
                home_team = game_plays['home_team'].iloc[0] if 'home_team' in game_plays.columns else None
                away_team = game_plays['away_team'].iloc[0] if 'away_team' in game_plays.columns else None
                
                # Try to get final scores
                if 'home_score' in game_plays.columns and 'away_score' in game_plays.columns:
                    home_score = game_plays['home_score'].max()
                    away_score = game_plays['away_score'].max()
                    
                    if primary_team == home_team and home_score > away_score:
                        wins += 1
                    elif primary_team == away_team and away_score > home_score:
                        wins += 1
                # Alternative: use posteam_score vs defteam_score if available
                elif 'posteam_score' in game_plays.columns and 'defteam_score' in game_plays.columns:
                    # Get QB's team score when they had possession
                    qb_team_score = game_plays[game_plays['posteam'] == primary_team]['posteam_score'].max()
                    # Get opponent score (when QB's team was on defense)
                    opp_score = game_plays[game_plays['defteam'] == primary_team]['defteam_score'].max()
                    if not pd.isna(qb_team_score) and not pd.isna(opp_score) and qb_team_score > opp_score:
                        wins += 1
        
        win_rate = (wins / n_games * 100) if n_games > 0 else 0.0
        
        qb_stats.append({
            'player_id': passer_id,
            'player_name': qb_name,
            'team': primary_team,
            'playoff_games': n_games,
            'playoff_wins': wins,
            'epa_per_play': epa_per_play,
            'completion_pct': completion_pct,
            'td_rate': td_rate,
            'int_rate': int_rate,  # Will be inverted in normalization
            'sack_rate': sack_rate,  # Will be inverted in normalization
            'win_rate': win_rate
        })
    
    if not qb_stats:
        logger.warning("No QBs found with sufficient playoff games")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(qb_stats)
    logger.info(f"Calculated playoff stats for {len(result_df)} QBs")
    
    return result_df


def normalize_qb_metrics(qb_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize QB metrics to 0-1 scale for radar chart.
    Inverts "bad" stats (INT rate, sack rate) so higher = better.
    
    Parameters:
    -----------
    qb_stats_df : pd.DataFrame
        QB statistics DataFrame
        
    Returns:
    --------
    pd.DataFrame : DataFrame with normalized metrics (0-1 scale)
    """
    if qb_stats_df.empty:
        return qb_stats_df
    
    df = qb_stats_df.copy()
    
    # Metrics to normalize (higher is better)
    positive_metrics = ['epa_per_play', 'completion_pct', 'td_rate', 'win_rate']
    
    # Metrics to invert (lower is better, so we invert them)
    negative_metrics = ['int_rate', 'sack_rate']
    
    # Normalize positive metrics using min-max scaling
    for metric in positive_metrics:
        if metric in df.columns:
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                df[f'{metric}_norm'] = (df[metric] - col_min) / (col_max - col_min)
            else:
                df[f'{metric}_norm'] = 0.5  # All same value, set to middle
    
    # Invert and normalize negative metrics
    for metric in negative_metrics:
        if metric in df.columns:
            # Invert: higher rate = worse, so we want (1 - normalized_rate)
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                # Normalize first, then invert
                normalized = (df[metric] - col_min) / (col_max - col_min)
                df[f'{metric}_norm'] = 1.0 - normalized  # Invert so higher = better
            else:
                df[f'{metric}_norm'] = 0.5
    
    return df


def get_qb_playoff_stats_for_season(season: int, playoff_teams: List[str], 
                                     min_playoff_games: int = 3,
                                     start_year: int = 1999) -> pd.DataFrame:
    """
    Get QB playoff statistics for a given season's playoff teams.
    Maps playoff teams to their QBs, then calculates those QBs' playoff stats
    across all historical playoff games.
    
    Parameters:
    -----------
    season : int
        Current season year
    playoff_teams : List[str]
        List of team abbreviations in playoffs
    min_playoff_games : int
        Minimum playoff games required
    start_year : int
        Starting year for historical data (default: 1999, when PBP data starts)
        
    Returns:
    --------
    pd.DataFrame : QB playoff statistics
    """
    # Load roster data to map teams to QBs
    logger.info(f"Loading roster data for season {season}")
    rosters_df = load_rosters(start_year=season, end_year=season)
    
    # Map playoff teams to their QBs
    team_to_qb = {}
    for team in playoff_teams:
        qb_info = get_qb_for_team(team, season, rosters_df)
        if qb_info and qb_info.get('player_id'):
            team_to_qb[team] = qb_info['player_id']
            logger.debug(f"{team} -> QB: {qb_info.get('player_name', 'Unknown')} ({qb_info['player_id']})")
    
    if not team_to_qb:
        logger.warning("Could not map any playoff teams to QBs")
        # Fall back to calculating stats for all QBs from playoff teams
        logger.info("Falling back to calculating stats for all QBs from playoff teams")
    
    # Load historical PBP data (all years to get full playoff history)
    logger.info(f"Loading playoff PBP data from {start_year} to {season}")
    pbp_df = load_pbp_data(start_year=start_year, end_year=season)
    
    if pbp_df.empty:
        logger.warning("No PBP data loaded")
        return pd.DataFrame()
    
    # Filter to playoff games only
    playoff_pbp = filter_playoff_games(pbp_df)
    
    if playoff_pbp.empty:
        logger.warning("No playoff games found in PBP data")
        return pd.DataFrame()
    
    # If we have QB mappings, filter to those QBs; otherwise use all QBs from playoff teams
    if team_to_qb:
        # Filter to QBs we care about (but get their stats from all playoff games, not just current season)
        qb_ids = list(team_to_qb.values())
        playoff_pbp_filtered = playoff_pbp[playoff_pbp['passer_id'].isin(qb_ids)].copy()
        if not playoff_pbp_filtered.empty:
            # Calculate stats for these specific QBs (don't filter by team in calculate function)
            # We'll pass all teams so it doesn't filter, but we've already filtered by QB
            all_teams = playoff_pbp_filtered['posteam'].unique().tolist()
            qb_stats = calculate_qb_playoff_stats(playoff_pbp_filtered, all_teams, min_playoff_games)
        else:
            logger.warning("No playoff plays found for mapped QBs")
            qb_stats = pd.DataFrame()
    else:
        # Fall back: calculate stats for all QBs from playoff teams
        qb_stats = calculate_qb_playoff_stats(playoff_pbp, playoff_teams, min_playoff_games)
    
    if qb_stats.empty:
        return qb_stats
    
    # Add current team information
    if team_to_qb:
        # Create reverse mapping (QB ID -> team)
        qb_to_team = {v: k for k, v in team_to_qb.items()}
        qb_stats['current_team'] = qb_stats['player_id'].map(qb_to_team)
        # Use current_team if available, otherwise use team from stats
        qb_stats['team'] = qb_stats['current_team'].fillna(qb_stats['team'])
    
    # Normalize metrics
    qb_stats = normalize_qb_metrics(qb_stats)
    
    return qb_stats

