"""
Aggregate historical NFL data from multiple sources into unified datasets.

This module combines data from:
- Pro-Football-Reference (1966-present)
- nflfastR (1999-present)
- Existing Kaggle data (1999-2022)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import data_config
from src.data.fetching.normalize_historical_data import normalize_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Position groupings
OFFENSIVE_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K']
DEFENSIVE_POSITIONS = ['CB', 'LB', 'DT', 'S', 'DE', 'OLB', 'ILB', 'NT', 'FS', 'SS']


def load_year_files(data_dir: Path, pattern: str, start_year: int = 1966,
                    end_year: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Load all CSV files matching a pattern from a directory.
    
    Parameters:
    -----------
    data_dir : Path
        Directory to search
    pattern : str
        Filename pattern (e.g., "team_stats_{year}.csv")
    start_year : int
        Starting year
    end_year : int, optional
        Ending year
        
    Returns:
    --------
    List[pd.DataFrame] : List of loaded DataFrames
    """
    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year
    
    dataframes = []
    
    for year in range(start_year, end_year + 1):
        filename = data_dir / pattern.format(year=year)
        if filename.exists():
            try:
                df = pd.read_csv(filename)
                if len(df) > 0:
                    dataframes.append(df)
                    logger.debug(f"Loaded {filename}")
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
        else:
            logger.debug(f"File not found: {filename}")
    
    return dataframes


def get_era_bucket(season: int) -> str:
    """
    Assign era bucket based on season.
    
    Parameters:
    -----------
    season : int
        Season year
        
    Returns:
    --------
    str : Era bucket identifier
    """
    if season < 1978:
        return 'pre_1978'
    elif season < 2002:
        return '1978_2001'
    elif season < 2021:
        return '2002_2020'
    else:
        return '2021_present'


def calculate_offensive_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fantasy points for offensive players.
    
    Standard scoring:
    - QB: (passing_tds * 4) + (passing_yards / 25) - (interceptions * 2) + (rushing_yards / 10) + (rushing_tds * 6)
    - RB/WR/TE: (rushing_yards / 10) + (receiving_yards / 10) + (rushing_tds * 6) + (receiving_tds * 6)
    - PPR: Add (receptions * 1) for RB/WR/TE
    """
    df = df.copy()
    
    # Standard scoring
    df['fantasy_points'] = 0.0
    
    # QB scoring
    qb_mask = df['position'] == 'QB'
    if qb_mask.any():
        df.loc[qb_mask, 'fantasy_points'] = (
            (df.loc[qb_mask, 'passing_tds'].fillna(0) * 4) +
            (df.loc[qb_mask, 'passing_yards'].fillna(0) / 25) -
            (df.loc[qb_mask, 'interceptions'].fillna(0) * 2) +
            (df.loc[qb_mask, 'rushing_yards'].fillna(0) / 10) +
            (df.loc[qb_mask, 'rushing_tds'].fillna(0) * 6)
        )
    
    # RB/WR/TE scoring
    skill_positions = ['RB', 'WR', 'TE']
    skill_mask = df['position'].isin(skill_positions)
    if skill_mask.any():
        df.loc[skill_mask, 'fantasy_points'] = (
            (df.loc[skill_mask, 'rushing_yards'].fillna(0) / 10) +
            (df.loc[skill_mask, 'receiving_yards'].fillna(0) / 10) +
            (df.loc[skill_mask, 'rushing_tds'].fillna(0) * 6) +
            (df.loc[skill_mask, 'receiving_tds'].fillna(0) * 6)
        )
        
        # PPR scoring
        df.loc[skill_mask, 'fantasy_points_ppr'] = (
            df.loc[skill_mask, 'fantasy_points'] +
            (df.loc[skill_mask, 'receptions'].fillna(0) * 1)
        )
    else:
        df['fantasy_points_ppr'] = df['fantasy_points']
    
    return df


def calculate_defensive_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fantasy points for defensive players (IDP).
    
    Standard IDP scoring:
    - Solo tackle: 1.0
    - Assist tackle: 0.5
    - Sack: 2.0
    - INT: 3.0
    - Forced fumble: 2.0
    - Fumble recovery: 2.0
    - TD: 6.0
    - Safety: 2.0
    """
    df = df.copy()
    
    df['fantasy_points'] = (
        (df['solo_tackles'].fillna(0) * 1.0) +
        (df['assist_tackles'].fillna(0) * 0.5) +
        (df['sacks'].fillna(0) * 2.0) +
        (df['interceptions'].fillna(0) * 3.0) +
        (df['forced_fumbles'].fillna(0) * 2.0) +
        (df['fumble_recoveries'].fillna(0) * 2.0) +
        (df['defensive_tds'].fillna(0) * 6.0) +
        (df['safeties'].fillna(0) * 2.0)
    )
    
    # PPR IDP (same as standard for defense)
    df['fantasy_points_ppr'] = df['fantasy_points']
    
    return df


def aggregate_team_season_stats(start_year: int = 1966,
                                end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Combine all team season statistics into a single dataset.
    
    Parameters:
    -----------
    start_year : int
        Starting year
    end_year : int, optional
        Ending year
        
    Returns:
    --------
    pd.DataFrame : Combined team season statistics
    """
    logger.info("Aggregating team season statistics...")
    
    pfr_dir = data_config.historical_data_dir / "pfr" / "team_stats"
    
    # Load PFR data
    dataframes = load_year_files(pfr_dir, "team_stats_{year}.csv", start_year, end_year)
    
    if not dataframes:
        logger.warning("No team stats files found")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Normalize
    combined = normalize_dataframe(combined, 'team_season')
    
    # Remove duplicates
    if 'team' in combined.columns and 'season' in combined.columns:
        combined = combined.drop_duplicates(subset=['team', 'season'], keep='last')
    
    # Calculate point_diff if not present
    if 'point_diff' not in combined.columns:
        if 'points_for' in combined.columns and 'points_against' in combined.columns:
            combined['point_diff'] = combined['points_for'] - combined['points_against']
    
    # Add playoff_round_reached if not present
    # Note: This can be populated from game_results or playoff bracket data
    # Values: 'None', 'Wild Card', 'Divisional', 'Conference', 'Super Bowl', 'Won Super Bowl'
    if 'playoff_round_reached' not in combined.columns:
        combined['playoff_round_reached'] = 'None'
        logger.debug("playoff_round_reached column added (default: 'None'). Can be populated from playoff game results.")
    
    # Sort by season, then team (deterministic)
    if 'season' in combined.columns:
        combined = combined.sort_values(['season', 'team']).reset_index(drop=True)
    
    logger.info(f"Aggregated {len(combined)} team-season records")
    
    return combined


def aggregate_player_season_stats(start_year: int = 1966,
                                  end_year: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine all player season statistics into separate offense and defense datasets.
    
    Parameters:
    -----------
    start_year : int
        Starting year
    end_year : int, optional
        Ending year
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (offense_df, defense_df)
    """
    logger.info("Aggregating player season statistics (offense and defense)...")
    
    pfr_dir = data_config.historical_data_dir / "pfr" / "player_stats"
    
    # Load PFR data (may have position-specific files)
    dataframes = []
    
    if end_year is None:
        end_year = datetime.now().year
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'CB', 'LB', 'DT', 'S', 'DE', 'OLB', 'ILB', 'NT', 'FS', 'SS', 'DEF']
    for year in range(start_year, end_year + 1):
        for pos in positions:
            filename = pfr_dir / f"player_stats_{year}_{pos.lower()}.csv"
            if filename.exists():
                try:
                    df = pd.read_csv(filename)
                    if len(df) > 0:
                        dataframes.append(df)
                except Exception as e:
                    logger.debug(f"Error loading {filename}: {e}")
        
        # Also try without position suffix
        filename = pfr_dir / f"player_stats_{year}.csv"
        if filename.exists():
            try:
                df = pd.read_csv(filename)
                if len(df) > 0:
                    dataframes.append(df)
            except Exception as e:
                logger.debug(f"Error loading {filename}: {e}")
    
    if not dataframes:
        logger.warning("No player stats files found")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Normalize
    combined = normalize_dataframe(combined, 'player_season')
    
    # Ensure position column exists
    if 'position' not in combined.columns:
        logger.warning("No position column found in player data")
        return pd.DataFrame(), pd.DataFrame()
    
    # Split into offense and defense
    offensive_mask = combined['position'].isin(OFFENSIVE_POSITIONS)
    defensive_mask = combined['position'].isin(DEFENSIVE_POSITIONS)
    
    offense_df = combined[offensive_mask].copy()
    defense_df = combined[defensive_mask].copy()
    
    # Process offense
    if len(offense_df) > 0:
        # Add era_bucket
        offense_df['era_bucket'] = offense_df['season'].apply(get_era_bucket)
        
        # Ensure required columns exist
        required_off_cols = ['player_id', 'player_name', 'position', 'team', 'season', 
                            'era_bucket', 'games_played', 'fantasy_points']
        for col in required_off_cols:
            if col not in offense_df.columns:
                if col == 'player_id':
                    # Generate player_id from player_name if missing
                    offense_df['player_id'] = offense_df['player_name'].astype(str).str.lower().str.replace(' ', '_')
                elif col == 'fantasy_points':
                    # Calculate fantasy points
                    offense_df = calculate_offensive_fantasy_points(offense_df)
                else:
                    offense_df[col] = None
        
        # Remove duplicates (use player_id if available, otherwise player_name)
        key_cols = ['player_id', 'season', 'team'] if 'player_id' in offense_df.columns else ['player_name', 'season', 'team']
        offense_df = offense_df.drop_duplicates(subset=key_cols, keep='last')
        
        # Sort deterministically
        sort_cols = ['season', 'player_id'] if 'player_id' in offense_df.columns else ['season', 'player_name']
        offense_df = offense_df.sort_values(sort_cols).reset_index(drop=True)
        
        logger.info(f"Aggregated {len(offense_df)} offensive player-season records")
    
    # Process defense
    if len(defense_df) > 0:
        # Add era_bucket
        defense_df['era_bucket'] = defense_df['season'].apply(get_era_bucket)
        
        # Ensure required columns exist
        required_def_cols = ['player_id', 'player_name', 'position', 'team', 'season',
                           'era_bucket', 'games_played', 'fantasy_points']
        for col in required_def_cols:
            if col not in defense_df.columns:
                if col == 'player_id':
                    # Generate player_id from player_name if missing
                    defense_df['player_id'] = defense_df['player_name'].astype(str).str.lower().str.replace(' ', '_')
                elif col == 'fantasy_points':
                    # Calculate fantasy points
                    defense_df = calculate_defensive_fantasy_points(defense_df)
                else:
                    defense_df[col] = None
        
        # Remove duplicates
        key_cols = ['player_id', 'season', 'team'] if 'player_id' in defense_df.columns else ['player_name', 'season', 'team']
        defense_df = defense_df.drop_duplicates(subset=key_cols, keep='last')
        
        # Sort deterministically
        sort_cols = ['season', 'player_id'] if 'player_id' in defense_df.columns else ['season', 'player_name']
        defense_df = defense_df.sort_values(sort_cols).reset_index(drop=True)
        
        logger.info(f"Aggregated {len(defense_df)} defensive player-season records")
    
    return offense_df, defense_df


def aggregate_game_results(start_year: int = 1966,
                          end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Combine all game results into a single dataset.
    
    Parameters:
    -----------
    start_year : int
        Starting year
    end_year : int, optional
        Ending year
        
    Returns:
    --------
    pd.DataFrame : Combined game results
    """
    logger.info("Aggregating game results...")
    
    pfr_dir = data_config.historical_data_dir / "pfr" / "game_results"
    
    # Load PFR data
    dataframes = load_year_files(pfr_dir, "game_results_{year}.csv", start_year, end_year)
    
    if not dataframes:
        logger.warning("No game results files found")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Normalize
    combined = normalize_dataframe(combined, 'game')
    
    # Remove duplicates
    if 'game_id' in combined.columns:
        combined = combined.drop_duplicates(subset=['game_id'], keep='last')
    elif 'season' in combined.columns and 'week' in combined.columns:
        # Try to create game_id if it doesn't exist
        if 'home_team' in combined.columns and 'away_team' in combined.columns:
            combined['game_id'] = (
                combined['season'].astype(str) + '_' +
                combined['week'].astype(str) + '_' +
                combined['home_team'].astype(str) + '_' +
                combined['away_team'].astype(str)
            )
            combined = combined.drop_duplicates(subset=['game_id'], keep='last')
    
    # Sort
    if 'season' in combined.columns and 'week' in combined.columns:
        combined = combined.sort_values(['season', 'week']).reset_index(drop=True)
    
    logger.info(f"Aggregated {len(combined)} game records")
    
    return combined


def aggregate_pbp_data(start_year: int = 1999,
                      end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Combine all play-by-play data into a single dataset.
    
    Parameters:
    -----------
    start_year : int
        Starting year (1999+)
    end_year : int, optional
        Ending year
        
    Returns:
    --------
    pd.DataFrame : Combined play-by-play data
    """
    logger.info("Aggregating play-by-play data...")
    
    nflfastr_dir = data_config.historical_data_dir / "nflfastr" / "pbp"
    
    # Try to load combined file first
    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year
    
    combined_file = nflfastr_dir / f"pbp_{start_year}_{end_year}.csv"
    if combined_file.exists():
        try:
            df = pd.read_csv(combined_file)
            logger.info(f"Loaded combined pbp file: {combined_file}")
            return normalize_dataframe(df, 'pbp')
        except Exception as e:
            logger.warning(f"Error loading combined file: {e}")
    
    # Otherwise, load individual year files
    dataframes = load_year_files(nflfastr_dir, "pbp_{year}.csv", start_year, end_year)
    
    if not dataframes:
        logger.warning("No play-by-play files found")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Normalize
    combined = normalize_dataframe(combined, 'pbp')
    
    # Remove duplicates
    if 'game_id' in combined.columns and 'play_id' in combined.columns:
        combined = combined.drop_duplicates(subset=['game_id', 'play_id'], keep='last')
    
    # Sort
    if 'season' in combined.columns and 'game_id' in combined.columns:
        combined = combined.sort_values(['season', 'game_id', 'play_id']).reset_index(drop=True)
    
    logger.info(f"Aggregated {len(combined)} play-by-play records")
    
    return combined


def get_expected_team_count(season: int) -> int:
    """Get expected number of teams for a given season."""
    if season < 1970:
        return 26  # AFL-NFL merger era
    elif season < 1995:
        return 28
    elif season < 1999:
        return 30
    else:
        return 32  # Current era


def validate_layer_a_completeness(df: pd.DataFrame, start_year: int = 1966) -> Dict[str, Any]:
    """
    Validate Layer A (team season stats) completeness.
    Logs warnings for missing data but does not fail.
    
    Returns:
    -------
    Dict with validation results
    """
    if df.empty:
        logger.warning("Layer A validation: DataFrame is empty")
        return {
            'missing_seasons': [],
            'missing_teams': {},
            'duplicate_records': [],
            'null_required_fields': {}
        }
    
    current_year = datetime.now().year
    expected_seasons = set(range(start_year, current_year + 1))
    actual_seasons = set(df['season'].unique()) if 'season' in df.columns else set()
    missing_seasons = sorted(expected_seasons - actual_seasons)
    
    if missing_seasons:
        logger.warning(f"Layer A: Missing seasons: {missing_seasons}")
    
    # Check for missing teams per season
    missing_teams = {}
    if 'season' in df.columns and 'team' in df.columns:
        for season in expected_seasons:
            if season not in actual_seasons:
                continue
            season_df = df[df['season'] == season]
            expected_team_count = get_expected_team_count(season)
            if len(season_df) < expected_team_count:
                # Try to identify which teams are missing (simplified - would need team list)
                missing_teams[season] = expected_team_count - len(season_df)
                logger.warning(f"Layer A: Season {season} has {len(season_df)} teams, expected {expected_team_count}")
    
    # Check duplicates
    duplicates = []
    if 'season' in df.columns and 'team' in df.columns:
        dup_mask = df.duplicated(subset=['season', 'team'], keep=False)
        if dup_mask.any():
            duplicates = df[dup_mask][['season', 'team']].to_dict('records')
            logger.warning(f"Layer A: Found {len(duplicates)} duplicate (season, team) records")
    
    # Check required fields
    required_fields = ['season', 'team', 'wins', 'losses', 'points_for', 'points_against']
    null_counts = {}
    for col in required_fields:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_counts[col] = int(null_count)
                logger.warning(f"Layer A: Column '{col}' has {null_count} null values")
    
    return {
        'missing_seasons': missing_seasons,
        'missing_teams': missing_teams,
        'duplicate_records': duplicates,
        'null_required_fields': null_counts
    }


def validate_layer_b_append_safety(existing_file: Path, new_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Ensure append-only semantics for Layer B.
    Logs warnings for overlapping records but does not fail.
    
    Returns:
    -------
    Dict with validation results
    """
    if not existing_file.exists():
        return {
            'new_records': len(new_data),
            'overlapping_records': 0,
            'era_bucket_coverage': new_data['era_bucket'].value_counts().to_dict() if 'era_bucket' in new_data.columns else {}
        }
    
    try:
        existing = pd.read_csv(existing_file)
        key_cols = ['player_id', 'season', 'team'] if 'player_id' in new_data.columns else ['player_name', 'season', 'team']
        
        # Check which keys exist in both
        existing_keys = set(zip(*[existing[col] for col in key_cols]))
        new_keys = set(zip(*[new_data[col] for col in key_cols]))
        overlapping = existing_keys & new_keys
        
        if overlapping:
            logger.warning(f"Layer B: Found {len(overlapping)} overlapping records (will be skipped to maintain append-only)")
        
        return {
            'overlapping_records': len(overlapping),
            'new_records': len(new_keys - existing_keys),
            'era_bucket_coverage': new_data['era_bucket'].value_counts().to_dict() if 'era_bucket' in new_data.columns else {}
        }
    except Exception as e:
        logger.warning(f"Layer B: Could not validate append safety: {e}")
        return {
            'new_records': len(new_data),
            'overlapping_records': 0,
            'era_bucket_coverage': {}
        }


def enforce_deterministic_aggregation(df: pd.DataFrame, sort_cols: List[str]) -> pd.DataFrame:
    """
    Ensure deterministic output order.
    Always sort by sort_cols before writing to CSV.
    """
    if df.empty:
        return df
    return df.sort_values(sort_cols).reset_index(drop=True)


def create_unified_dataset() -> Dict[str, pd.DataFrame]:
    """
    Create unified datasets combining all sources.
    
    Returns:
    --------
    Dict[str, pd.DataFrame] : Dictionary mapping dataset names to DataFrames
    """
    logger.info("Creating unified historical datasets...")
    
    output_dir = data_config.historical_data_dir / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {}
    
    # Aggregate each data type
    try:
        team_stats = aggregate_team_season_stats()
        if len(team_stats) > 0:
            # Validate Layer A
            validation_results = validate_layer_a_completeness(team_stats)
            logger.info("Layer A validation complete (warnings logged above if any issues)")
            
            # Enforce deterministic sorting
            team_stats = enforce_deterministic_aggregation(team_stats, ['season', 'team'])
            
            filename = output_dir / "team_season_stats_1966_present.csv"
            team_stats.to_csv(filename, index=False)
            datasets['team_season_stats'] = team_stats
            logger.info(f"Saved team season stats to {filename}")
    except Exception as e:
        logger.error(f"Error aggregating team stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        game_results = aggregate_game_results()
        if len(game_results) > 0:
            filename = output_dir / "game_results_1966_present.csv"
            game_results.to_csv(filename, index=False)
            datasets['game_results'] = game_results
            logger.info(f"Saved game results to {filename}")
    except Exception as e:
        logger.error(f"Error aggregating game results: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        offense_stats, defense_stats = aggregate_player_season_stats()
        
        # Handle offense (append-only)
        if len(offense_stats) > 0:
            offense_file = output_dir / "player_seasons_offense.csv"
            validation_results = validate_layer_b_append_safety(offense_file, offense_stats)
            logger.info(f"Layer B (Offense): {validation_results['new_records']} new records, "
                       f"{validation_results['overlapping_records']} overlapping (will skip)")
            
            # Append-only: only add new records
            if offense_file.exists():
                try:
                    existing = pd.read_csv(offense_file)
                    key_cols = ['player_id', 'season', 'team'] if 'player_id' in offense_stats.columns else ['player_name', 'season', 'team']
                    existing_keys = set(zip(*[existing[col] for col in key_cols]))
                    new_keys = set(zip(*[offense_stats[col] for col in key_cols]))
                    new_mask = [key not in existing_keys for key in new_keys]
                    offense_stats = offense_stats[new_mask]
                    if len(offense_stats) > 0:
                        offense_stats = pd.concat([existing, offense_stats], ignore_index=True)
                    else:
                        offense_stats = existing
                except Exception as e:
                    logger.warning(f"Could not read existing offense file, creating new: {e}")
            
            # Enforce deterministic sorting
            sort_cols = ['season', 'player_id'] if 'player_id' in offense_stats.columns else ['season', 'player_name']
            offense_stats = enforce_deterministic_aggregation(offense_stats, sort_cols)
            
            offense_stats.to_csv(offense_file, index=False)
            datasets['player_seasons_offense'] = offense_stats
            logger.info(f"Saved offensive player season stats to {offense_file}")
        
        # Handle defense (append-only)
        if len(defense_stats) > 0:
            defense_file = output_dir / "player_seasons_defense.csv"
            validation_results = validate_layer_b_append_safety(defense_file, defense_stats)
            logger.info(f"Layer B (Defense): {validation_results['new_records']} new records, "
                       f"{validation_results['overlapping_records']} overlapping (will skip)")
            
            # Append-only: only add new records
            if defense_file.exists():
                try:
                    existing = pd.read_csv(defense_file)
                    key_cols = ['player_id', 'season', 'team'] if 'player_id' in defense_stats.columns else ['player_name', 'season', 'team']
                    existing_keys = set(zip(*[existing[col] for col in key_cols]))
                    new_keys = set(zip(*[defense_stats[col] for col in key_cols]))
                    new_mask = [key not in existing_keys for key in new_keys]
                    defense_stats = defense_stats[new_mask]
                    if len(defense_stats) > 0:
                        defense_stats = pd.concat([existing, defense_stats], ignore_index=True)
                    else:
                        defense_stats = existing
                except Exception as e:
                    logger.warning(f"Could not read existing defense file, creating new: {e}")
            
            # Enforce deterministic sorting
            sort_cols = ['season', 'player_id'] if 'player_id' in defense_stats.columns else ['season', 'player_name']
            defense_stats = enforce_deterministic_aggregation(defense_stats, sort_cols)
            
            defense_stats.to_csv(defense_file, index=False)
            datasets['player_seasons_defense'] = defense_stats
            logger.info(f"Saved defensive player season stats to {defense_file}")
            
    except Exception as e:
        logger.error(f"Error aggregating player stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        pbp_data = aggregate_pbp_data()
        if len(pbp_data) > 0:
            filename = output_dir / "pbp_data_1999_present.csv"
            pbp_data.to_csv(filename, index=False)
            datasets['pbp_data'] = pbp_data
            logger.info(f"Saved play-by-play data to {filename}")
    except Exception as e:
        logger.error(f"Error aggregating pbp data: {e}")
    
    logger.info(f"\nCreated {len(datasets)} unified datasets")
    
    return datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate historical NFL data')
    parser.add_argument('--data-type', choices=['team', 'player', 'games', 'pbp', 'all'],
                       default='all', help='Type of data to aggregate')
    parser.add_argument('--start-year', type=int, default=1966, help='Starting year')
    parser.add_argument('--end-year', type=int, default=None, help='Ending year')
    
    args = parser.parse_args()
    
    if args.data_type == 'all':
        datasets = create_unified_dataset()
    elif args.data_type == 'team':
        df = aggregate_team_season_stats(args.start_year, args.end_year)
        filename = data_config.historical_data_dir / "aggregated" / "team_season_stats_1966_present.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    elif args.data_type == 'player':
        offense_df, defense_df = aggregate_player_season_stats(args.start_year, args.end_year)
        output_dir = data_config.historical_data_dir / "aggregated"
        
        if len(offense_df) > 0:
            offense_file = output_dir / "player_seasons_offense.csv"
            # Handle append-only
            if offense_file.exists():
                existing = pd.read_csv(offense_file)
                key_cols = ['player_id', 'season', 'team'] if 'player_id' in offense_df.columns else ['player_name', 'season', 'team']
                existing_keys = set(zip(*[existing[col] for col in key_cols]))
                new_keys = set(zip(*[offense_df[col] for col in key_cols]))
                new_mask = [key not in existing_keys for key in new_keys]
                offense_df = offense_df[new_mask]
                if len(offense_df) > 0:
                    offense_df = pd.concat([existing, offense_df], ignore_index=True)
                else:
                    offense_df = existing
            offense_df = enforce_deterministic_aggregation(offense_df, ['season', 'player_id'] if 'player_id' in offense_df.columns else ['season', 'player_name'])
            offense_df.to_csv(offense_file, index=False)
            print(f"Saved offense to {offense_file}")
        
        if len(defense_df) > 0:
            defense_file = output_dir / "player_seasons_defense.csv"
            # Handle append-only
            if defense_file.exists():
                existing = pd.read_csv(defense_file)
                key_cols = ['player_id', 'season', 'team'] if 'player_id' in defense_df.columns else ['player_name', 'season', 'team']
                existing_keys = set(zip(*[existing[col] for col in key_cols]))
                new_keys = set(zip(*[defense_df[col] for col in key_cols]))
                new_mask = [key not in existing_keys for key in new_keys]
                defense_df = defense_df[new_mask]
                if len(defense_df) > 0:
                    defense_df = pd.concat([existing, defense_df], ignore_index=True)
                else:
                    defense_df = existing
            defense_df = enforce_deterministic_aggregation(defense_df, ['season', 'player_id'] if 'player_id' in defense_df.columns else ['season', 'player_name'])
            defense_df.to_csv(defense_file, index=False)
            print(f"Saved defense to {defense_file}")
    elif args.data_type == 'games':
        df = aggregate_game_results(args.start_year, args.end_year)
        filename = data_config.historical_data_dir / "aggregated" / "game_results_1966_present.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    elif args.data_type == 'pbp':
        df = aggregate_pbp_data(args.start_year, args.end_year)
        filename = data_config.historical_data_dir / "aggregated" / "pbp_data_1999_present.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")

