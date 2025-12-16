"""
Fetch current/live NFL statistics for comparison with predictions.

This script provides multiple methods to fetch NFL statistics:
1. nfl-data-py (recommended)
2. Manual CSV upload
3. Web scraping (Pro Football Reference)
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_with_nfldatapy(season: int = None) -> Optional[pd.DataFrame]:
    """
    Fetch NFL stats using nfl-data-py package (weekly data).
    
    Parameters:
    -----------
    season : int, optional
        Season year. Defaults to current season.
        
    Returns:
    --------
    pd.DataFrame : Weekly player statistics
    """
    try:
        import nfl_data_py as nfl
        
        if season is None:
            season = datetime.now().year
            if datetime.now().month < 9:
                season -= 1
        
        logger.info(f"Fetching NFL weekly stats for season {season} using nfl-data-py...")
        
        # Fetch weekly data
        weekly_data = nfl.import_weekly_data([season])
        
        if weekly_data is not None and not weekly_data.empty:
            logger.info(f"Successfully fetched {len(weekly_data)} weekly records")
            return weekly_data
        else:
            logger.warning("No data returned from nfl-data-py")
            return None
            
    except ImportError:
        logger.error("nfl-data-py not installed. Install with: pip install nfl-data-py")
        return None
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def fetch_season_stats(season: int = None, try_previous_seasons: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch season-level player statistics.
    
    Parameters:
    -----------
    season : int, optional
        Season year. If None, uses most recent complete season.
    try_previous_seasons : bool
        If True, try previous seasons if current season fails
        
    Returns:
    --------
    pd.DataFrame : Season-level player statistics
    """
    try:
        import nfl_data_py as nfl
        
        if season is None:
            # Default to most recent complete season
            # NFL season typically ends in February, so if we're before September, use previous year
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            # If before September, previous season is complete
            # If after September but before February, current season might be incomplete
            if current_month < 9:
                season = current_year - 1
            elif current_month >= 2:  # After Super Bowl, previous season is complete
                season = current_year - 1
            else:  # September - January, current season is ongoing
                season = current_year - 1  # Use previous complete season
        
        seasons_to_try = [season]
        if try_previous_seasons:
            # Add previous seasons as fallback
            for i in range(1, 3):  # Try up to 2 previous seasons
                seasons_to_try.append(season - i)
        
        weekly_data = None
        successful_season = None
        
        for attempt_season in seasons_to_try:
            try:
                logger.info(f"Attempting to fetch season stats for {attempt_season}...")
                
                # Fetch weekly data and aggregate to season level
                weekly_data = nfl.import_weekly_data([attempt_season])
                
                if weekly_data is None:
                    logger.warning(f"No weekly data returned for {attempt_season}")
                    weekly_data = None
                    continue
                
                if weekly_data.empty:
                    logger.warning(f"Empty weekly data returned for {attempt_season}")
                    weekly_data = None
                    continue
                
                logger.info(f"Successfully fetched {len(weekly_data)} weekly records for {attempt_season}")
                logger.info(f"Weekly data columns: {list(weekly_data.columns)[:15]}...")
                successful_season = attempt_season
                break
                
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "Not Found" in error_msg:
                    logger.warning(f"Season {attempt_season} not available (404). Trying previous season...")
                    continue
                else:
                    logger.warning(f"Error fetching {attempt_season}: {e}")
                    continue
        
        if weekly_data is None or weekly_data.empty:
            logger.error(f"Could not fetch data for any of the attempted seasons: {seasons_to_try}")
            return None
        
        season = successful_season  # Use the successful season
        
        # Aggregate to season level by player
        # Group by player and sum stats
        agg_columns = {
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum',
            'receptions': 'sum',
            'fantasy_points': 'sum',
            'fantasy_points_ppr': 'sum',
        }
        
        # Find available columns to aggregate
        available_agg = {k: v for k, v in agg_columns.items() if k in weekly_data.columns}
        
        # Group by player (use player_id or player_name if available)
        # But we need to preserve team information - use the most common team for each player
        group_by_cols = []
        if 'player_id' in weekly_data.columns:
            group_by_cols.append('player_id')
        if 'player_name' in weekly_data.columns:
            group_by_cols.append('player_name')
        if 'position' in weekly_data.columns:
            group_by_cols.append('position')
        
        if not group_by_cols:
            logger.warning("No grouping columns found, returning weekly data as-is")
            return weekly_data
        
        # Find team column (could be 'team', 'recent_team', 'team_abbr', etc.)
        team_col = None
        for col in ['recent_team', 'team', 'team_abbr', 'team_name']:
            if col in weekly_data.columns:
                team_col = col
                logger.info(f"Found team column: '{team_col}'")
                break
        
        if not team_col:
            logger.warning("No team column found in weekly data. Available columns:")
            logger.warning(f"  {list(weekly_data.columns)[:20]}...")
        
        # Aggregate stats
        season_stats = weekly_data.groupby(group_by_cols, as_index=False).agg(available_agg)
        
        # Add team information - use mode (most common team) for each player
        if team_col:
            # Get most common team for each player
            def get_mode(x):
                mode_vals = x.mode()
                if len(mode_vals) > 0:
                    return mode_vals.iloc[0]
                else:
                    return x.iloc[0] if len(x) > 0 else None
            
            team_mode = weekly_data.groupby(group_by_cols)[team_col].agg(get_mode).reset_index()
            team_mode.columns = group_by_cols + ['team']
            
            season_stats = season_stats.merge(
                team_mode,
                on=group_by_cols,
                how='left'
            )
            logger.info(f"Added team information from '{team_col}' column")
        else:
            logger.warning("No team column found in weekly data - team information will be missing")
        
        logger.info(f"Aggregated to {len(season_stats)} player records")
        return season_stats
            
    except ImportError:
        logger.error("nfl-data-py not installed. Install with: pip install nfl-data-py")
        return None
    except Exception as e:
        logger.error(f"Error fetching season stats: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def calculate_fantasy_points_standard(df: pd.DataFrame, position: str = None) -> pd.DataFrame:
    """
    Calculate standard fantasy points from NFL stats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with NFL statistics
    position : str, optional
        Position filter (QB, RB, WR, etc.)
        
    Returns:
    --------
    pd.DataFrame : DataFrame with fantasy_points column added
    """
    df = df.copy()
    
    # Filter by position if specified
    if position and 'position' in df.columns:
        df = df[df['position'] == position.upper()]
    
    # Calculate fantasy points based on position
    if 'position' in df.columns:
        # QB scoring
        qb_mask = df['position'] == 'QB'
        if qb_mask.any():
            df.loc[qb_mask, 'fantasy_points_standard'] = (
                (df.loc[qb_mask, 'passing_tds'].fillna(0) * 4) +
                (df.loc[qb_mask, 'passing_yards'].fillna(0) / 25) +
                (df.loc[qb_mask, 'interceptions'].fillna(0) * -2) +
                (df.loc[qb_mask, 'rushing_yards'].fillna(0) / 10) +
                (df.loc[qb_mask, 'rushing_tds'].fillna(0) * 6)
            )
        
        # RB/WR/TE scoring
        skill_positions = ['RB', 'WR', 'TE']
        skill_mask = df['position'].isin(skill_positions)
        if skill_mask.any():
            df.loc[skill_mask, 'fantasy_points_standard'] = (
                (df.loc[skill_mask, 'rushing_yards'].fillna(0) / 10) +
                (df.loc[skill_mask, 'receiving_yards'].fillna(0) / 10) +
                (df.loc[skill_mask, 'rushing_tds'].fillna(0) * 6) +
                (df.loc[skill_mask, 'receiving_tds'].fillna(0) * 6) +
                (df.loc[skill_mask, 'receptions'].fillna(0) * 0)  # Standard scoring (PPR would be * 1)
            )
    else:
        # Generic calculation if no position column
        df['fantasy_points_standard'] = (
            (df.get('rushing_yards', pd.Series(0)) / 10) +
            (df.get('receiving_yards', pd.Series(0)) / 10) +
            (df.get('rushing_tds', pd.Series(0)) * 6) +
            (df.get('receiving_tds', pd.Series(0)) * 6)
        )
    
    return df


def save_live_stats(df: pd.DataFrame, season: int = None, output_dir: str = "data") -> Path:
    """
    Save fetched live stats to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Statistics DataFrame
    season : int, optional
        Season year. If None, tries to infer from DataFrame.
    output_dir : str
        Output directory
        
    Returns:
    --------
    Path : Path to saved file
    """
    if season is None:
        # Try to get season from DataFrame
        if 'season' in df.columns:
            season = int(df['season'].iloc[0])
        else:
            # Default to previous year
            season = datetime.now().year - 1
    
    # Use data directory (not data/raw or data/processed) for live stats
    if output_dir == "data":
        output_path = Path("data")
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"live_nfl_stats_{season}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Saved live stats to {filename}")
    
    return filename


def main():
    """Main function to fetch and save live NFL stats."""
    logger.info("Fetching live NFL statistics...")
    
    # Fetch stats (will automatically use most recent complete season)
    stats = fetch_season_stats(season=None, try_previous_seasons=True)
    
    if stats is not None:
        # Get the season that was actually fetched
        fetched_season = stats['season'].iloc[0] if 'season' in stats.columns else None
        if fetched_season is None:
            # Try to infer from data or use current year - 1
            fetched_season = datetime.now().year - 1
        
        # Calculate fantasy points
        stats = calculate_fantasy_points_standard(stats)
        
        # Save to file
        save_live_stats(stats, fetched_season)
        
        # Print summary
        print(f"\nLive NFL Stats Summary (Season {fetched_season}):")
        print(f"  Total players: {len(stats)}")
        print(f"  Columns: {list(stats.columns)[:10]}...")  # Show first 10 columns
        
        if 'position' in stats.columns:
            pos_counts = stats['position'].value_counts().to_dict()
            print(f"  Positions: {pos_counts}")
        
        # Check for fantasy points column (might be named differently)
        fp_cols = [col for col in stats.columns if 'fantasy' in col.lower() or 'fp' in col.lower()]
        if fp_cols:
            fp_col = fp_cols[0]
            print(f"  Avg {fp_col}: {stats[fp_col].mean():.2f}")
            print(f"  Top 10 players by {fp_col}:")
            
            # Get player name column
            name_cols = [col for col in stats.columns if 'name' in col.lower() or 'player' in col.lower()]
            if name_cols:
                display_cols = [name_cols[0]]
                if 'position' in stats.columns:
                    display_cols.append('position')
                display_cols.append(fp_col)
                top_10 = stats.nlargest(10, fp_col)[display_cols]
                print(top_10.to_string(index=False))
            else:
                top_10 = stats.nlargest(10, fp_col)
                print(top_10[[c for c in top_10.columns if c != fp_col][:2] + [fp_col]].to_string(index=False))
        elif 'fantasy_points_standard' in stats.columns:
            print(f"  Avg fantasy points: {stats['fantasy_points_standard'].mean():.2f}")
            print(f"  Top 10 players:")
            name_cols = [col for col in stats.columns if 'name' in col.lower()]
            if name_cols:
                display_cols = [name_cols[0]]
                if 'position' in stats.columns:
                    display_cols.append('position')
                display_cols.append('fantasy_points_standard')
                top_10 = stats.nlargest(10, 'fantasy_points_standard')[display_cols]
                print(top_10.to_string(index=False))
    else:
        logger.error("Could not fetch live stats. Check your internet connection and nfl-data-py installation.")


def test_nfldatapy_installation():
    """Test if nfl-data-py is installed and working."""
    try:
        import nfl_data_py as nfl
        print("✅ nfl-data-py is installed")
        
        # Check available methods
        print("\nAvailable methods in nfl_data_py:")
        methods = [m for m in dir(nfl) if not m.startswith('_') and callable(getattr(nfl, m))]
        for method in methods[:10]:  # Show first 10
            print(f"  - {method}")
        
        # Try to fetch a small sample
        print("\nTesting data fetch...")
        try:
            test_data = nfl.import_weekly_data([2024])
            if test_data is not None and not test_data.empty:
                print(f"✅ Successfully fetched {len(test_data)} rows")
                print(f"   Columns: {list(test_data.columns)[:10]}...")
                return True
            else:
                print("⚠️  No data returned")
                return False
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
            
    except ImportError:
        print("❌ nfl-data-py is NOT installed")
        print("   Install with: pip install nfl-data-py")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if user wants to test installation
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_nfldatapy_installation()
    else:
        main()

