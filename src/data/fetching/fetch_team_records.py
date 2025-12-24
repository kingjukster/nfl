"""
Fetch actual NFL team win/loss records using nfl-data-py and PFR historical data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import data_config

logger = logging.getLogger(__name__)


def fetch_team_records(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch actual team win/loss records for a season.
    
    Returns DataFrame with columns: team, win, loss, tie, win_pct
    """
    try:
        import nfl_data_py as nfl
        
        # Fetch schedule data
        schedule = nfl.import_schedules([season])
        
        if schedule.empty:
            logger.warning(f"No schedule data for season {season}")
            return None
        
        # CRITICAL: Filter to regular season only (exclude playoffs)
        # Playoff games would give away the Super Bowl winner
        if 'game_type' in schedule.columns:
            schedule = schedule[schedule['game_type'] == 'REG'].copy()
            logger.debug(f"Filtered to regular season games using game_type column")
        elif 'season_type' in schedule.columns:
            schedule = schedule[schedule['season_type'] == 'REG'].copy()
            logger.debug(f"Filtered to regular season games using season_type column")
        elif 'week' in schedule.columns:
            # Regular season is typically weeks 1-18 (or 1-17 for older seasons)
            # Playoffs start at week 19+
            max_reg_season_week = 18 if season >= 2021 else 17
            schedule = schedule[schedule['week'] <= max_reg_season_week].copy()
            logger.debug(f"Filtered to regular season games using week <= {max_reg_season_week}")
        else:
            logger.warning("No game_type, season_type, or week column found. Cannot filter playoffs!")
            logger.warning("Records may include playoff games, which will affect seeding accuracy.")
        
        if schedule.empty:
            logger.warning(f"No regular season games found after filtering")
            return None
        
        logger.info(f"Using {len(schedule)} regular season games for record calculation")
        
        # Calculate team records
        team_records = []
        
        for team in schedule['home_team'].unique():
            if pd.isna(team):
                continue
            
            # Home games
            home_games = schedule[schedule['home_team'] == team]
            home_wins = (home_games['home_score'] > home_games['away_score']).sum()
            home_losses = (home_games['home_score'] < home_games['away_score']).sum()
            home_ties = (home_games['home_score'] == home_games['away_score']).sum()
            
            # Away games
            away_games = schedule[schedule['away_team'] == team]
            away_wins = (away_games['away_score'] > away_games['home_score']).sum()
            away_losses = (away_games['away_score'] < away_games['home_score']).sum()
            away_ties = (away_games['away_score'] == away_games['home_score']).sum()
            
            # Total
            wins = home_wins + away_wins
            losses = home_losses + away_losses
            ties = home_ties + away_ties
            total_games = wins + losses + ties
            
            if total_games > 0:
                win_pct = (wins + 0.5 * ties) / total_games
            else:
                win_pct = 0.0
            
            team_records.append({
                'team': team,
                'win': wins,
                'loss': losses,
                'tie': ties,
                'win_pct': win_pct,
                'season': season
            })
        
        df = pd.DataFrame(team_records)
        
        if df.empty:
            logger.warning(f"No team records calculated for season {season}")
            return None
        
        logger.info(f"Fetched records for {len(df)} teams in season {season}")
        return df
        
    except ImportError:
        logger.warning("nfl-data-py not available, cannot fetch team records")
        return None
    except Exception as e:
        logger.warning(f"Error fetching team records: {e}")
        return None


def fetch_team_records_historical(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch team records from historical PFR data.
    
    Parameters:
    -----------
    season : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Team records, or None if not found
    """
    # Try to load from aggregated historical data
    aggregated_file = data_config.historical_data_dir / "aggregated" / "team_season_stats_1966_present.csv"
    
    if aggregated_file.exists():
        try:
            df = pd.read_csv(aggregated_file)
            season_data = df[df['season'] == season].copy()
            
            if len(season_data) > 0:
                # Extract win/loss records if available
                if 'wins' in season_data.columns and 'losses' in season_data.columns:
                    result = season_data[['team', 'wins', 'losses']].copy()
                    result.rename(columns={'wins': 'win', 'losses': 'loss'}, inplace=True)
                    
                    # Add ties if available
                    if 'ties' in season_data.columns:
                        result['tie'] = season_data['ties']
                    else:
                        result['tie'] = 0
                    
                    # Calculate win_pct
                    total_games = result['win'] + result['loss'] + result['tie']
                    result['win_pct'] = (result['win'] + 0.5 * result['tie']) / total_games
                    result['season'] = season
                    
                    logger.info(f"Loaded historical records for {len(result)} teams in season {season}")
                    return result
        except Exception as e:
            logger.warning(f"Error loading historical data: {e}")
    
    # Try to load from individual year file
    pfr_file = data_config.historical_data_dir / "pfr" / "team_stats" / f"team_stats_{season}.csv"
    if pfr_file.exists():
        try:
            df = pd.read_csv(pfr_file)
            # Similar extraction logic as above
            if 'wins' in df.columns and 'losses' in df.columns:
                result = df[['team', 'wins', 'losses']].copy()
                result.rename(columns={'wins': 'win', 'losses': 'loss'}, inplace=True)
                if 'ties' in df.columns:
                    result['tie'] = df['ties']
                else:
                    result['tie'] = 0
                total_games = result['win'] + result['loss'] + result['tie']
                result['win_pct'] = (result['win'] + 0.5 * result['tie']) / total_games
                result['season'] = season
                return result
        except Exception as e:
            logger.warning(f"Error loading PFR file: {e}")
    
    return None


def fetch_team_records_with_fallback(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch team records with fallback to historical data if nfl-data-py fails.
    
    Parameters:
    -----------
    season : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Team records
    """
    # Try nfl-data-py first (for recent years)
    records = fetch_team_records(season)
    
    if records is not None:
        return records
    
    # Fallback to historical PFR data
    logger.info(f"Falling back to historical PFR data for season {season}")
    return fetch_team_records_historical(season)


def main():
    """Test fetching team records"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NFL team records')
    parser.add_argument('--season', type=int, default=2023)
    parser.add_argument('--output', type=str, default='data/processed/team_records.csv')
    parser.add_argument('--use-historical', action='store_true',
                       help='Use historical PFR data instead of nfl-data-py')
    
    args = parser.parse_args()
    
    if args.use_historical:
        records = fetch_team_records_historical(args.season)
    else:
        records = fetch_team_records_with_fallback(args.season)
    
    if records is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(output_path, index=False)
        print(f"\nTeam records saved to {output_path}")
        print(f"\nSample records:")
        print(records.head(10))
    else:
        print("Failed to fetch team records")


if __name__ == '__main__':
    main()

