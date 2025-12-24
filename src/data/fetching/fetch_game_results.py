"""
Fetch game-by-game NFL results for tiebreaker calculations
Supports both nfl-data-py and historical PFR data
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


def fetch_game_results(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch game-by-game results for a season.
    
    Returns DataFrame with columns:
    - team, opponent, home_away, result (W/L/T), score, opponent_score
    - division_game, conference_game
    """
    try:
        import nfl_data_py as nfl
        
        # Fetch schedule data
        schedule = nfl.import_schedules([season])
        
        if schedule.empty:
            logger.warning(f"No schedule data for season {season}")
            return None
        
        # CRITICAL: Filter to regular season only (exclude playoffs)
        # Playoff games should not be used for seeding calculations
        if 'game_type' in schedule.columns:
            schedule = schedule[schedule['game_type'] == 'REG'].copy()
            logger.debug(f"Filtered to regular season games using game_type column")
        elif 'season_type' in schedule.columns:
            schedule = schedule[schedule['season_type'] == 'REG'].copy()
            logger.debug(f"Filtered to regular season games using season_type column")
        elif 'week' in schedule.columns:
            # Regular season is typically weeks 1-18 (or 1-17 for older seasons)
            max_reg_season_week = 18 if season >= 2021 else 17
            schedule = schedule[schedule['week'] <= max_reg_season_week].copy()
            logger.debug(f"Filtered to regular season games using week <= {max_reg_season_week}")
        else:
            logger.warning("No game_type, season_type, or week column found. Cannot filter playoffs!")
        
        if schedule.empty:
            logger.warning(f"No regular season games found after filtering")
            return None
        
        logger.info(f"Processing {len(schedule)} regular season games")
        
        # Process into game-by-game format
        games = []
        
        for _, row in schedule.iterrows():
            home_team = row.get('home_team')
            away_team = row.get('away_team')
            home_score = row.get('home_score', 0)
            away_score = row.get('away_score', 0)
            
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Determine result
            if home_score > away_score:
                home_result = 'W'
                away_result = 'L'
            elif home_score < away_score:
                home_result = 'L'
                away_result = 'W'
            else:
                home_result = 'T'
                away_result = 'T'
            
            # Home team perspective
            games.append({
                'team': home_team,
                'opponent': away_team,
                'home_away': 'home',
                'result': home_result,
                'score': home_score,
                'opponent_score': away_score,
                'season': season,
                'week': row.get('week', None),
                'game_id': row.get('game_id', None)
            })
            
            # Away team perspective
            games.append({
                'team': away_team,
                'opponent': home_team,
                'home_away': 'away',
                'result': away_result,
                'score': away_score,
                'opponent_score': home_score,
                'season': season,
                'week': row.get('week', None),
                'game_id': row.get('game_id', None)
            })
        
        df = pd.DataFrame(games)
        
        if df.empty:
            logger.warning(f"No game results calculated for season {season}")
            return None
        
        # Add division/conference info (simplified - would need team metadata)
        # For now, mark as unknown - can be enhanced later
        df['division_game'] = False  # Would need division info
        df['conference_game'] = False  # Would need conference info
        
        logger.info(f"Fetched {len(df)} game records for season {season}")
        return df
        
    except ImportError:
        logger.warning("nfl-data-py not available, cannot fetch game results")
        return None
    except Exception as e:
        logger.warning(f"Error fetching game results: {e}")
        return None


def calculate_head_to_head(teams: list, game_data: pd.DataFrame) -> dict:
    """
    Calculate head-to-head records between teams.
    
    Returns dict mapping (team1, team2) -> (wins, losses, ties)
    """
    h2h = {}
    
    for team1 in teams:
        for team2 in teams:
            if team1 >= team2:  # Avoid duplicates
                continue
            
            # Games where these teams played
            games = game_data[
                ((game_data['team'] == team1) & (game_data['opponent'] == team2)) |
                ((game_data['team'] == team2) & (game_data['opponent'] == team1))
            ]
            
            if games.empty:
                continue
            
            # Count wins for team1
            team1_wins = len(games[
                ((games['team'] == team1) & (games['result'] == 'W')) |
                ((games['team'] == team2) & (games['result'] == 'L'))
            ])
            
            team2_wins = len(games[
                ((games['team'] == team2) & (games['result'] == 'W')) |
                ((games['team'] == team1) & (games['result'] == 'L'))
            ])
            
            ties = len(games[games['result'] == 'T'])
            
            h2h[(team1, team2)] = {
                team1: {'wins': team1_wins, 'losses': team2_wins, 'ties': ties},
                team2: {'wins': team2_wins, 'losses': team1_wins, 'ties': ties}
            }
    
    return h2h


def fetch_game_results_historical(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch game results from historical PFR data.
    
    Parameters:
    -----------
    season : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Game results, or None if not found
    """
    # Try to load from aggregated historical data
    aggregated_file = data_config.historical_data_dir / "aggregated" / "game_results_1966_present.csv"
    
    if aggregated_file.exists():
        try:
            df = pd.read_csv(aggregated_file)
            season_data = df[df['season'] == season].copy()
            
            if len(season_data) > 0:
                # Convert to the format expected by calculate_head_to_head
                # PFR game data may need transformation
                logger.info(f"Loaded historical game results for season {season}")
                return season_data
        except Exception as e:
            logger.warning(f"Error loading historical game data: {e}")
    
    # Try to load from individual year file
    pfr_file = data_config.historical_data_dir / "pfr" / "game_results" / f"game_results_{season}.csv"
    if pfr_file.exists():
        try:
            df = pd.read_csv(pfr_file)
            logger.info(f"Loaded PFR game results for season {season}")
            return df
        except Exception as e:
            logger.warning(f"Error loading PFR file: {e}")
    
    return None


def fetch_game_results_with_fallback(season: int) -> Optional[pd.DataFrame]:
    """
    Fetch game results with fallback to historical data if nfl-data-py fails.
    
    Parameters:
    -----------
    season : int
        Season year
        
    Returns:
    --------
    pd.DataFrame : Game results
    """
    # Try nfl-data-py first (for recent years)
    games = fetch_game_results(season)
    
    if games is not None:
        return games
    
    # Fallback to historical PFR data
    logger.info(f"Falling back to historical PFR data for season {season}")
    return fetch_game_results_historical(season)


def main():
    """Test fetching game results"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NFL game results')
    parser.add_argument('--season', type=int, default=2023)
    parser.add_argument('--output', type=str, default='data/processed/game_results.csv')
    parser.add_argument('--use-historical', action='store_true',
                       help='Use historical PFR data instead of nfl-data-py')
    
    args = parser.parse_args()
    
    if args.use_historical:
        games = fetch_game_results_historical(args.season)
    else:
        games = fetch_game_results_with_fallback(args.season)
    
    if games is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_path, index=False)
        print(f"\nGame results saved to {output_path}")
        print(f"Total games: {len(games) // 2 if 'team' in games.columns else len(games)}")
        print(f"\nSample games:")
        print(games.head(10))
        
        # Test head-to-head calculation
        if len(games) > 0 and 'team' in games.columns:
            sample_teams = games['team'].unique()[:4]
            h2h = calculate_head_to_head(list(sample_teams), games)
            print(f"\nHead-to-head sample (first 4 teams):")
            for (t1, t2), records in list(h2h.items())[:3]:
                print(f"  {t1} vs {t2}: {records}")
    else:
        print("Failed to fetch game results")


if __name__ == '__main__':
    main()

